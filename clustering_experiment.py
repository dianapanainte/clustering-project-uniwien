from pathlib import Path
import pandas as pd
import numpy as np
from clustering_module.pipeline import ClusteringPipeline
from clustering_module.clustering import KMeansPlusClustering, TwoStageClusteringStrategy, HDBSCANClustering
from clustering_module.features import (
    ModularFeatureExtraction, VolumeFeatures, CalendarFeatures,
    SeasonalFeatures, TrendFeatures, PCAFeatures
)
from clustering_module.preprocessing import NoImputation, RollingOutlierHandling, ShapeScaling
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from datetime import datetime

CSV_FILE = 'sample_23.csv'

def load_data(file_name: str, folder: str = "data") -> pd.DataFrame:
    path = Path(folder) / file_name
    df = pd.read_csv(path, index_col=0)
    df = df.dropna(how='all')
    df = df[df.sum(axis=1) > 1]
    print(f"Loaded {df.shape[0]} households, {df.shape[1]} days")
    return df

def compute_raw_profile(cluster_df):
    """Compute interpretable stats from raw consumption data."""
    raw = cluster_df
    dates = pd.to_datetime(raw.columns)
    is_weekend = dates.weekday >= 5
    step = raw.shape[1] // 4
    return {
        'avg_usage':    raw.mean(axis=1).mean(),
        'variability':  (raw.std(axis=1) / (raw.mean(axis=1) + 1e-9)).mean(),
        'peak':         (raw.max(axis=1) / (raw.mean(axis=1) + 1e-9)).mean(),
        'weekend_bias': raw.iloc[:, is_weekend].mean(axis=1).mean() / (raw.iloc[:, ~is_weekend].mean(axis=1).mean() + 1e-9),
        'trend':        (raw.iloc[:, -30:].mean(axis=1) - raw.iloc[:, :30].mean(axis=1)).mean(),
        'winter':       raw.iloc[:, 0:step].mean(axis=1).mean(),
        'summer':       raw.iloc[:, 2*step:3*step].mean(axis=1).mean(),
    }

# --- STEP 1: Extract features ONCE ---
data = load_data(CSV_FILE)

pipeline = ClusteringPipeline(
    feature_strategy=ModularFeatureExtraction(blocks=[
        VolumeFeatures(),
        CalendarFeatures(),
        SeasonalFeatures(n_segments=4),
        TrendFeatures(window=30),
        PCAFeatures(n_components=3),
    ]),
    zero_strategy=NoImputation(),
    outlier_strategy=RollingOutlierHandling(k=3, window=7),
    norm_strategy=ShapeScaling(),
    cluster_strategy=KMeansPlusClustering(n_clusters=2)
)

features = pipeline.extract_features(data)

nan_counts = features.isnull().sum()
print(nan_counts[nan_counts > 0] if nan_counts.any() else "No NaNs found")
print(f"Feature matrix shape: {features.shape}")
print(f"Features: {list(features.columns)}")

# --- STEP 2: Run all strategies ONCE and cache results ---
strategies = {
    'kmeans_k2': KMeansPlusClustering(n_clusters=2),
    'kmeans_k3': KMeansPlusClustering(n_clusters=3),
    'kmeans_k4': KMeansPlusClustering(n_clusters=4),
    'kmeans_k5': KMeansPlusClustering(n_clusters=5),
    'two_stage_k3_tight': TwoStageClusteringStrategy(
                        first_stage=KMeansPlusClustering(n_clusters=3),
                        second_stage=HDBSCANClustering(min_cluster_size=200)
                    ),
    'two_stage_k3_loose': TwoStageClusteringStrategy(
                        first_stage=KMeansPlusClustering(n_clusters=3),
                        second_stage=HDBSCANClustering(min_cluster_size=500)
                    ),
    'two_stage_k4_tight': TwoStageClusteringStrategy(
                        first_stage=KMeansPlusClustering(n_clusters=4),
                        second_stage=HDBSCANClustering(min_cluster_size=200)
                    ),
    'two_stage_k4_loose': TwoStageClusteringStrategy(
                        first_stage=KMeansPlusClustering(n_clusters=4),
                        second_stage=HDBSCANClustering(min_cluster_size=500)
                    ),
}

print("\n--- Clustering Comparison ---")
cached_results = {}
skipped_results = {}

for name, strategy in strategies.items():
    labels = strategy.fit_predict(features)
    n_clusters = len(set(labels)) - (1 if -1 in set(labels) else 0)
    n_noise = int((labels == -1).sum())

    if n_clusters < 2:
        print(f"\n{name}: skipped — only {n_clusters} valid cluster(s), {n_noise} noise points")
        skipped_results[name] = {
            'labels': labels,
            'n_clusters': n_clusters,
            'n_noise': n_noise
        }
        continue

    sil = silhouette_score(features, labels)
    db  = davies_bouldin_score(features, labels)
    ch  = calinski_harabasz_score(features, labels)
    cached_results[name] = {
        'labels': labels,
        'n_clusters': n_clusters,
        'n_noise': n_noise,
        'silhouette': sil,
        'davies_bouldin': db,
        'calinski_harabasz': ch
    }
    print(f"\n{name}:")
    print(f"  Clusters: {n_clusters}, Noise points: {n_noise}")
    print(f"  Silhouette:        {sil:.4f}")
    print(f"  Davies-Bouldin:    {db:.4f}")
    print(f"  Calinski-Harabasz: {ch:.4f}")

    results_df = data.copy()
    results_df['cluster'] = labels
    for cluster_id in sorted(set(labels)):
        if cluster_id == -1:
            continue
        cluster = results_df[results_df['cluster'] == cluster_id]
        p = compute_raw_profile(cluster.drop(columns='cluster'))
        print(f"  Cluster {cluster_id} (n={len(cluster)}): "
              f"avg={p['avg_usage']:.2f}, "
              f"var={p['variability']:.2f}, "
              f"peak={p['peak']:.2f}, "
              f"weekend_bias={p['weekend_bias']:.2f}, "
              f"trend={p['trend']:.2f}, "
              f"winter={p['winter']:.2f}, "
              f"summer={p['summer']:.2f}")

# print skipped summary
if skipped_results:
    print("\n--- Skipped Strategies ---")
    for name, r in skipped_results.items():
        print(f"  {name}: {r['n_clusters']} valid cluster(s), {r['n_noise']} noise points")

# --- STEP 3: Elbow + Silhouette plot ---
inertias = []
silhouettes = []
k_range = range(2, 10)

for k in k_range:
    model = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
    k_labels = model.fit_predict(features)
    inertias.append(model.inertia_)
    silhouettes.append(silhouette_score(features, k_labels))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(k_range, inertias, 'bo-')
ax1.set_xlabel('k')
ax1.set_title('Elbow Plot — pick where the curve bends')
ax2.plot(k_range, silhouettes, 'ro-')
ax2.set_xlabel('k')
ax2.set_title('Silhouette Score — pick the peak')
plt.tight_layout()
plt.savefig('plots/elbow_plot.png')
print("\nElbow plot saved as plots/elbow_plot.png")

# --- STEP 4: Cluster Profiles for ALL successful strategies ---
print("\n--- Cluster Profiles ---")
fig_profiles, axes = plt.subplots(len(cached_results), 1,
                                   figsize=(14, 5 * len(cached_results)))

if len(cached_results) == 1:
    axes = [axes]

for idx, (name, r) in enumerate(cached_results.items()):
    results_df = data.copy()
    results_df['cluster'] = r['labels']
    ax = axes[idx]

    for cluster_id in sorted(set(r['labels'])):
        if cluster_id == -1:
            continue
        cluster = results_df[results_df['cluster'] == cluster_id]
        mean_profile = cluster.drop(columns='cluster').mean(axis=0)
        ax.plot(mean_profile.values, label=f'Cluster {cluster_id} (n={len(cluster)})')

    ax.set_title(f'{name} — Silhouette: {r["silhouette"]:.4f}, DB: {r["davies_bouldin"]:.4f}')
    ax.set_xlabel('Day of year')
    ax.set_ylabel('Consumption')
    ax.legend()

plt.tight_layout()
plt.savefig('plots/cluster_profiles.png')
print("Cluster profiles saved as plots/cluster_profiles.png")

# --- STEP 5: Write markdown report ---
report_path = Path('results.md')
with open(report_path, 'w') as f:
    f.write(f"# Clustering Experiment Report\n")
    f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
    f.write(f"**Dataset:** {CSV_FILE} — {data.shape[0]} households, {data.shape[1]} days\n\n")

    f.write(f"## Feature Blocks Used\n")
    for block in pipeline.feature_strategy.blocks:
        f.write(f"- {block.__class__.__name__}\n")
    f.write(f"\n**Feature matrix shape:** {features.shape[0]} rows x {features.shape[1]} columns\n")
    f.write(f"**Features:** {', '.join(features.columns)}\n\n")

    f.write(f"## Clustering Results\n\n")
    f.write(f"| Method | Clusters | Noise | Silhouette | Davies-Bouldin | Calinski-Harabasz |\n")
    f.write(f"|--------|----------|-------|--------------|------------------|---------------------|\n")
    for name, r in cached_results.items():
        f.write(f"| {name} | {r['n_clusters']} | {r['n_noise']} | {r['silhouette']:.4f} | {r['davies_bouldin']:.4f} | {r['calinski_harabasz']:.4f} |\n")

    f.write(f"\n## Elbow Plot\n")
    f.write(f"![Elbow Plot](plots/elbow_plot.png)\n\n")

    f.write(f"## Cluster Profiles (all strategies)\n\n")
    f.write(f"![Cluster Profiles](plots/cluster_profiles.png)\n\n")

    for name, r in cached_results.items():
        f.write(f"### {name}\n\n")
        results_df = data.copy()
        results_df['cluster'] = r['labels']
        f.write(f"| Cluster | Size | Avg Usage | Variability | Peak | Weekend Bias | Trend | Winter | Summer |\n")
        f.write(f"|---------|------|-----------|-------------|------|--------------|-------|--------|--------|\n")
        for cluster_id in sorted(set(r['labels'])):
            if cluster_id == -1:
                continue
            cluster = results_df[results_df['cluster'] == cluster_id]
            p = compute_raw_profile(cluster.drop(columns='cluster'))
            f.write(f"| {cluster_id} "
                    f"| {len(cluster)} "
                    f"| {p['avg_usage']:.4f} "
                    f"| {p['variability']:.4f} "
                    f"| {p['peak']:.4f} "
                    f"| {p['weekend_bias']:.4f} "
                    f"| {p['trend']:.4f} "
                    f"| {p['winter']:.4f} "
                    f"| {p['summer']:.4f} |\n")
        f.write(f"\n")

    if skipped_results:
        f.write(f"## Skipped Strategies\n\n")
        f.write(f"| Method | Valid Clusters | Noise Points | Reason |\n")
        f.write(f"|--------|---------------|--------------|--------|\n")
        for name, r in skipped_results.items():
            f.write(f"| {name} | {r['n_clusters']} | {r['n_noise']} | All points classified as noise |\n")
        f.write(f"\n")
        f.write(f"### Skipped Strategy Cluster Profiles\n\n")
        for name, r in skipped_results.items():
            f.write(f"#### {name}\n\n")
            results_df = data.copy()
            results_df['cluster'] = r['labels']
            unique_labels = sorted(set(r['labels']))
            if len(unique_labels) == 1 and -1 in unique_labels:
                f.write(f"All {r['n_noise']} points classified as noise — no cluster profiles available.\n\n")
                continue
            f.write(f"| Cluster | Size | Avg Usage | Variability | Peak | Weekend Bias | Trend | Winter | Summer |\n")
            f.write(f"|---------|------|-----------|-------------|------|--------------|-------|--------|--------|\n")
            for cluster_id in unique_labels:
                if cluster_id == -1:
                    continue
                cluster = results_df[results_df['cluster'] == cluster_id]
                p = compute_raw_profile(cluster.drop(columns='cluster'))
                f.write(f"| {cluster_id} "
                        f"| {len(cluster)} "
                        f"| {p['avg_usage']:.4f} "
                        f"| {p['variability']:.4f} "
                        f"| {p['peak']:.4f} "
                        f"| {p['weekend_bias']:.4f} "
                        f"| {p['trend']:.4f} "
                        f"| {p['winter']:.4f} "
                        f"| {p['summer']:.4f} |\n")
            f.write(f"\n")

    f.write(f"## Notes\n")
    f.write(f"- \n")

print(f"Report saved to {report_path}")