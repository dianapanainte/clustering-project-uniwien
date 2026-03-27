"""
Clustering script for KDD project:
"Leveraging Clustering for Large-Scale Time Series Forecasting"

Applies multiple clustering algorithms to household electricity consumption
data preprocessed in preprocessing.py. Evaluates cluster quality and generates
comprehensive visualizations.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import warnings
import json
import time

warnings.filterwarnings('ignore')

# ── Configuration ──────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).parent / "data"
PREP_DIR = DATA_DIR / "preprocessed"
PLOT_DIR = Path(__file__).parent / "plots"
OUT_DIR  = Path(__file__).parent / "results"

PLOT_DIR.mkdir(exist_ok=True)
OUT_DIR.mkdir(exist_ok=True)

K_RANGE = range(2, 16)  # k=2 to k=15
RANDOM_STATE = 42

# ── 1. Load data ──────────────────────────────────────────────────────────
print("Loading preprocessed data ...")
ids = pd.read_csv(PREP_DIR / "ids.csv")["ID"]
features_df = pd.read_csv(PREP_DIR / "features.csv")
raw_ts = pd.read_csv(PREP_DIR / "raw.csv")
zscore_ts = pd.read_csv(PREP_DIR / "zscore.csv")
log_ts = pd.read_csv(PREP_DIR / "log.csv")

dates = pd.to_datetime(raw_ts.columns)

# Features for clustering (drop id)
feature_cols = [c for c in features_df.columns if c != "id"]
X_features_raw = features_df[feature_cols].values

# Handle inf/nan in features
X_features_raw = np.nan_to_num(X_features_raw, nan=0.0, posinf=0.0, neginf=0.0)

# Cap extreme values in winter_summer_ratio and weekend_weekday_ratio
# (some households have near-zero summer/weekday causing extreme ratios)
for col_idx, col_name in enumerate(feature_cols):
    if col_name in ['winter_summer_ratio', 'weekend_weekday_ratio']:
        p99 = np.percentile(X_features_raw[:, col_idx], 99)
        X_features_raw[:, col_idx] = np.clip(X_features_raw[:, col_idx], 0, p99)

# Standardize features
scaler = StandardScaler()
X_features = scaler.fit_transform(X_features_raw)

print(f"Feature matrix: {X_features.shape}")
print(f"Time series matrix: {raw_ts.shape}")

# ── 2. PCA for dimensionality reduction ───────────────────────────────────
print("\nRunning PCA on features ...")
pca = PCA(n_components=min(10, X_features.shape[1]))
X_pca = pca.fit_transform(X_features)
explained_var = pca.explained_variance_ratio_
cumvar = np.cumsum(explained_var)
n_components_90 = int(np.argmax(cumvar >= 0.90)) + 1

print(f"  Components for 90% variance: {n_components_90}")
print(f"  Explained variance (first 5): {explained_var[:5].round(3)}")

# PCA on z-score time series
pca_ts = PCA(n_components=10)
X_ts_pca = pca_ts.fit_transform(zscore_ts.values)
ts_cumvar = np.cumsum(pca_ts.explained_variance_ratio_)
print(f"  Z-score TS PCA: first 10 components explain {ts_cumvar[-1]*100:.1f}% variance")

# Plot PCA variance
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].bar(range(1, len(explained_var)+1), explained_var, alpha=0.7, label='Individual')
axes[0].plot(range(1, len(explained_var)+1), cumvar, 'ro-', label='Cumulative')
axes[0].axhline(y=0.90, color='gray', linestyle='--', alpha=0.5)
axes[0].set_xlabel("Component")
axes[0].set_ylabel("Explained Variance Ratio")
axes[0].set_title("PCA on Extracted Features")
axes[0].legend()

axes[1].bar(range(1, 11), pca_ts.explained_variance_ratio_, alpha=0.7, label='Individual')
axes[1].plot(range(1, 11), ts_cumvar, 'ro-', label='Cumulative')
axes[1].set_xlabel("Component")
axes[1].set_ylabel("Explained Variance Ratio")
axes[1].set_title("PCA on Z-score Time Series")
axes[1].legend()
plt.tight_layout()
plt.savefig(PLOT_DIR / "10_pca_variance.png", dpi=150)
plt.close()
print("Saved: 10_pca_variance.png")


# ══════════════════════════════════════════════════════════════════════════
# EXPERIMENT 1: K-Means++ on extracted features
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("EXPERIMENT 1: K-Means++ on extracted features (standardized)")
print("=" * 70)

inertias_feat = []
silhouettes_feat = []
ch_scores_feat = []
db_scores_feat = []

for k in K_RANGE:
    km = KMeans(n_clusters=k, init='k-means++', n_init=10, max_iter=300,
                random_state=RANDOM_STATE)
    labels = km.fit_predict(X_features)
    inertias_feat.append(km.inertia_)
    sil = silhouette_score(X_features, labels, sample_size=min(5000, len(labels)),
                           random_state=RANDOM_STATE)
    silhouettes_feat.append(sil)
    ch_scores_feat.append(calinski_harabasz_score(X_features, labels))
    db_scores_feat.append(davies_bouldin_score(X_features, labels))
    print(f"  k={k:2d}: inertia={km.inertia_:12.0f}, silhouette={sil:.4f}, "
          f"CH={ch_scores_feat[-1]:.0f}, DB={db_scores_feat[-1]:.3f}")

# ══════════════════════════════════════════════════════════════════════════
# EXPERIMENT 2: K-Means++ on z-score time series (PCA-reduced)
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("EXPERIMENT 2: K-Means++ on z-score time series (PCA-10)")
print("=" * 70)

inertias_ts = []
silhouettes_ts = []
ch_scores_ts = []
db_scores_ts = []

for k in K_RANGE:
    km = KMeans(n_clusters=k, init='k-means++', n_init=10, max_iter=300,
                random_state=RANDOM_STATE)
    labels = km.fit_predict(X_ts_pca)
    inertias_ts.append(km.inertia_)
    sil = silhouette_score(X_ts_pca, labels, sample_size=min(5000, len(labels)),
                           random_state=RANDOM_STATE)
    silhouettes_ts.append(sil)
    ch_scores_ts.append(calinski_harabasz_score(X_ts_pca, labels))
    db_scores_ts.append(davies_bouldin_score(X_ts_pca, labels))
    print(f"  k={k:2d}: inertia={km.inertia_:12.0f}, silhouette={sil:.4f}, "
          f"CH={ch_scores_ts[-1]:.0f}, DB={db_scores_ts[-1]:.3f}")

# ══════════════════════════════════════════════════════════════════════════
# EXPERIMENT 3: K-Means++ on log-transformed time series (PCA-reduced)
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("EXPERIMENT 3: K-Means++ on log-transformed time series (PCA-10)")
print("=" * 70)

# Standardize log-transformed data
log_scaler = StandardScaler()
X_log_scaled = log_scaler.fit_transform(log_ts.values)
pca_log = PCA(n_components=10)
X_log_pca = pca_log.fit_transform(X_log_scaled)

inertias_log = []
silhouettes_log = []

for k in K_RANGE:
    km = KMeans(n_clusters=k, init='k-means++', n_init=10, max_iter=300,
                random_state=RANDOM_STATE)
    labels = km.fit_predict(X_log_pca)
    inertias_log.append(km.inertia_)
    sil = silhouette_score(X_log_pca, labels, sample_size=min(5000, len(labels)),
                           random_state=RANDOM_STATE)
    silhouettes_log.append(sil)
    print(f"  k={k:2d}: inertia={km.inertia_:12.0f}, silhouette={sil:.4f}")

# ── Elbow and Silhouette Comparison Plots ─────────────────────────────────

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Elbow curves
axes[0, 0].plot(list(K_RANGE), inertias_feat, 'bo-', label='Features')
axes[0, 0].set_xlabel("k")
axes[0, 0].set_ylabel("Inertia")
axes[0, 0].set_title("Elbow Method: Features")
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].plot(list(K_RANGE), inertias_ts, 'rs-', label='Z-score TS')
axes[0, 1].plot(list(K_RANGE), inertias_log, 'g^-', label='Log TS')
axes[0, 1].set_xlabel("k")
axes[0, 1].set_ylabel("Inertia")
axes[0, 1].set_title("Elbow Method: Time Series (PCA-10)")
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Silhouette scores
axes[1, 0].plot(list(K_RANGE), silhouettes_feat, 'bo-', label='Features')
axes[1, 0].plot(list(K_RANGE), silhouettes_ts, 'rs-', label='Z-score TS')
axes[1, 0].plot(list(K_RANGE), silhouettes_log, 'g^-', label='Log TS')
axes[1, 0].set_xlabel("k")
axes[1, 0].set_ylabel("Silhouette Score")
axes[1, 0].set_title("Silhouette Score Comparison")
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Calinski-Harabasz (features vs z-score)
axes[1, 1].plot(list(K_RANGE), ch_scores_feat, 'bo-', label='Features')
axes[1, 1].plot(list(K_RANGE), ch_scores_ts, 'rs-', label='Z-score TS')
axes[1, 1].set_xlabel("k")
axes[1, 1].set_ylabel("Calinski-Harabasz Score")
axes[1, 1].set_title("Calinski-Harabasz Index")
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.suptitle("Clustering Evaluation: K-Means++ (k=2..15)", fontsize=13)
plt.tight_layout()
plt.savefig(PLOT_DIR / "11_elbow_silhouette.png", dpi=150)
plt.close()
print("\nSaved: 11_elbow_silhouette.png")

# ── Determine optimal k for features ─────────────────────────────────────
# Use silhouette score as primary criterion
best_k_feat = list(K_RANGE)[np.argmax(silhouettes_feat)]
print(f"\nBest k for features (by silhouette): {best_k_feat}")
print(f"  Silhouette = {max(silhouettes_feat):.4f}")

best_k_ts = list(K_RANGE)[np.argmax(silhouettes_ts)]
print(f"Best k for z-score TS (by silhouette): {best_k_ts}")
print(f"  Silhouette = {max(silhouettes_ts):.4f}")


# ══════════════════════════════════════════════════════════════════════════
# FINAL CLUSTERING: K-Means on features with optimal k
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print(f"FINAL K-MEANS CLUSTERING: k={best_k_feat} on extracted features")
print("=" * 70)

km_final = KMeans(n_clusters=best_k_feat, init='k-means++', n_init=20, max_iter=500,
                  random_state=RANDOM_STATE)
labels_feat = km_final.fit_predict(X_features)
features_df["cluster_kmeans"] = labels_feat

# Cluster sizes
cluster_sizes = pd.Series(labels_feat).value_counts().sort_index()
print("\nCluster sizes:")
for c, n in cluster_sizes.items():
    print(f"  Cluster {c}: {n} households ({n/len(labels_feat)*100:.1f}%)")

# Cluster profiles (mean feature values)
print("\nCluster profiles (mean feature values):")
cluster_profiles = features_df.groupby("cluster_kmeans")[feature_cols].mean()
print(cluster_profiles.round(3).to_string())


# ══════════════════════════════════════════════════════════════════════════
# EXPERIMENT 4: Hierarchical Clustering (Ward) on features
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print(f"EXPERIMENT 4: Hierarchical Clustering (Ward) k={best_k_feat}")
print("=" * 70)

hc = AgglomerativeClustering(n_clusters=best_k_feat, linkage='ward')
labels_hc = hc.fit_predict(X_features)
features_df["cluster_hierarchical"] = labels_hc

sil_hc = silhouette_score(X_features, labels_hc, sample_size=min(5000, len(labels_hc)),
                          random_state=RANDOM_STATE)
ch_hc = calinski_harabasz_score(X_features, labels_hc)
db_hc = davies_bouldin_score(X_features, labels_hc)
print(f"  Silhouette: {sil_hc:.4f}")
print(f"  Calinski-Harabasz: {ch_hc:.0f}")
print(f"  Davies-Bouldin: {db_hc:.3f}")

hc_sizes = pd.Series(labels_hc).value_counts().sort_index()
print("\nCluster sizes:")
for c, n in hc_sizes.items():
    print(f"  Cluster {c}: {n} households ({n/len(labels_hc)*100:.1f}%)")


# ══════════════════════════════════════════════════════════════════════════
# EXPERIMENT 5: DBSCAN on features
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("EXPERIMENT 5: DBSCAN on extracted features (PCA-reduced)")
print("=" * 70)

# Use PCA-reduced features for DBSCAN (works better in lower dims)
X_feat_pca = PCA(n_components=n_components_90).fit_transform(X_features)

# Try different eps values
from sklearn.neighbors import NearestNeighbors
nn = NearestNeighbors(n_neighbors=10)
nn.fit(X_feat_pca)
distances, _ = nn.kneighbors(X_feat_pca)
k_dist = np.sort(distances[:, -1])

# Plot k-distance
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(k_dist)
ax.set_xlabel("Points (sorted)")
ax.set_ylabel("10-NN Distance")
ax.set_title("k-Distance Graph for DBSCAN eps Selection")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(PLOT_DIR / "12_dbscan_kdist.png", dpi=150)
plt.close()
print("Saved: 12_dbscan_kdist.png")

# Try DBSCAN with a few eps values
best_dbscan_sil = -1
best_dbscan_eps = None
best_dbscan_labels = None

for eps in [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]:
    db = DBSCAN(eps=eps, min_samples=10)
    labels_db = db.fit_predict(X_feat_pca)
    n_clusters = len(set(labels_db)) - (1 if -1 in labels_db else 0)
    n_noise = (labels_db == -1).sum()

    if n_clusters >= 2:
        mask = labels_db != -1
        sil_db = silhouette_score(X_feat_pca[mask], labels_db[mask],
                                   sample_size=min(5000, mask.sum()),
                                   random_state=RANDOM_STATE)
    else:
        sil_db = -1

    print(f"  eps={eps:.1f}: {n_clusters} clusters, {n_noise} noise points ({n_noise/len(labels_db)*100:.1f}%), "
          f"silhouette={sil_db:.4f}")

    if sil_db > best_dbscan_sil and n_clusters >= 2:
        best_dbscan_sil = sil_db
        best_dbscan_eps = eps
        best_dbscan_labels = labels_db

if best_dbscan_labels is not None:
    features_df["cluster_dbscan"] = best_dbscan_labels
    print(f"\nBest DBSCAN: eps={best_dbscan_eps}, silhouette={best_dbscan_sil:.4f}")
    dbscan_sizes = pd.Series(best_dbscan_labels).value_counts().sort_index()
    for c, n in dbscan_sizes.items():
        label = "NOISE" if c == -1 else f"Cluster {c}"
        print(f"  {label}: {n} households ({n/len(best_dbscan_labels)*100:.1f}%)")


# ══════════════════════════════════════════════════════════════════════════
# ALSO: K-Means on Z-score TS for comparison
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print(f"K-Means on Z-score Time Series: k={best_k_ts}")
print("=" * 70)

km_ts = KMeans(n_clusters=best_k_ts, init='k-means++', n_init=20, max_iter=500,
               random_state=RANDOM_STATE)
labels_ts = km_ts.fit_predict(X_ts_pca)
features_df["cluster_ts_kmeans"] = labels_ts

ts_sizes = pd.Series(labels_ts).value_counts().sort_index()
print("Cluster sizes:")
for c, n in ts_sizes.items():
    print(f"  Cluster {c}: {n} households ({n/len(labels_ts)*100:.1f}%)")


# ══════════════════════════════════════════════════════════════════════════
# VISUALIZATIONS
# ══════════════════════════════════════════════════════════════════════════

# ── Cluster consumption profiles (K-Means on features) ────────────────────
print("\nGenerating cluster visualization plots ...")

fig, axes = plt.subplots(best_k_feat, 1, figsize=(14, 4 * best_k_feat), sharex=True)
if best_k_feat == 1:
    axes = [axes]

for c in range(best_k_feat):
    mask = labels_feat == c
    cluster_ts = raw_ts.values[mask]
    mean_profile = cluster_ts.mean(axis=0)
    q25 = np.percentile(cluster_ts, 25, axis=0)
    q75 = np.percentile(cluster_ts, 75, axis=0)

    axes[c].plot(dates, mean_profile, linewidth=1.5, label='Mean')
    axes[c].fill_between(dates, q25, q75, alpha=0.2, label='IQR')
    axes[c].set_ylabel("kWh")
    axes[c].set_title(f"Cluster {c} (n={mask.sum()}, "
                      f"avg={features_df.loc[mask, 'mean'].mean():.1f} kWh/day)")
    axes[c].legend(loc='upper right')
    axes[c].grid(True, alpha=0.2)

axes[-1].set_xlabel("Date")
plt.suptitle(f"K-Means Cluster Profiles (k={best_k_feat}, features)", fontsize=13)
plt.tight_layout()
plt.savefig(PLOT_DIR / "13_cluster_profiles_kmeans.png", dpi=150)
plt.close()
print("Saved: 13_cluster_profiles_kmeans.png")

# ── t-SNE visualization ──────────────────────────────────────────────────
print("Running t-SNE (this may take a minute) ...")
# Subsample for t-SNE (faster + cleaner plot)
np.random.seed(RANDOM_STATE)
n_sample = min(5000, len(X_features))
sample_idx = np.random.choice(len(X_features), n_sample, replace=False)

tsne = TSNE(n_components=2, perplexity=30, random_state=RANDOM_STATE, max_iter=1000)
X_tsne = tsne.fit_transform(X_features[sample_idx])

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# K-Means labels
scatter1 = axes[0].scatter(X_tsne[:, 0], X_tsne[:, 1],
                           c=labels_feat[sample_idx], cmap='tab10',
                           alpha=0.5, s=5)
axes[0].set_title(f"t-SNE: K-Means (k={best_k_feat}, features)")
axes[0].set_xlabel("t-SNE 1")
axes[0].set_ylabel("t-SNE 2")
plt.colorbar(scatter1, ax=axes[0], label='Cluster')

# Hierarchical labels
scatter2 = axes[1].scatter(X_tsne[:, 0], X_tsne[:, 1],
                           c=labels_hc[sample_idx], cmap='tab10',
                           alpha=0.5, s=5)
axes[1].set_title(f"t-SNE: Hierarchical (k={best_k_feat}, Ward)")
axes[1].set_xlabel("t-SNE 1")
axes[1].set_ylabel("t-SNE 2")
plt.colorbar(scatter2, ax=axes[1], label='Cluster')

plt.suptitle("t-SNE Visualization of Clusters (5000 sample)", fontsize=13)
plt.tight_layout()
plt.savefig(PLOT_DIR / "14_tsne_clusters.png", dpi=150)
plt.close()
print("Saved: 14_tsne_clusters.png")

# ── PCA 2D visualization ─────────────────────────────────────────────────
pca2d = PCA(n_components=2)
X_pca2d = pca2d.fit_transform(X_features)

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

scatter1 = axes[0].scatter(X_pca2d[:, 0], X_pca2d[:, 1],
                           c=labels_feat, cmap='tab10', alpha=0.3, s=3)
axes[0].set_title(f"PCA: K-Means (k={best_k_feat})")
axes[0].set_xlabel(f"PC1 ({pca2d.explained_variance_ratio_[0]*100:.1f}%)")
axes[0].set_ylabel(f"PC2 ({pca2d.explained_variance_ratio_[1]*100:.1f}%)")
plt.colorbar(scatter1, ax=axes[0], label='Cluster')

scatter2 = axes[1].scatter(X_pca2d[:, 0], X_pca2d[:, 1],
                           c=labels_hc, cmap='tab10', alpha=0.3, s=3)
axes[1].set_title(f"PCA: Hierarchical (k={best_k_feat})")
axes[1].set_xlabel(f"PC1 ({pca2d.explained_variance_ratio_[0]*100:.1f}%)")
axes[1].set_ylabel(f"PC2 ({pca2d.explained_variance_ratio_[1]*100:.1f}%)")
plt.colorbar(scatter2, ax=axes[1], label='Cluster')

plt.suptitle("PCA 2D Visualization of Clusters", fontsize=13)
plt.tight_layout()
plt.savefig(PLOT_DIR / "15_pca_clusters.png", dpi=150)
plt.close()
print("Saved: 15_pca_clusters.png")

# ── Cluster feature comparison (box plots) ────────────────────────────────
key_features = ['mean', 'std', 'cv', 'winter_summer_ratio', 'trend_slope',
                'weekend_weekday_ratio', 'autocorr_lag1', 'peak_month']

fig, axes = plt.subplots(2, 4, figsize=(18, 10))
axes = axes.flatten()

for i, feat in enumerate(key_features):
    data_by_cluster = [features_df.loc[labels_feat == c, feat].values for c in range(best_k_feat)]
    bp = axes[i].boxplot(data_by_cluster, labels=[str(c) for c in range(best_k_feat)],
                         patch_artist=True)
    colors = plt.cm.tab10(np.linspace(0, 1, best_k_feat))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    axes[i].set_title(feat)
    axes[i].set_xlabel("Cluster")
    axes[i].grid(True, alpha=0.2)

plt.suptitle(f"Feature Distributions per Cluster (K-Means, k={best_k_feat})", fontsize=13)
plt.tight_layout()
plt.savefig(PLOT_DIR / "16_cluster_features_boxplot.png", dpi=150)
plt.close()
print("Saved: 16_cluster_features_boxplot.png")

# ── Cluster heatmap: normalized profiles ──────────────────────────────────
cluster_means = []
for c in range(best_k_feat):
    mask = labels_feat == c
    # Normalize each cluster's mean profile to [0, 1] for comparison
    profile = raw_ts.values[mask].mean(axis=0)
    if profile.max() > profile.min():
        profile_norm = (profile - profile.min()) / (profile.max() - profile.min())
    else:
        profile_norm = np.zeros_like(profile)
    cluster_means.append(profile_norm)

# Resample to monthly for cleaner heatmap
monthly_profiles = []
month_labels = []
for m in range(1, 13):
    col_mask = [i for i, d in enumerate(dates) if d.month == m]
    month_labels.append(dates[col_mask[0]].strftime('%b'))
    for c in range(best_k_feat):
        pass
    monthly_profiles.append([raw_ts.values[labels_feat == c][:, col_mask].mean()
                             for c in range(best_k_feat)])

monthly_heatmap = np.array(monthly_profiles).T  # (k, 12)

fig, ax = plt.subplots(figsize=(12, max(4, best_k_feat)))
sns.heatmap(monthly_heatmap, annot=True, fmt=".1f", cmap='YlOrRd',
            xticklabels=month_labels,
            yticklabels=[f"Cluster {c} (n={cluster_sizes[c]})" for c in range(best_k_feat)],
            ax=ax)
ax.set_title("Average Monthly Consumption by Cluster (kWh)")
ax.set_xlabel("Month")
plt.tight_layout()
plt.savefig(PLOT_DIR / "17_cluster_monthly_heatmap.png", dpi=150)
plt.close()
print("Saved: 17_cluster_monthly_heatmap.png")

# ── Algorithm comparison summary ──────────────────────────────────────────

# Compute silhouette for K-Means on features at optimal k
sil_km_feat = silhouette_score(X_features, labels_feat,
                                sample_size=min(5000, len(labels_feat)),
                                random_state=RANDOM_STATE)
ch_km_feat = calinski_harabasz_score(X_features, labels_feat)
db_km_feat = davies_bouldin_score(X_features, labels_feat)

sil_ts = silhouette_score(X_ts_pca, labels_ts,
                          sample_size=min(5000, len(labels_ts)),
                          random_state=RANDOM_STATE)
ch_ts = calinski_harabasz_score(X_ts_pca, labels_ts)
db_ts = davies_bouldin_score(X_ts_pca, labels_ts)

comparison = {
    "KMeans_Features": {
        "k": best_k_feat,
        "silhouette": round(sil_km_feat, 4),
        "calinski_harabasz": round(ch_km_feat, 1),
        "davies_bouldin": round(db_km_feat, 4),
        "cluster_sizes": {int(c): int(n) for c, n in cluster_sizes.items()}
    },
    "Hierarchical_Features": {
        "k": best_k_feat,
        "silhouette": round(sil_hc, 4),
        "calinski_harabasz": round(ch_hc, 1),
        "davies_bouldin": round(db_hc, 4),
        "cluster_sizes": {int(c): int(n) for c, n in hc_sizes.items()}
    },
    "KMeans_ZscoreTS": {
        "k": best_k_ts,
        "silhouette": round(sil_ts, 4),
        "calinski_harabasz": round(ch_ts, 1),
        "davies_bouldin": round(db_ts, 4),
        "cluster_sizes": {int(c): int(n) for c, n in ts_sizes.items()}
    }
}

if best_dbscan_labels is not None:
    comparison["DBSCAN_Features"] = {
        "eps": best_dbscan_eps,
        "silhouette": round(best_dbscan_sil, 4),
        "n_clusters": int(len(set(best_dbscan_labels)) - (1 if -1 in best_dbscan_labels else 0)),
        "noise_points": int((best_dbscan_labels == -1).sum())
    }

with open(OUT_DIR / "clustering_comparison.json", "w") as f:
    json.dump(comparison, f, indent=2)

# Save cluster assignments
assignments = pd.DataFrame({
    "ID": ids,
    "cluster_kmeans_features": labels_feat,
    "cluster_hierarchical": labels_hc,
    "cluster_kmeans_ts": labels_ts,
})
if best_dbscan_labels is not None:
    assignments["cluster_dbscan"] = best_dbscan_labels
assignments.to_csv(OUT_DIR / "cluster_assignments.csv", index=False)
print("\nSaved: cluster_assignments.csv")

# Save cluster profiles
cluster_profiles.to_csv(OUT_DIR / "cluster_profiles.csv")
print("Saved: cluster_profiles.csv")

# ── Summary plot comparing all methods ────────────────────────────────────
methods = ['K-Means\n(Features)', 'Hierarchical\n(Features)', 'K-Means\n(Z-score TS)']
sils = [sil_km_feat, sil_hc, sil_ts]
dbs = [db_km_feat, db_hc, db_ts]

if best_dbscan_labels is not None:
    methods.append(f'DBSCAN\n(eps={best_dbscan_eps})')
    sils.append(best_dbscan_sil)
    mask_db = best_dbscan_labels != -1
    dbs.append(davies_bouldin_score(X_feat_pca[mask_db], best_dbscan_labels[mask_db])
               if mask_db.sum() > best_k_feat else 0)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
colors = ['steelblue', 'coral', 'seagreen', 'mediumpurple']

axes[0].bar(methods, sils, color=colors[:len(methods)], edgecolor='black')
axes[0].set_ylabel("Silhouette Score (higher is better)")
axes[0].set_title("Silhouette Score Comparison")
for i, v in enumerate(sils):
    axes[0].text(i, v + 0.005, f'{v:.3f}', ha='center', fontsize=10)

axes[1].bar(methods, dbs, color=colors[:len(methods)], edgecolor='black')
axes[1].set_ylabel("Davies-Bouldin Index (lower is better)")
axes[1].set_title("Davies-Bouldin Index Comparison")
for i, v in enumerate(dbs):
    axes[1].text(i, v + 0.02, f'{v:.3f}', ha='center', fontsize=10)

plt.suptitle("Clustering Algorithm Comparison", fontsize=13)
plt.tight_layout()
plt.savefig(PLOT_DIR / "18_algorithm_comparison.png", dpi=150)
plt.close()
print("Saved: 18_algorithm_comparison.png")

# ── Final summary ─────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("CLUSTERING SUMMARY")
print("=" * 70)
print(f"\nBest approach: K-Means++ on extracted features")
print(f"Optimal k: {best_k_feat}")
print(f"Silhouette score: {sil_km_feat:.4f}")
print(f"Calinski-Harabasz: {ch_km_feat:.0f}")
print(f"Davies-Bouldin: {db_km_feat:.4f}")
print(f"\nCluster sizes: {dict(cluster_sizes)}")
print(f"\nAll results saved to: {OUT_DIR}")
print(f"All plots saved to: {PLOT_DIR}")
print("\n=== Clustering complete ===")
