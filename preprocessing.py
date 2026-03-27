"""
Preprocessing script for KDD project:
"Leveraging Clustering for Large-Scale Time Series Forecasting"

Loads household electricity consumption data (2023), explores distributions,
handles outliers, and produces multiple preprocessed versions for clustering.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

# ── Configuration ──────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).parent / "data"
PLOT_DIR = Path(__file__).parent / "plots"
OUT_DIR  = Path(__file__).parent / "data" / "preprocessed"

PLOT_DIR.mkdir(exist_ok=True)
OUT_DIR.mkdir(exist_ok=True)

# ── 1. Load data ──────────────────────────────────────────────────────────
print("Loading sample_23.csv ...")
df = pd.read_csv(DATA_DIR / "sample_23.csv")
ids = df["ID"]
ts = df.drop(columns=["ID"])  # shape: (17547, 365)

dates = pd.to_datetime(ts.columns)
print(f"Loaded {ts.shape[0]} households, {ts.shape[1]} days")
print(f"Date range: {dates.min().date()} to {dates.max().date()}")

# ── 2. Basic statistics ──────────────────────────────────────────────────
household_mean = ts.mean(axis=1)
household_std  = ts.std(axis=1)
household_sum  = ts.sum(axis=1)
household_max  = ts.max(axis=1)
household_min  = ts.min(axis=1)

stats = {
    "n_households": int(ts.shape[0]),
    "n_days": int(ts.shape[1]),
    "overall_mean_kwh_per_day": float(household_mean.mean()),
    "overall_median_kwh_per_day": float(household_mean.median()),
    "overall_std_kwh_per_day": float(household_mean.std()),
    "min_household_mean": float(household_mean.min()),
    "max_household_mean": float(household_mean.max()),
    "zero_consumption_households": int((household_sum == 0).sum()),
    "near_zero_households_lt_0.1": int((household_mean < 0.1).sum()),
    "missing_values": int(ts.isna().sum().sum()),
}

# Percentiles
for p in [1, 5, 25, 50, 75, 95, 99]:
    stats[f"p{p}_household_mean"] = float(np.percentile(household_mean, p))

print("\n=== Dataset Statistics ===")
for k, v in stats.items():
    print(f"  {k}: {v}")

# Save stats
with open(OUT_DIR.parent / "stats.json", "w") as f:
    json.dump(stats, f, indent=2)

# ── 3. Visualizations: Data Exploration ───────────────────────────────────

# 3a. Distribution of household mean consumption
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].hist(household_mean, bins=100, edgecolor='black', alpha=0.7)
axes[0].set_xlabel("Mean daily consumption (kWh)")
axes[0].set_ylabel("Number of households")
axes[0].set_title("Distribution of Mean Daily Consumption")
axes[0].axvline(household_mean.median(), color='red', linestyle='--', label=f'Median={household_mean.median():.2f}')
axes[0].axvline(household_mean.mean(), color='orange', linestyle='--', label=f'Mean={household_mean.mean():.2f}')
axes[0].legend()

# Log-scale y-axis for better view of tail
axes[1].hist(household_mean, bins=100, edgecolor='black', alpha=0.7)
axes[1].set_xlabel("Mean daily consumption (kWh)")
axes[1].set_ylabel("Number of households (log scale)")
axes[1].set_title("Distribution (log y-axis)")
axes[1].set_yscale('log')
axes[1].axvline(np.percentile(household_mean, 99), color='red', linestyle='--',
                label=f'99th percentile={np.percentile(household_mean, 99):.1f}')
axes[1].legend()
plt.tight_layout()
plt.savefig(PLOT_DIR / "01_distribution_mean_consumption.png", dpi=150)
plt.close()
print("Saved: 01_distribution_mean_consumption.png")

# 3b. Box plot
fig, ax = plt.subplots(figsize=(10, 4))
bp = ax.boxplot(household_mean, vert=False, widths=0.6)
ax.set_xlabel("Mean daily consumption (kWh)")
ax.set_title("Boxplot of Household Mean Daily Consumption")
plt.tight_layout()
plt.savefig(PLOT_DIR / "02_boxplot_mean_consumption.png", dpi=150)
plt.close()
print("Saved: 02_boxplot_mean_consumption.png")

# 3c. Average consumption over time (aggregate daily)
daily_avg = ts.mean(axis=0)
daily_median = ts.median(axis=0)

fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(dates, daily_avg, label='Mean', alpha=0.8, linewidth=0.8)
ax.plot(dates, daily_median, label='Median', alpha=0.8, linewidth=0.8)
ax.fill_between(dates,
                ts.quantile(0.25, axis=0),
                ts.quantile(0.75, axis=0),
                alpha=0.2, label='IQR (25th-75th)')
ax.set_xlabel("Date")
ax.set_ylabel("Daily consumption (kWh)")
ax.set_title("Aggregate Daily Consumption Profile (2023)")
ax.legend()
plt.tight_layout()
plt.savefig(PLOT_DIR / "03_daily_consumption_profile.png", dpi=150)
plt.close()
print("Saved: 03_daily_consumption_profile.png")

# 3d. Monthly average consumption
monthly_idx = dates.month
monthly_means = []
for m in range(1, 13):
    cols = [c for c, d in zip(ts.columns, dates) if d.month == m]
    monthly_means.append(ts[cols].mean(axis=1).mean())

fig, ax = plt.subplots(figsize=(10, 5))
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
ax.bar(months, monthly_means, color='steelblue', edgecolor='black')
ax.set_ylabel("Mean daily consumption (kWh)")
ax.set_title("Average Daily Consumption by Month (2023)")
plt.tight_layout()
plt.savefig(PLOT_DIR / "04_monthly_consumption.png", dpi=150)
plt.close()
print("Saved: 04_monthly_consumption.png")

# 3e. Sample time series from different quantiles
fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharex=True)
quantiles = [0.05, 0.25, 0.50, 0.75, 0.95, 0.99]
for i, q in enumerate(quantiles):
    ax = axes[i // 2, i % 2]
    target_val = household_mean.quantile(q)
    idx = (household_mean - target_val).abs().idxmin()
    ax.plot(dates, ts.loc[idx], linewidth=0.7)
    ax.set_title(f"Household at {q*100:.0f}th percentile (mean={household_mean.loc[idx]:.2f} kWh)")
    ax.set_ylabel("kWh")
axes[-1, 0].set_xlabel("Date")
axes[-1, 1].set_xlabel("Date")
plt.suptitle("Sample Household Consumption Profiles", fontsize=13)
plt.tight_layout()
plt.savefig(PLOT_DIR / "05_sample_household_profiles.png", dpi=150)
plt.close()
print("Saved: 05_sample_household_profiles.png")

# 3f. Correlation heatmap: monthly aggregates
monthly_data = pd.DataFrame()
for m in range(1, 13):
    cols = [c for c, d in zip(ts.columns, dates) if d.month == m]
    monthly_data[months[m-1]] = ts[cols].mean(axis=1)

fig, ax = plt.subplots(figsize=(10, 8))
corr = monthly_data.corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', ax=ax, vmin=0.5, vmax=1.0)
ax.set_title("Correlation Between Monthly Consumption")
plt.tight_layout()
plt.savefig(PLOT_DIR / "06_monthly_correlation.png", dpi=150)
plt.close()
print("Saved: 06_monthly_correlation.png")

# ── 4. Outlier Analysis ──────────────────────────────────────────────────

# IQR method on household means
Q1 = household_mean.quantile(0.25)
Q3 = household_mean.quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

n_outliers_iqr = int(((household_mean < lower_bound) | (household_mean > upper_bound)).sum())
print(f"\nOutlier analysis (IQR on household means):")
print(f"  Q1={Q1:.2f}, Q3={Q3:.2f}, IQR={IQR:.2f}")
print(f"  Lower bound={lower_bound:.2f}, Upper bound={upper_bound:.2f}")
print(f"  Outliers: {n_outliers_iqr} households ({n_outliers_iqr/len(household_mean)*100:.1f}%)")

# Percentile-based capping (winsorization at 1st and 99th percentile)
p01 = household_mean.quantile(0.01)
p99 = household_mean.quantile(0.99)
print(f"  1st percentile mean: {p01:.2f} kWh")
print(f"  99th percentile mean: {p99:.2f} kWh")

# ── 5. Preprocessing Approaches ──────────────────────────────────────────

# Approach 0: Raw data (no preprocessing)
print("\nSaving preprocessed versions ...")
ts.to_csv(OUT_DIR / "raw.csv", index=False)
ids.to_csv(OUT_DIR / "ids.csv", index=False)
print("  Saved: raw.csv")

# Approach 1: Z-score normalization (per household)
ts_zscore = ts.sub(household_mean, axis=0).div(household_std.replace(0, 1), axis=0)
ts_zscore.to_csv(OUT_DIR / "zscore.csv", index=False)
print("  Saved: zscore.csv")

# Approach 2: Min-max normalization (per household)
ts_range = (household_max - household_min).replace(0, 1)
ts_minmax = ts.sub(household_min, axis=0).div(ts_range, axis=0)
ts_minmax.to_csv(OUT_DIR / "minmax.csv", index=False)
print("  Saved: minmax.csv")

# Approach 3: Log transformation (log1p to handle zeros)
ts_log = np.log1p(ts)
ts_log.to_csv(OUT_DIR / "log.csv", index=False)
print("  Saved: log.csv")

# Approach 4: Z-score on log-transformed (combines log + normalization)
log_mean = ts_log.mean(axis=1)
log_std  = ts_log.std(axis=1)
ts_log_zscore = ts_log.sub(log_mean, axis=0).div(log_std.replace(0, 1), axis=0)
ts_log_zscore.to_csv(OUT_DIR / "log_zscore.csv", index=False)
print("  Saved: log_zscore.csv")

# ── 6. Visualization: Compare Preprocessing ──────────────────────────────

# Pick a representative household near the median
median_idx = (household_mean - household_mean.median()).abs().idxmin()

fig, axes = plt.subplots(5, 1, figsize=(14, 16), sharex=True)
titles = ['Raw', 'Z-score', 'Min-Max', 'Log', 'Log + Z-score']
datasets = [ts, ts_zscore, ts_minmax, ts_log, ts_log_zscore]

for ax, title, data in zip(axes, titles, datasets):
    ax.plot(dates, data.loc[median_idx], linewidth=0.7)
    ax.set_title(f"{title} (median household, ID={ids.loc[median_idx]})")
    ax.set_ylabel("Value")
axes[-1].set_xlabel("Date")
plt.suptitle("Preprocessing Comparison: Median Household", fontsize=13)
plt.tight_layout()
plt.savefig(PLOT_DIR / "07_preprocessing_comparison.png", dpi=150)
plt.close()
print("Saved: 07_preprocessing_comparison.png")

# ── 7. Feature extraction for clustering ──────────────────────────────────
print("\nExtracting time series features ...")

features = pd.DataFrame()
features["id"] = ids

# Basic stats
features["mean"] = household_mean.values
features["std"]  = household_std.values
features["cv"]   = (household_std / household_mean.replace(0, np.nan)).fillna(0).values
features["min"]  = household_min.values
features["max"]  = household_max.values
features["median"] = ts.median(axis=1).values
features["skewness"] = ts.skew(axis=1).values
features["kurtosis"] = ts.kurtosis(axis=1).values

# Trend: linear regression slope over the year
day_numbers = np.arange(ts.shape[1], dtype=np.float64)
day_centered = day_numbers - day_numbers.mean()
denominator = np.sum(day_centered ** 2)
ts_centered = ts.values - ts.values.mean(axis=1, keepdims=True)
features["trend_slope"] = (ts_centered @ day_centered) / denominator

# Seasonality: ratio of winter (Dec-Feb) to summer (Jun-Aug) consumption
winter_cols = [c for c, d in zip(ts.columns, dates) if d.month in [12, 1, 2]]
summer_cols = [c for c, d in zip(ts.columns, dates) if d.month in [6, 7, 8]]
winter_mean = ts[winter_cols].mean(axis=1)
summer_mean = ts[summer_cols].mean(axis=1)
features["winter_summer_ratio"] = (winter_mean / summer_mean.replace(0, np.nan)).fillna(1).values

# Peak month (1-12)
features["peak_month"] = monthly_data.idxmax(axis=1).map({m: i+1 for i, m in enumerate(months)}).values

# Weekend vs weekday ratio
day_of_week = dates.dayofweek  # Monday=0, Sunday=6
weekend_cols = [c for c, dow in zip(ts.columns, day_of_week) if dow >= 5]
weekday_cols = [c for c, dow in zip(ts.columns, day_of_week) if dow < 5]
weekend_mean = ts[weekend_cols].mean(axis=1)
weekday_mean = ts[weekday_cols].mean(axis=1)
features["weekend_weekday_ratio"] = (weekend_mean / weekday_mean.replace(0, np.nan)).fillna(1).values

# Autocorrelation at lag 1 and lag 7
def compute_autocorr(data, lag):
    """Compute autocorrelation at a given lag for each row."""
    n = data.shape[1]
    mean = data.mean(axis=1, keepdims=True)
    centered = data - mean
    var = np.sum(centered ** 2, axis=1)
    cov = np.sum(centered[:, :n-lag] * centered[:, lag:], axis=1)
    return np.where(var > 0, cov / var, 0)

ts_arr = ts.values
features["autocorr_lag1"] = compute_autocorr(ts_arr, 1)
features["autocorr_lag7"] = compute_autocorr(ts_arr, 7)

# Entropy (approximate, based on histogram of daily values)
def row_entropy(row, bins=20):
    counts, _ = np.histogram(row, bins=bins)
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    return -np.sum(probs * np.log2(probs))

features["entropy"] = np.apply_along_axis(row_entropy, 1, ts_arr)

# Zero-day count
features["zero_days"] = (ts == 0).sum(axis=1).values

# Save features
features.to_csv(OUT_DIR / "features.csv", index=False)
print(f"Extracted {features.shape[1] - 1} features for {features.shape[0]} households")
print("Saved: features.csv")

# Feature summary
print("\n=== Feature Summary ===")
print(features.drop(columns=["id"]).describe().round(3).to_string())

# ── 8. Feature distribution plots ─────────────────────────────────────────

feat_cols = [c for c in features.columns if c != "id"]
n_feats = len(feat_cols)
ncols = 4
nrows = (n_feats + ncols - 1) // ncols

fig, axes = plt.subplots(nrows, ncols, figsize=(16, 3 * nrows))
axes = axes.flatten()
for i, col in enumerate(feat_cols):
    axes[i].hist(features[col], bins=60, edgecolor='black', alpha=0.7, linewidth=0.3)
    axes[i].set_title(col, fontsize=10)
    axes[i].tick_params(labelsize=8)
for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)
plt.suptitle("Feature Distributions", fontsize=13)
plt.tight_layout()
plt.savefig(PLOT_DIR / "08_feature_distributions.png", dpi=150)
plt.close()
print("Saved: 08_feature_distributions.png")

# Feature correlation
fig, ax = plt.subplots(figsize=(12, 10))
corr = features[feat_cols].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', ax=ax, center=0,
            xticklabels=True, yticklabels=True)
ax.set_title("Feature Correlation Matrix")
ax.tick_params(labelsize=8)
plt.tight_layout()
plt.savefig(PLOT_DIR / "09_feature_correlation.png", dpi=150)
plt.close()
print("Saved: 09_feature_correlation.png")

print("\n=== Preprocessing complete ===")
