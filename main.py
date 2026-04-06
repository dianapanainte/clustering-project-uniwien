from pathlib import Path
import pandas as pd
import numpy as np

from clustering_module.pipeline import ClusteringPipeline
from clustering_module.clustering import HierarchicalClustering, KMeansClustering
from clustering_module.preprocessing import IQRMeanConsumption, IQRSelection, LogImputation, MeanImputation, MinMaxScaling, NoImputation, NoOutlierHandling, NoScaling, StandardScaling

CSV_FILE = 'sample_23.csv'

def load_data(file_name: str, folder: str = "data") -> pd.DataFrame:
    path = Path(folder) / file_name
    return pd.read_csv(path)

data = load_data(CSV_FILE)

# Experiment 1: Mean Imputation + IQR + Standard Scaler + KMeans
pipeline_1 = ClusteringPipeline(
    zero_strategy=LogImputation(),
    outlier_strategy=IQRMeanConsumption(),
    norm_strategy=MinMaxScaling(),
    cluster_strategy=KMeansClustering(n_clusters=3)
)

results = pipeline_1.run(data)

# --- THE FORECASTING PART ---
# Now can iterate over the clusters and train separate models
for cluster_id in results['cluster'].unique():
    cluster_data = results[results['cluster'] == cluster_id]
    print(f"Training forecasting model for Cluster {cluster_id} with {len(cluster_data)} samples...")
    # forecasting_model.fit(cluster_data)