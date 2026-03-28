import pandas as pd
import numpy as np

from clustering_module.pipeline import ClusteringPipeline
from clustering_module.clustering import HierarchicalClustering, KMeansClustering
from clustering_module.preprocessing import IQRSelection, MeanImputation, StandardScaling

data = pd.DataFrame(np.random.rand(100, 5), columns=[f'feature_{i}' for i in range(5)])

# Experiment 1: Mean Imputation + IQR + Standard Scaler + KMeans
pipeline_1 = ClusteringPipeline(
    zero_strategy=MeanImputation(),
    outlier_strategy=IQRSelection(),
    norm_strategy=StandardScaling(),
    cluster_strategy=KMeansClustering(n_clusters=5)
)

results = pipeline_1.run(data)

# --- THE FORECASTING PART ---
# Now can iterate over the clusters and train separate models
for cluster_id in results['cluster'].unique():
    cluster_data = results[results['cluster'] == cluster_id]
    print(f"Training forecasting model for Cluster {cluster_id} with {len(cluster_data)} samples...")
    # forecasting_model.fit(cluster_data)