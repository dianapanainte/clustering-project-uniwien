# this file contains other pipelines that we used to compare with the main pipeline (with KMeans++ clustering)
# run this file to see the results of all pipelines and the MAE comparison across them

import pandas as pd
from pathlib import Path

from clustering_module.clustering import DBSCANClustering, HDBSCANClustering, HierarchicalClustering, KMeansClustering, KMeansPlusClustering, TwoStageClusteringStrategy
from clustering_module.features import PCAFeatures
from clustering_module.pipeline import ClusteringPipeline
from clustering_module.preprocessing import IQRMeanConsumption, LogImputation, MinMaxScaling
from main import cluster_and_forecast

CSV_FILE = 'sample_23.csv'

def load_data(file_name: str, folder: str = "data") -> pd.DataFrame:
    path = Path(folder) / file_name
    return pd.read_csv(path)

data = load_data(CSV_FILE)

pipeline_2 = ClusteringPipeline(
    feature_strategy=PCAFeatures(n_components=15),
    zero_strategy=LogImputation(),
    outlier_strategy=IQRMeanConsumption(),
    norm_strategy=MinMaxScaling(),
    cluster_strategy=TwoStageClusteringStrategy(first_stage=KMeansPlusClustering(n_clusters=3), second_stage=KMeansClustering(n_clusters=3), min_cluster_size=10)
)

pipeline_3 = ClusteringPipeline(
    feature_strategy=PCAFeatures(n_components=15),
    zero_strategy=LogImputation(),
    outlier_strategy=IQRMeanConsumption(),
    norm_strategy=MinMaxScaling(),
    cluster_strategy=KMeansClustering(n_clusters=3)
)

pipeline_4 = ClusteringPipeline(
    feature_strategy=PCAFeatures(n_components=15),
    zero_strategy=LogImputation(),
    outlier_strategy=IQRMeanConsumption(),
    norm_strategy=MinMaxScaling(),
    cluster_strategy=HierarchicalClustering(n_clusters=3)
)

pipeline_5 = ClusteringPipeline(
    feature_strategy=PCAFeatures(n_components=15),
    zero_strategy=LogImputation(),
    outlier_strategy=IQRMeanConsumption(),
    norm_strategy=MinMaxScaling(),
    cluster_strategy=HDBSCANClustering(min_cluster_size=5)
)

pipeline_6 = ClusteringPipeline(
    feature_strategy=PCAFeatures(n_components=15),
    zero_strategy=LogImputation(),
    outlier_strategy=IQRMeanConsumption(),
    norm_strategy=MinMaxScaling(),
    cluster_strategy=DBSCANClustering(eps=0.5, min_samples=5)
)

if __name__ == "__main__":
    mae_across_all_pipelines = {}
    print("\nRunning Pipeline 2: Two-Stage Clustering - KMeans++ followed by KMeans")
    cluster_and_forecast(data, pipeline_2, mae_across_all_pipelines)
    print("\nRunning Pipeline 3: KMeans Clustering")
    cluster_and_forecast(data, pipeline_3, mae_across_all_pipelines)
    print("\nRunning Pipeline 4: Hierarchical Clustering")
    cluster_and_forecast(data, pipeline_4, mae_across_all_pipelines)
    print("\nRunning Pipeline 5: HDBSCAN Clustering")
    cluster_and_forecast(data, pipeline_5, mae_across_all_pipelines)
    print("\nRunning Pipeline 6: DBSCAN Clustering")
    cluster_and_forecast(data, pipeline_6, mae_across_all_pipelines)

    print("\nMAE Comparison Across Pipelines:")
    for pipeline_name, mae_value in mae_across_all_pipelines.items():
        print(f"{pipeline_name}: MAE = {mae_value:.4f} kW * h")
        
    pd.DataFrame.from_dict(mae_across_all_pipelines, orient='index', columns=['MAE']).to_csv('results_forecasting/mae_comparison.csv')