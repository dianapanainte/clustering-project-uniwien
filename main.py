from pathlib import Path
import pandas as pd

from clustering_module.features import PCAFeatures
from clustering_module.pipeline import ClusteringPipeline
from clustering_module.clustering import DBSCANClustering, HDBSCANClustering, HierarchicalClustering, KMeansClustering, KMeansPlusClustering, TwoStageClusteringStrategy
from clustering_module.preprocessing import IQRMeanConsumption, LogImputation, MinMaxScaling
from forecasting.compute_mae import calculate_mae
from forecasting.predict_xgboost import HouseholdForecaster

CSV_FILE = 'sample_23.csv'

def load_data(file_name: str, folder: str = "data") -> pd.DataFrame:
    path = Path(folder) / file_name
    return pd.read_csv(path)

data = load_data(CSV_FILE)

pipeline_1 = ClusteringPipeline(
    feature_strategy=PCAFeatures(n_components=15),
    zero_strategy=LogImputation(),
    outlier_strategy=IQRMeanConsumption(),
    norm_strategy=MinMaxScaling(),
    cluster_strategy=KMeansPlusClustering(n_clusters=3)
)

def cluster_and_forecast(data: pd.DataFrame, pipeline: ClusteringPipeline, mae_across_all_pipelines: dict = None):
    results = pipeline.run(data)

    forecasting_model = HouseholdForecaster()
    filenames = []

    for cluster_id in results['cluster'].unique():
        cluster_data = results[results['cluster'] == cluster_id]
        cluster_data.to_csv(f"results_forecasting/cluster_{cluster_id}.csv", index=False)
        forecasting_model.fit(f"results_forecasting/cluster_{cluster_id}.csv")
        forecasting_model.predict(csv_path=f"results_forecasting/forecast_24_cluster_{cluster_id}.csv")
        filenames.append(f"results_forecasting/forecast_24_cluster_{cluster_id}.csv")

    dfs = [pd.read_csv(f, index_col='ID') for f in filenames]
    combined = pd.concat(dfs)
    combined.to_csv('results_forecasting/forecast_24.csv', index=True)
    print(f'Prediction written in "forecast_24.csv"')

    mae_value, mean_actual = calculate_mae('results_forecasting/forecast_24.csv', './data/sample_24.csv')
    print(f"Mean = {mean_actual:.4f} kW * h")
    print(f"MAE = {mae_value:.4f} kW * h")
    print(f"Normalized MAE = {mae_value / mean_actual * 100:.2f}%")

    mae_across_all_pipelines[pipeline.cluster_strategy.__class__.__name__] = mae_value

if __name__ == "__main__":
    mae_across_all_pipelines = {}
    print("Running Pipeline 1: KMeans++ Clustering")
    cluster_and_forecast(data, pipeline_1, mae_across_all_pipelines)