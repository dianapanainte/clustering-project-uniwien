from pathlib import Path
import pandas as pd

from clustering_module.features import PCAFeatures
from clustering_module.pipeline import ClusteringPipeline
from clustering_module.clustering import KMeansPlusClustering
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
results = pipeline_1.run(data)

forecasting_model = HouseholdForecaster()
filenames = []

for cluster_id in results['cluster'].unique():
    cluster_data = results[results['cluster'] == cluster_id]
    cluster_data.to_csv(f"cluster_{cluster_id}.csv", index=False)
    forecasting_model.fit(f"cluster_{cluster_id}.csv")
    forecasting_model.predict(csv_path=f"forecast_24_cluster_{cluster_id}.csv")
    filenames.append(f"forecast_24_cluster_{cluster_id}.csv")

dfs = [pd.read_csv(f, index_col='ID') for f in filenames]
combined = pd.concat(dfs)
combined.to_csv('forecast_24.csv', index=True)
print(f'Prediction written in "forecast_24.csv"')

mae_value, mean_actual = calculate_mae('forecast_24.csv', './data/sample_24.csv')
print(f"Mean = {mean_actual:.4f} kW * h")
print(f"MAE = {mae_value:.4f} kW * h")
print(f"Normalized MAE = {mae_value / mean_actual * 100:.2f}%")
