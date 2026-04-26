from pathlib import Path
import pandas as pd

from clustering_module.features import PCAFeatures
from clustering_module.pipeline import ClusteringPipeline
from clustering_module.clustering import KMeansPlusClustering
from clustering_module.preprocessing import IQRMeanConsumption, LogImputation, MinMaxScaling
from forecasting.predict_xgboost import HouseholdForecaster

CSV_FILE = 'sample_23.csv'

def load_data(file_name: str, folder: str = "data") -> pd.DataFrame:
    path = Path(folder) / file_name
    return pd.read_csv(path)

data = load_data(CSV_FILE)

# Experiment 1: Mean Imputation + IQR + Standard Scaler + KMeans
pipeline_1 = ClusteringPipeline(
    feature_strategy=PCAFeatures(),
    zero_strategy=LogImputation(),
    outlier_strategy=IQRMeanConsumption(),
    norm_strategy=MinMaxScaling(),
    cluster_strategy=KMeansPlusClustering(n_clusters=3)
)

results = pipeline_1.run(data)

# --- THE FORECASTING PART ---
forecasting_model = HouseholdForecaster(
    xgb_params={"n_estimators": 300, "device": "cuda"}
)

for cluster_id in results['cluster'].unique():
    cluster_data = results[results['cluster'] == cluster_id]
    cluster_data.to_csv(f"cluster_{cluster_id}.csv", index=False)
    forecasting_model.fit(f"cluster_{cluster_id}.csv")
    df = forecasting_model.predict(csv_path=f"forecast_24_cluster_{cluster_id}.csv")
