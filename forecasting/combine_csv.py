import pandas as pd

files = [
    'forecast_24_xgboost_cluster_0.csv',
    'forecast_24_xgboost_cluster_1.csv',
    'forecast_24_xgboost_cluster_2.csv',
]

dfs = [pd.read_csv(f, index_col='ID') for f in files]
combined = pd.concat(dfs)
combined.to_csv('forecast_24.csv', index=True)
