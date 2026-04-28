import sys
import pandas as pd
import numpy as np


df = pd.read_csv(sys.argv[1], index_col='ID')
forecast_dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
print(f'Forecast days: {len(forecast_dates)}')


def croston(series, forecast_horizon):
    alpha = 0.1
    series = series.values

    non_zero = series[series > 0]
    q = non_zero[0] if len(non_zero) > 0 else 0
    p = len(series) / (len(non_zero) + 1e-9)

    for val in series:
        if val > 0:
            q = alpha * val + (1 - alpha) * q
            p = alpha * 1   + (1 - alpha) * p
        else:
            p = alpha * (p + 1) + (1 - alpha) * p

    forecast_value = q / p
    return np.full(forecast_horizon, forecast_value)


forecasts = {}
for hh_id, row in df.iterrows():
    forecasts[hh_id] = croston(row, len(forecast_dates))

forecast_df = pd.DataFrame(forecasts, index=forecast_dates).T
forecast_df.index.name = 'ID'
forecast_df.columns = forecast_df.columns.strftime('%Y-%m-%d')

forecast_df.to_csv(sys.argv[2])
print(f'Forecast shape: {forecast_df.shape}')
