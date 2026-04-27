import numpy as np
import pandas as pd
import sys


df_2023 = pd.read_csv(sys.argv[1], index_col=0)

household_means = df_2023.mean(axis=1)
df_normalised = df_2023.div(household_means, axis=0)

cluster_profile = df_normalised.mean(axis=0)

forecast_profile_366 = np.interp(
    np.linspace(0, 364, 366),
    np.arange(365),
    cluster_profile.values
)

df_2024_forecast = pd.DataFrame(
    np.outer(household_means.values, forecast_profile_366),
    index=df_2023.index,
    columns=pd.date_range('2024-01-01', periods=366).strftime('%Y-%m-%d')
)

df_2024_forecast.to_csv(sys.argv[2])
