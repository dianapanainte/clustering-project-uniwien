import pandas as pd
from prophet import Prophet
from joblib import Parallel, delayed
import numpy as np
import sys


INPUT_FILE = sys.argv[1]
OUTPUT_FILE = sys.argv[2]


def predict_household(row):
    hh_id = row['ID']
    y_values = row.drop('ID')
    
    data = pd.DataFrame({
        'ds': pd.to_datetime(y_values.index),
        'y': y_values.values
    })
    data['y'] = np.log1p(data['y'])
    
    model = Prophet(
        # seasonality_mode='multiplicative',
        yearly_seasonality=False, # Energy often has strong seasonal patterns
        weekly_seasonality=True,
        daily_seasonality=True    # If your data is hourly, this is crucial
    )
    model.fit(data)
    
    future = model.make_future_dataframe(periods=366, freq='D')
    forecast = model.predict(future)
    
    forecast_24 = forecast.loc[forecast['ds'] >= '2024-01-01', ['ds', 'yhat']].copy()
    forecast_24['yhat'] = np.expm1(forecast_24['yhat'])
    
    prediction_dict = dict(zip(forecast_24['ds'].dt.strftime('%Y-%m-%d'), forecast_24['yhat']))
    prediction_dict['ID'] = int(hh_id)
    
    return prediction_dict


df = pd.read_csv(INPUT_FILE)
results = Parallel(n_jobs=-1)(delayed(predict_household)(row) for _, row in df.iterrows())

final_df = pd.DataFrame(results)

cols = ['ID'] + [c for c in final_df.columns if c != 'ID']
final_df = final_df[cols]

final_df.to_csv(OUTPUT_FILE, index=False)
