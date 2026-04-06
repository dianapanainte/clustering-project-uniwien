import pandas as pd
from prophet import Prophet
from joblib import Parallel, delayed
import numpy as np


df = pd.read_csv('sample_23.csv', sep=',')

def calculate_mae(forecast_file, test_file):
    forecast = pd.read_csv(forecast_file, index_col=0)
    test = pd.read_csv(test_file, index_col=0)
    
    f_long = forecast.stack().reset_index()
    f_long.columns = ['ID', 'Date', 'Prediction']
    
    t_long = test.stack().reset_index()
    t_long.columns = ['ID', 'Date', 'Actual']
    
    merged = pd.merge(f_long, t_long, on=['ID', 'Date'])
    mae = np.mean(np.abs(merged['Actual'] - merged['Prediction']))
    
    return mae

def calculate_average_consumption(df, by_household=True):
    if 'ID' in df.columns:
        df = df.set_index('ID')
    
    if by_household:
        return df.mean(axis=1)
    else:
        return df.stack().mean()

def predict_household(row):
    hh_id = row['ID']
    y_values = row.drop('ID')
    
    data = pd.DataFrame({
        'ds': pd.to_datetime(y_values.index),
        'y': y_values.values
    })
    
    model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
    model.fit(data)
    
    future = model.make_future_dataframe(periods=366, freq='D')
    forecast = model.predict(future)
    
    forecast_24 = forecast.loc[forecast['ds'] >= '2024-01-01', ['ds', 'yhat']].copy()
    
    prediction_dict = dict(zip(forecast_24['ds'].dt.strftime('%Y-%m-%d'), forecast_24['yhat']))
    prediction_dict['ID'] = int(hh_id)
    
    return prediction_dict


results = Parallel(n_jobs=-1)(delayed(predict_household)(row) for _, row in df.iterrows())

final_df = pd.DataFrame(results)

cols = ['ID'] + [c for c in final_df.columns if c != 'ID']
final_df = final_df[cols]

final_df.to_csv('forecast_24.csv', index=False)

avg_per_home = calculate_average_consumption(pd.read_csv('sample_24.csv', sep=','), by_household=False)
print(f'Average consumption = {avg_per_home:.4f} kW * h')
mae_value = calculate_mae('forecast_24.csv', 'sample_24.csv')
print(f"MAE = {mae_value:.4f} kW * h")
