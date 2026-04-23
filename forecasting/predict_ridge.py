import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
import sys


def forecast_fast_autocorr(input_file, output_file):
    df = pd.read_csv(input_file)
    ids = df['ID']
    data_23 = df.drop(columns=['ID']).values 
    
    date_cols_24 = pd.date_range(start='2024-01-01', periods=366).strftime('%Y-%m-%d').tolist()
    results = []

    for i in range(len(data_23)):
        y = data_23[i]
        X = []
        target = []
        
        for t in range(30, 365):
            lag1 = y[t-1]
            lag7 = y[t-7]
            roll_mean = np.mean(y[t-30:t])
            X.append([lag1, lag7, roll_mean])
            target.append(y[t])
        
        model = Ridge(alpha=1.0)
        model.fit(X, target)
        
        history = list(y[-30:])
        forecast_24 = []

        for _ in range(366):
            lag1 = history[-1]
            lag7 = history[-7]
            roll_mean = np.mean(history[-30:])
            
            pred = model.predict([[lag1, lag7, roll_mean]])[0]
            pred = np.clip(pred, 0, np.max(y) * 1.5)
            
            forecast_24.append(pred)
            history.append(pred)
            
        row_dict = {'ID': int(ids.iloc[i])}
        row_dict.update(dict(zip(date_cols_24, forecast_24)))
        results.append(row_dict)

    final_df = pd.DataFrame(results)
    final_df[['ID'] + date_cols_24].to_csv(output_file, index=False)

forecast_fast_autocorr(sys.argv[1], sys.argv[2])
