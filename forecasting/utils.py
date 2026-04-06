import numpy as np
import pandas as pd
import sys


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

def main():
    print(f'Comparing {sys.argv[1]} with {sys.argv[2]}')
    mae_value = calculate_mae(sys.argv[1], sys.argv[2])
    print(f"MAE = {mae_value:.4f} kW * h")


if __name__ == '__main__':
    main()
