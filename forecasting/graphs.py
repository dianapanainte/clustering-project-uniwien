import matplotlib.pyplot as plt
import numpy as np

methods = ['XGBoost', 'Profile-Based', 'Naive', 'Ridge', 'LightGBM', 'Prophet']
mae_values = [3.3360, 3.7610, 3.9149, 4.3169, 4.3192, 4.8577]

data = sorted(zip(methods, mae_values), key=lambda x: x[1])
sorted_methods, sorted_mae = zip(*data)

plt.figure(figsize=(10, 6))
bars = plt.bar(sorted_methods, sorted_mae, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b'])

plt.ylabel('MAE (kWh)')
plt.title('Comparison of Forecasting Methods\nDataset-level')
plt.xticks(rotation=45, ha='right')

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 2), va='bottom', ha='center', fontsize=10)

plt.tight_layout()
plt.savefig('mae_cmp.png')

# Cluster 0

methods = ['LightGBM', 'LSTM', 'Ridge', 'Croston', 'Profile-Based', 'XGBoost']
mae_values = [6.8124, 2.8576, 1.8060, 1.2435, 1.1122, 1.0533]

data = sorted(zip(methods, mae_values), key=lambda x: x[1])
sorted_methods, sorted_mae = zip(*data)

plt.figure(figsize=(10, 6))
bars = plt.bar(sorted_methods, sorted_mae, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b'])

plt.ylabel('MAE (kWh)')
plt.title('Comparison of Forecasting Methods\nCluster #0')
plt.xticks(rotation=45, ha='right')

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 2), va='bottom', ha='center', fontsize=10)

plt.tight_layout()
plt.savefig('mae_cmp_cluster_0.png')

# Cluster 1

methods = ['LightGBM', 'LSTM', 'Ridge', 'Croston', 'Profile-Based', 'XGBoost']
mae_values = [2.7814, 3.3517, 2.6689, 2.8379, 2.3812, 2.3564]

data = sorted(zip(methods, mae_values), key=lambda x: x[1])
sorted_methods, sorted_mae = zip(*data)

plt.figure(figsize=(10, 6))
bars = plt.bar(sorted_methods, sorted_mae, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b'])

plt.ylabel('MAE (kWh)')
plt.title('Comparison of Forecasting Methods\nCluster #1')
plt.xticks(rotation=45, ha='right')

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 2), va='bottom', ha='center', fontsize=10)

plt.tight_layout()
plt.savefig('mae_cmp_cluster_1.png')

# Cluster 2

methods = ['LightGBM', 'LSTM', 'Ridge', 'Croston', 'Profile-Based', 'XGBoost']
mae_values = [10.8708, 20.2043, 12.2306, 15.7740, 9.7707, 8.2451]

data = sorted(zip(methods, mae_values), key=lambda x: x[1])
sorted_methods, sorted_mae = zip(*data)

plt.figure(figsize=(10, 6))
bars = plt.bar(sorted_methods, sorted_mae, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b'])

plt.ylabel('MAE (kWh)')
plt.title('Comparison of Forecasting Methods\nCluster #2')
plt.xticks(rotation=45, ha='right')

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 2), va='bottom', ha='center', fontsize=10)

plt.tight_layout()
plt.savefig('mae_cmp_cluster_2.png')
