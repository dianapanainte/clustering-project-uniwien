import numpy as np
from sklearn.discriminant_analysis import StandardScaler
from .base import OutlierStrategy, ZeroValueStrategy, NormalizationStrategy
import pandas as pd

# Outlier Management Implementations
class IQRSelection(OutlierStrategy):
    def handle(self, df):
        return df.clip(lower=df.quantile(0.05), upper=df.quantile(0.95), axis=1)

class ZScoreSelection(OutlierStrategy):
    def handle(self, df):
        from scipy import stats
        return df[(np.abs(stats.zscore(df)) < 3).all(axis=1)]
    
class NoOutlierHandling(OutlierStrategy):
    def handle(self, df):
        return df


# Zero Value Implementations
class MeanImputation(ZeroValueStrategy):
    def handle(self, df):
        return df.replace(0, df.mean())
    
class NoImputation(ZeroValueStrategy):
    def handle(self, df):
        return df.replace(0, 0)


# Normalization Implementations
class StandardScaling(NormalizationStrategy):
    def transform(self, df):
        scaler = StandardScaler()
        return pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    
class MinMaxScaling(NormalizationStrategy):
    def transform(self, df):
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        return pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    
class NoScaling(NormalizationStrategy):
    def transform(self, df):
        return df