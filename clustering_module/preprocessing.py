import numpy as np
from sklearn.discriminant_analysis import StandardScaler
from .base import OutlierStrategy, ZeroValueStrategy, NormalizationStrategy
import pandas as pd

# Outlier Management Implementations
class IQRSelection(OutlierStrategy):
    def handle(self, df):
        return df.clip(lower=df.quantile(0.05), upper=df.quantile(0.95), axis=1)
    # TODO - Discuss what exactly are the percentages for outliers. 

class IQRMeanConsumption(OutlierStrategy):
    def handle(self, df):
        Q1 = df.quantile(0.25)
        Q3 = df.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return df.apply(lambda x: np.where((x < lower_bound[x.name]) | (x > upper_bound[x.name]), x.mean(), x))

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
    
class LogImputation(ZeroValueStrategy):
    def handle(self, df):
        return df.replace(0, np.log1p(df[df > 0].mean()))
    
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
    
# Custom Implementations
class RollingOutlierHandling(OutlierStrategy):
    def __init__(self, k=3, window=7):
        self.k = k
        self.window = window

    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.dropna(how='all')
        df_t = df.T
        r_mean = df_t.rolling(window=self.window, center=True).mean()
        r_std = df_t.rolling(window=self.window, center=True).std()
        df_clean_t = df_t.clip(lower=r_mean - (self.k * r_std),
                                upper=r_mean + (self.k * r_std))
        df_clean_t = df_clean_t.ffill().bfill().fillna(0)
        return df_clean_t.T


class ShapeScaling(NormalizationStrategy):
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        scaled = df.sub(df.mean(axis=1), axis=0).div(df.std(axis=1), axis=0)
        return scaled.fillna(0)  # households with zero variance become 0


class HybridScaling(NormalizationStrategy):
    def __init__(self, alpha=0.5):
        self.alpha = alpha

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        mins = df.min(axis=1)
        maxs = df.max(axis=1)
        shape_only = df.sub(mins, axis=0).div(maxs - mins, axis=0)
        return (shape_only * (1 - self.alpha)) + (df * self.alpha)