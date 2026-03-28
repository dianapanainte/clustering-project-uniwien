from sklearn.discriminant_analysis import StandardScaler
from .base import OutlierStrategy, ZeroValueStrategy, NormalizationStrategy
import pandas as pd

# Outlier Management Implementations
class IQRSelection(OutlierStrategy):
    def handle(self, df):
        # Logic to remove/clip outliers using IQR
        return df.clip(lower=df.quantile(0.05), upper=df.quantile(0.95), axis=1)

# Zero Value Implementations
class MeanImputation(ZeroValueStrategy):
    def handle(self, df):
        return df.replace(0, df.mean())

# Normalization Implementations
class StandardScaling(NormalizationStrategy):
    def transform(self, df):
        scaler = StandardScaler()
        return pd.DataFrame(scaler.fit_transform(df), columns=df.columns)