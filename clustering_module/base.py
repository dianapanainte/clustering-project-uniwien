from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# --- 1. Outlier Management Interface ---
class OutlierStrategy(ABC):
    @abstractmethod
    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        pass

# --- 2. Zero Value Management Interface ---
class ZeroValueStrategy(ABC):
    @abstractmethod
    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        pass

# --- 3. Normalization Interface ---
class NormalizationStrategy(ABC):
    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        pass

# --- 4. Clustering Interface ---
class ClusteringStrategy(ABC):
    @abstractmethod
    def fit_predict(self, df: pd.DataFrame) -> np.ndarray:
        pass