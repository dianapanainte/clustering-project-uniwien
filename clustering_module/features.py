import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from sklearn.decomposition import PCA
from .base import FeatureStrategy


# --- Base Block ---
class FeatureBlock(ABC):
    @abstractmethod
    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        pass


# --- Block 1: Volume & Volatility ---
class VolumeFeatures(FeatureBlock):
    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        features = pd.DataFrame(index=df.index)
        features['avg_daily_usage'] = df.mean(axis=1)
        features['day_to_day_variability'] = df.std(axis=1) / (features['avg_daily_usage'] + 1e-9)
        features['peak_intensity'] = df.max(axis=1) / (features['avg_daily_usage'] + 1e-9)
        return features


# --- Block 2: Calendar Features ---
class CalendarFeatures(FeatureBlock):
    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        features = pd.DataFrame(index=df.index)
        dates = pd.to_datetime(df.columns)
        is_weekend = dates.weekday >= 5
        features['weekend_avg'] = df.iloc[:, is_weekend].mean(axis=1)
        features['weekday_avg'] = df.iloc[:, ~is_weekend].mean(axis=1)
        features['weekend_bias'] = features['weekend_avg'] / (features['weekday_avg'] + 1e-9)
        return features


# --- Block 3: Seasonal PAA Blocks ---
class SeasonalFeatures(FeatureBlock):
    def __init__(self, n_segments=4):
        self.n_segments = n_segments

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        features = pd.DataFrame(index=df.index)
        step = df.shape[1] // self.n_segments
        seasons = ['Winter_Q1', 'Spring_Q2', 'Summer_Q3', 'Autumn_Q4']
        for i in range(self.n_segments):
            features[seasons[i]] = df.iloc[:, i*step:(i+1)*step].mean(axis=1)
        return features


# --- Block 4: Trend ---
class TrendFeatures(FeatureBlock):
    def __init__(self, window=30):
        self.window = window

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        features = pd.DataFrame(index=df.index)
        features['trend'] = (
            df.iloc[:, -self.window:].mean(axis=1) -
            df.iloc[:, :self.window].mean(axis=1)
        )
        return features


# --- Block 5: PCA (optional) ---
class PCAFeatures(FeatureBlock):
    def __init__(self, n_components=3):
        self.n_components = n_components

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        features = pd.DataFrame(index=df.index)
        pca = PCA(n_components=self.n_components)
        pca_results = pca.fit_transform(df)
        for i in range(self.n_components):
            features[f'pca_{i+1}'] = pca_results[:, i]
        return features


# --- Modular Extractor ---
class ModularFeatureExtraction(FeatureStrategy):
    def __init__(self, blocks: list):
        self.blocks = blocks

    def extract(self, df: pd.DataFrame) -> pd.DataFrame:
        df_filled = df.ffill(axis=1).bfill(axis=1).fillna(0)
        features = pd.DataFrame(index=df_filled.index)
        for block in self.blocks:
            block_features = block.compute(df_filled)
            features = pd.concat([features, block_features], axis=1)
        features = features.fillna(0)  # safety net
        return features