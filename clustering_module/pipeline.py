import pandas as pd
from .clustering import ClusteringStrategy
from .preprocessing import ZeroValueStrategy, OutlierStrategy, NormalizationStrategy
from .features import FeatureStrategy  




class ClusteringPipeline:

    def __init__(self,
                 feature_strategy: FeatureStrategy,  # add this
                 zero_strategy: ZeroValueStrategy,
                 outlier_strategy: OutlierStrategy,
                 norm_strategy: NormalizationStrategy,
                 cluster_strategy: ClusteringStrategy):
        self.feature_strategy = feature_strategy      # add this
        self.zero_strategy = zero_strategy
        self.outlier_strategy = outlier_strategy
        self.norm_strategy = norm_strategy
        self.cluster_strategy = cluster_strategy

    def run(self, df: pd.DataFrame):
        print("Starting Pipeline...")
        data = df.copy()
        data = self.feature_strategy.compute(data)
        data = self.zero_strategy.handle(data)
        data = self.outlier_strategy.handle(data)
        data = self.norm_strategy.transform(data)
        clusters = self.cluster_strategy.fit_predict(data)
        result = df.copy()        # copy original, don't modify it
        result['cluster'] = clusters
        return result
    
    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        data = df.copy()
        data = self.feature_strategy.extract(data)
        data = self.zero_strategy.handle(data)
        data = self.outlier_strategy.handle(data)
        data = self.norm_strategy.transform(data)
        return data