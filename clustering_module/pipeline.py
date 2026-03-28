import pandas as pd
from .clustering import ClusteringStrategy
from .preprocessing import ZeroValueStrategy, OutlierStrategy, NormalizationStrategy

class ClusteringPipeline:
    def __init__(self, 
                 zero_strategy: ZeroValueStrategy,
                 outlier_strategy: OutlierStrategy,
                 norm_strategy: NormalizationStrategy,
                 cluster_strategy: ClusteringStrategy):
        self.zero_strategy = zero_strategy
        self.outlier_strategy = outlier_strategy
        self.norm_strategy = norm_strategy
        self.cluster_strategy = cluster_strategy

    def run(self, df: pd.DataFrame):
        print("Starting Pipeline...")
        data = df.copy()
        
        # Execute steps in order
        data = self.zero_strategy.handle(data)
        data = self.outlier_strategy.handle(data)
        data = self.norm_strategy.transform(data)
        
        clusters = self.cluster_strategy.fit_predict(data)
        
        # Add cluster labels back to the original dataframe
        df['cluster'] = clusters
        return df