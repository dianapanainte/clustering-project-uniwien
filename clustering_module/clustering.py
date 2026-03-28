from sklearn.cluster import KMeans
from .base import ClusteringStrategy


class KMeansClustering(ClusteringStrategy):
    def __init__(self, n_clusters=3):
        self.n_clusters = n_clusters
        
    def fit_predict(self, df):
        print("Running KMeans Clustering...")
        model = KMeans(n_clusters=self.n_clusters, n_init=10)
        return model.fit_predict(df)
    
class HierarchicalClustering(ClusteringStrategy):
    def __init__(self, n_clusters=3):
        self.n_clusters = n_clusters
        
    def fit_predict(self, df):
        from sklearn.cluster import AgglomerativeClustering
        print("Running Agglomerative Clustering...")
        model = AgglomerativeClustering(n_clusters=self.n_clusters)
        return model.fit_predict(df)