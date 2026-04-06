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
    
class DBSCANClustering(ClusteringStrategy):
    def __init__(self, eps=0.5, min_samples=5):
        self.eps = eps
        self.min_samples = min_samples
        
    def fit_predict(self, df):
        from sklearn.cluster import DBSCAN
        print("Running DBSCAN Clustering...")
        model = DBSCAN(eps=self.eps, min_samples=self.min_samples)
        return model.fit_predict(df)