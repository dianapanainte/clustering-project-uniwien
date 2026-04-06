from sklearn.cluster import KMeans
import numpy as np
from .base import ClusteringStrategy
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import HDBSCAN
from sklearn.cluster import DBSCAN


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
    
class KMeansPlusClustering(ClusteringStrategy):
    def __init__(self, n_clusters=5, random_state=42):
        self.n_clusters = n_clusters
        self.random_state = random_state

    def fit_predict(self, df):
        print("Running KMeans++ Clustering...")
        input_data = df.drop(columns=['cluster'], errors='ignore')
        # no scaler here — pipeline already scaled the data
        model = KMeans(
            n_clusters=self.n_clusters,
            init='k-means++',
            n_init=10,
            random_state=self.random_state
        )
        return model.fit_predict(input_data)
    
class HDBSCANClustering(ClusteringStrategy):
    def __init__(self, min_cluster_size=5):
        self.min_cluster_size = min_cluster_size

    def fit_predict(self, df):
        print("Running HDBSCAN Clustering...")
        model = HDBSCAN(min_cluster_size=self.min_cluster_size)
        return model.fit_predict(df)
    
class DBSCANClustering(ClusteringStrategy):
    def __init__(self, eps=0.5, min_samples=5):
        self.eps = eps
        self.min_samples = min_samples

    def fit_predict(self, df):
        print("Running DBSCAN Clustering...")
        model = DBSCAN(eps=self.eps, min_samples=self.min_samples)
        return model.fit_predict(df)
    
class TwoStageClusteringStrategy(ClusteringStrategy):
    def __init__(self, first_stage: ClusteringStrategy, second_stage: ClusteringStrategy, min_cluster_size=10):
        self.first_stage = first_stage
        self.second_stage = second_stage
        self.min_cluster_size = min_cluster_size

    def fit_predict(self, df):
        print("Running Two-Stage Clustering...")
        first_labels = self.first_stage.fit_predict(df)

        final_labels = np.full(len(df), -1)
        sub_cluster_offset = 0

        for cluster_id in sorted(set(first_labels)):
            mask = first_labels == cluster_id
            cluster_data = df[mask]
            print(f"  Stage 1 - Cluster {cluster_id}: {mask.sum()} samples")

            if mask.sum() < self.min_cluster_size:
                print(f"  Skipping — too small for second stage")
                continue

            sub_labels = self.second_stage.fit_predict(cluster_data)
            n_sub = len(set(sub_labels)) - (1 if -1 in set(sub_labels) else 0)
            n_noise = (sub_labels == -1).sum()
            print(f"  Stage 2 - Sub-clusters: {n_sub}, Noise: {n_noise}")

            indices = np.where(mask)[0]
            for i, idx in enumerate(indices):
                if sub_labels[i] != -1:
                    final_labels[idx] = sub_labels[i] + sub_cluster_offset

            sub_cluster_offset += n_sub

        return final_labels