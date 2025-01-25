import numpy as np
import pandas as pd

from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans, MiniBatchKMeans, AgglomerativeClustering
from sklearn.preprocessing import normalize
from sklearn.metrics import silhouette_score, pairwise_distances


"""
    Key Features:
        - Linkage Criteria: Single, Complete, Average, Ward's;
        - Distance Matrics: Euclidean, Manhattan, Cosine, etc;
        - Data PreProcessing: Scaling, Dimensionality reduction, handling missing data;
        - Tree Representation
        
        
    given a dataset (d1, d2, d3, ....dN) of size N
    at the top we have all data in one cluster
    the cluster is split using a flat clustering method eg. K-Means etc
    repeat
    choose the best cluster among all the clusters to split
    split that cluster by the flat clustering algorithm
    until each data is in its own singleton cluster   
    
    Utilizar K-Means Clustering do cuml como modelo para Hierarchical Clustering GPU - Divise Approach  
""" 

"""            
    Pecos params:
            
    nr_splits (int, optional): The out-degree of each internal node of the tree. Default is `16`.
            
    min_codes (int): The number of direct child nodes that the top level of the hierarchy should have.
            
    max_leaf_size (int, optional): The maximum size of each leaf node of the tree. Default is `100`.
            
    spherical (bool, optional): True will l2-normalize the centroids of k-means after each iteration. Default is `True`.
            
    seed (int, optional): Random seed. Default is `0`.
            
    kmeans_max_iter (int, optional): Maximum number of iterations for each k-means problem. Default is `20`.
            
    threads (int, optional): Number of threads to use. `-1` denotes all CPUs. Default is `-1`.
            
    do_sample (bool, optional): Do sampling if is True. Default is False.
    We use linear sampling strategy with warmup, which linearly increases sampling rate from `min_sample_rate` to `max_sample_rate`.
    The top (total_layer * `warmup_ratio`) layers are warmup_layers which use a fixed sampling rate `min_sample_rate`.
    The sampling rate for layer l is `min_sample_rate`+max(l+1-warmup_layer,0)*(`max_sample_rate`-min_sample_rate)/(total_layers-warmup_layers).
    Please refer to 'self.get_layer_sample_rate()' function for complete definition.
            
    max_sample_rate (float, optional): the maximum samplng rate at the end of the linear sampling strategy. Default is `1.0`.
            
    min_sample_rate (float, optional): the minimum sampling rate at the begining warmup stage of the linear sampling strategy. Default is `0.1`.
    Note that 0 < min_sample_rate <= max_sample_rate <= 1.0.
            
    warmup_ratio: (float, optional): The ratio of warmup layers. 0 <= warmup_ratio <= 1.0. Default is 0.4.
    
    from soyclustering import SphericalKMeans -> Spherical Kmeans, right now normalizing the data
"""

class DivisiveHierarchicalClustering:
    
    # n_splits best case for the training data
    def __init__(self, n_splits=24, n_iter=100, min_leaf_size=10, max_leaf_size=100, spherical=True, distance_metric="cosine", linkage='complete', final_fit=False, seed=0):
        
        self.n_splits = n_splits
        self.n_iter = n_iter
        self.min_leaf_size = min_leaf_size
        self.max_leaf_size = max_leaf_size
        self.spherical = spherical
        self.distance_metric = distance_metric
        self.linkage = linkage
        self.final_fit=final_fit
        self.seed = seed
        
        self.kmeans_params = {
            'n_clusters': self.n_splits,
        }
    
    def fit(self, X, kmeans_args={}, agglo_args={}, verbose=1):
        
        self.print_verbose(f"Starting KMeans with {self.n_splits}", verbose=1)
        
        kmeans_model = self._mini_batch_kmeans_sklearn(X)
        labels = kmeans_model.labels_
        
        hybrid_clusters = []
        
        # print(labels)
        
        # labels_df = pd.DataFrame(labels,  columns=['cluster_label'])
        
        # print(labels_df.groupby('cluster_label').size().reset_index(name='count'))
        
        self.print_verbose(f"Number of labels: {len(np.unique(labels))}", verbose=verbose)
        
        for cluster_label in np.unique(labels):
            self.print_verbose(f"Cluster label number {cluster_label}", verbose=verbose)
            
            clusters_points = X[labels == cluster_label]
            if clusters_points.shape[0] > self.min_leaf_size:
                agglomerative_labels = self._agglomerative_clustering_sklearn(clusters_points.toarray()).labels_
                hybrid_clusters.append((cluster_label, agglomerative_labels))
            else:
                hybrid_clusters.append((cluster_label, clusters_points))
            
        self.print_verbose("Returning", verbose=verbose)
        
        if self.final_fit:
            centroid_distances = pairwise_distances(kmeans_model.cluster_centers_)
            # Precomputed
            final_agglomerative = self._agglomerative_clustering_sklearn(centroid_distances)
            final_labels = final_agglomerative.labels_
            return final_labels
        
        return hybrid_clusters
            
    
    def print_verbose(self, string, verbose):
        if verbose == 1:
            print(string)
    
    def _mini_batch_kmeans_sklearn(self, X):
        if self.spherical:
            X = normalize(X, norm='l2')
        return MiniBatchKMeans(n_clusters=self.n_splits, max_iter=self.n_iter, random_state=self.seed).fit(X)
    
    def _kmeans_sklearn(self, X):
        if self.spherical:
            X = normalize(X, norm='l2')
        return KMeans(n_clusters=self.n_splits, max_iter=self.n_iter, random_state=self.seed).fit(X)
    
    def _agglomerative_clustering_sklearn(self, X):
        return AgglomerativeClustering(n_clusters=None, distance_threshold=0.5, linkage=self.linkage, metric=self.distance_metric).fit(X)
    
    

class KmeansRanker():
    
    def __init__(self):
        pass
    
    """
        normalized mini_batch_kmeans, random state = 0
         
        BEST {'clusters': np.int64(24), 'silhouette_avg': np.float64(0.004969569595632096)}, 
    """
    
    def ranker(self, X):
        
        scores = []
        
        X_normalized = normalize(X, norm='l2')
        
        for i in np.arange(2, 10):
            
            # kmeans = KMeans(n_clusters=i, random_state=0).fit(X)
            kmeans = MiniBatchKMeans(n_clusters=i, random_state=0).fit(X_normalized)
            kmeans_labels = kmeans.labels_
            
            silhouette_avg = silhouette_score(X_normalized, kmeans_labels)
            
            scores.append({'clusters':i, 
                           'silhouette_avg': silhouette_avg})
            
            print("Ran KMeans", i)
            
        return scores
            
            
        
# Example usage
if __name__ == "__main__":
    # Generate synthetic data
    seed = np.random.seed(42)
    X, _ = make_blobs(n_samples=200, centers=4, n_features=10, random_state=42) # 50 samples with 2 features
    
    # Create and fit the Divisive Hierarchical K-Means model
    dhkmeans = DivisiveHierarchicalClustering()
    
    #n_splits=16, min_leaf_size=20, max_leaf_size=100, spherical=True, seed=0, kmeans_max_iter=20
    clusters = dhkmeans.fit(X)
    
    print(clusters)

    
        