import numpy as np

from sklearn.cluster import AgglomerativeClustering, KMeans, Birch

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
class HierarchicalClustering:
    
    def __init__(self):
        
        """
            nr_splits,
            
        """
        
        pass
    
    @classmethod
    def fit():
        pass
    

class DivisiveHierarchicalClustering:
    
    def __init__(self, max_clusters=2):
        """
            :param max_clusters: Maximum number of clusters to form.
        """
        self.max_clusters = max_clusters
        self.dendrogram = []
        
        
    def fit(self, data):
        """
            :param data: NumPy array of shape (n_samples, n_features).
            :return: List of clusters with indices of data points.
        """
        
        clusters = [np.arange(len(data))] # Start will all points in one cluster
        
        # print(clusters)
        
        self.dendrogram.append((None, clusters[0])) # Root Cluster
        
        # 1 - 16
        while len(clusters) < self.max_clusters:
            # Select the largerst cluster to split
            cluster_to_split = max(clusters, key=len)
            
            # print(cluster_to_split)
            
            # Split the selected cluster into two using K-Means
            cluster_indices = cluster_to_split
            cluster_data = data[cluster_indices]
            
            # print(cluster_data)
            
            # Clustering Algorithm
            kmeans = KMeans(n_clusters=2, random_state=42)
            labels = kmeans.fit_predict(cluster_data)
            
            # Create new clusters
            cluster_1 = cluster_indices[labels == 0]
            cluster_2 = cluster_indices[labels == 1]
            
            # print(cluster_1, cluster_2)
            
            # Replace the original cluster with the two subclusters
            clusters.remove(cluster_to_split)
            clusters.extend([cluster_1, cluster_2])
            
            print(clusters)
            
            self.dendrogram.append((cluster_to_split, cluster_1, cluster_2))
            
            # print(self.dendrogram)
            
        return clusters
        
        
# Example usage
if __name__ == "__main__":
    # Generate synthetic data
    seed = np.random.seed(42)
    data = np.random.rand(100, 2)  # 100 samples with 2 features

    # print(data)
    
    # Perform divisive hierarchical clustering
    model = DivisiveHierarchicalClustering(max_clusters=16)
    clusters = model.fit(data)

    # Print cluster indices
    for i, cluster in enumerate(clusters):
        print(f"Cluster {i + 1}: {cluster}")
        