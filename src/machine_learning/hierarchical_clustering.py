import numpy as np
import pandas as pd
import math

from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans, MiniBatchKMeans, AgglomerativeClustering
from sklearn.preprocessing import normalize
from sklearn.metrics import silhouette_score, pairwise_distances, pairwise_distances_argmin_min
from sklearn.utils import check_array

from scipy.spatial.distance import cdist
from scipy.sparse import vstack, csr_matrix

from collections import Counter

from src.machine_learning.cpu.ml import AgglomerativeClusteringCPU, KMeansCPU, MiniBatchKMeansCPU

                 
class DivisiveHierarchicalClustering():
    
    def __init__(self, model=None, model_clustering_type=None, model_linear_type=None):
        self.model = model
        self.model_clustering_type = model_clustering_type
        self.model_linear_type = model_linear_type
    
    @classmethod
    def fit(cls, X):
        
        def compute_n_iter(num_classes):
            return 100 + (10 * num_classes)
            
        def compute_branching_factor(L, k_min=2, k_max=20):
            return min(max(L // 100, k_min), k_max)

        def tokenize_labels(labels_list):
            seen = {}  # Dictionary to store unique labels and their numeric values
            numeric_values = []

            for token in labels_list:
                if token not in seen:
                    seen[token] = len(seen)  # Assign a new numeric value
                numeric_values.append(seen[token])

            return np.array(numeric_values)
        
        def checkHowManyTimesALabelIsChosen(labels_list):
            return list(Counter(labels_list).items())
        
        def execute_pipeline(X, min_leaf_size=200, max_leaf_size=500, random_state=0, spherical=True):
            
            # Embeddings, Label To Filter
            def get_embeddings_from_cluster_label(X, Y, label):
                return X[Y == label], np.where(Y == label)[0]
            
            # Compute WCSS (Intra Cluster Invariance)
            def variance_score(X):
                return np.sum((X - X.mean(axis=0))**2)
            
            # Compute Silhouette Score
            def silhouette_score_sklearn(X, Y):
                return silhouette_score(X, Y)

            n_splits = compute_branching_factor(X.shape[0])
            n_iter = compute_n_iter(n_splits)
            
            X_normalized = normalize(X)
            # Y = MiniBatchKMeansCPU.fit(X_normalized, {'n_clusters':n_splits, 'max_iter':n_iter, 'random_state':random_state}).model.labels_
            kmeans_model = KMeansCPU.fit(X_normalized, {'n_clusters':n_splits, 'max_iter':n_iter, 'random_state':random_state}).model
            Y = kmeans_model.labels_
            
            # print(np.unique(Y))
            # print(len(kmeans_model.cluster_centers_))
            
            initial_kmeans_clusters_centers = kmeans_model.cluster_centers_
            
            new_combined_labels = [None] * X.shape[0]
            centroid_list = []
            
            for indice in np.unique(Y):
                
                emb, inds = get_embeddings_from_cluster_label(X, Y, indice)
                n_ind = len(inds)
                
                # print(indice, n_ind)
                
                wcss = variance_score(emb.toarray())
                print(indice, n_ind, wcss)
                
                # print(indice, n_ind)
                
                if (n_ind >= min_leaf_size and n_ind <= max_leaf_size) or n_ind >= max_leaf_size:
                    
                    n_splits = compute_branching_factor(n_ind)
                    n_iter = compute_n_iter(n_splits)
                    
                    emb_normalized = normalize(emb)
                    # kmeans_labels = MiniBatchKMeansCPU.fit(emb_normalized, {'n_clusters':n_splits, 'max_iter':n_iter, 'random_state':random_state}).model.labels_
                    kmeans_model = KMeansCPU.fit(emb_normalized, {'n_clusters':n_splits, 'max_iter':n_iter, 'random_state':random_state}).model
                    
                    kmeans_labels = kmeans_model.labels_ 
                    
                    centroid_list.extend(kmeans_model.cluster_centers_)
                    
                    # Combine the original cluster label (indice) with the new sub-cluster label (a, b, ...)                    
                    for idx, label in zip(inds, kmeans_labels):
                        new_combined_labels[idx] = f"{indice}{chr(65 + int(label))}"  # 'A', 'B', etc.
                        
                else:
                    
                    centroid_list.append(initial_kmeans_clusters_centers[indice])
                    # If the cluster does not undergo clustering, keep the original label
                    for i in inds:
                        new_combined_labels[i] = f"{indice}"  # Keep original label  
                    
            
            sil_score = silhouette_score_sklearn(X, tokenize_labels(new_combined_labels))
            
            # print(checkHowManyTimesALabelIsChosen(new_combined_labels))
            
            return np.array(centroid_list), tokenize_labels(new_combined_labels), sil_score
        
        centroid_array, labels, sil_score = execute_pipeline(X)
        
        print(checkHowManyTimesALabelIsChosen(labels))
        
        return centroid_array, labels, sil_score
        
        """
        agglomerative_model = AgglomerativeClusteringCPU.fit(centroid_array, {'n_clusters': 50, 'linkage': 'ward', 'metric': 'euclidean'}).model
        agglomerative_labels = agglomerative_model.labels_     
           
        closest_centroids = np.argmin(cdist(X.toarray(), centroid_array), axis=1)
        mapped_labels = agglomerative_labels[closest_centroids]
        
        sil_score = silhouette_score(X, mapped_labels)
        
        print(sil_score)
    
        # MiniBatchKMneas, ['1A' '12I' '12I' ... '4Q' '4E' '4Q'] 0.002835875033179563
        # Normalized MiniBatchKMeans['1B' '12D' '12D' ... '4I' '4I' '4I'] 0.0034478190115853004
        
        # Kmeans, ['10C' '10C' '10C' ... '19T' '5B' '19T'] 0.008872945873208136
        # ---> Spherical Normalized KMeans, ['10B' '10B' '10F' ... '19D' '5A' '19D'] 0.011431971189709515 (one rodeo), min_leaf_size = 10, max_leaf_size=50, 0.0124846
        
        # Agglomerative Clustering, 0.0017651454611719664     
        """

    @classmethod
    def fit_template(cls, X):
        
        def compute_n_iter(num_classes):
            return 100 + (10 * num_classes)
            
        def compute_branching_factor(L, k_min=2, k_max=16):
            return min(max(L // 100, k_min), k_max)

        def tokenize_labels(labels_list):
            seen = {}  # Dictionary to store unique labels and their numeric values
            numeric_values = []

            for token in labels_list:
                if token not in seen:
                    seen[token] = len(seen)  # Assign a new numeric value
                numeric_values.append(seen[token])

            return np.array(numeric_values)
        
        def checkHowManyTimesALabelIsChosen(labels_list):
            return list(Counter(labels_list).items())
        
        def merge_small_clusters(X, labels, min_cluster_size=10):
            """
            Merges clusters with fewer than `min_cluster_size` points based on their distance to centroids.

            Parameters:
            - X: Data points (numpy array of shape [n_samples, n_features]).
            - labels: Array of cluster labels for each data point.
            - min_cluster_size: Minimum size for a cluster to not be merged (default is 10).

            Returns:
            - Updated labels array after merging small clusters.
            """
            # Get unique cluster labels
            unique_labels = np.unique(labels)
            
            # Exclude centroids of clusters with fewer than `min_cluster_size`
            valid_labels = [label for label in unique_labels if np.sum(labels == label) >= min_cluster_size]
            
            # Compute centroids of each cluster
            centroids = np.array([
                X[labels == label].mean(axis=0).A.flatten()  # Convert mean to dense
                for label in unique_labels
            ])
        
            # Create a copy of the labels to update
            updated_labels = labels.copy()

            for label in unique_labels:                
                cluster_indices = np.where(labels == label)[0]
                
                if len(cluster_indices) <= min_cluster_size:
                    # Get the points in the small cluster
                    cluster_points = X[cluster_indices]
                    
                    # Compute distances of the small cluster points to all valid centroids
                    distances = pairwise_distances_argmin_min(cluster_points, centroids)[1]
                    
                    # Assign the points in the small cluster to the nearest valid centroid
                    closest_centroid_idx = np.argmin(distances)
                    
                    # Assign the small cluster's points to the closest cluster
                    updated_labels[labels == label] = valid_labels[closest_centroid_idx]
            
            return updated_labels
        
        # Depth = 1, 0.005757
        # Depth = 2, 0.008548
        # Depth = 3, 0.011572, clearing min=10, 0.012697098340191351 127
        # Depth = 4, 0.011774 
        def execute_pipeline(X, depth=3, max_leaf_size=100, prefix="", random_state=0, inertia_threshold=1e-3, silhouette_threshold=0, spherical=True):
            
            X_size = X.shape[0]
            
            if X_size < max_leaf_size or depth == 0:
                return {i: prefix for i in range(X_size)}
            
            n_splits = compute_branching_factor(X.shape[0])
            n_iter = compute_n_iter(n_splits)
            
            if spherical:
                X = normalize(X, norm="l2", axis=1)
                
            kmeans_model = KMeansCPU.fit(X, {'n_clusters':n_splits, 'max_iter':n_iter, 'random_state':random_state}).model
            cluster_labels = kmeans_model.labels_
            
            # Store indices for each cluster
            labels_dict = {}
            unique_clusters = np.unique(cluster_labels)
            
            for i, cluster_id in enumerate(unique_clusters):
                cluster_indices = np.where(cluster_labels == cluster_id)[0]
                new_prefix = prefix + chr(67 + i)
                
                # Recursively cluster each group
                sub_labels = execute_pipeline(X[cluster_indices], depth - 1, max_leaf_size, new_prefix, random_state, spherical)
                
                # Map back to global indices
                for local_idx, global_idx in enumerate(cluster_indices):
                    labels_dict[global_idx] = sub_labels[local_idx]                    
                    
            return labels_dict            
        
        labels_dict = execute_pipeline(X)
        
        labels_list = [None] * len(labels_dict)
        
        for idx, label in labels_dict.items():
            labels_list[int(idx)] = label
        
        counter_list = checkHowManyTimesALabelIsChosen(labels_list)

        merged_labels_list = merge_small_clusters(X, np.array(labels_list), min_cluster_size=50)
        
        counter_list = checkHowManyTimesALabelIsChosen(merged_labels_list)
        print(counter_list, len(counter_list))
        
        print(silhouette_score(X, merged_labels_list), len(counter_list))
        
        return merged_labels_list
                    
    
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
