import numpy as np

from sklearn.preprocessing import normalize
from sklearn.metrics import silhouette_score, pairwise_distances_argmin_min

from collections import Counter

from src.machine_learning.cpu.ml import KMeansCPU

     
class DivisiveHierarchicalClustering():
    
    def __init__(self, model_clustering_type=None, labels_=None, cluster_centroids_=None):
        self.model_clustering_type = model_clustering_type
        self.labels_ = labels_
        self.cluster_centroids_ = cluster_centroids_
    
    @classmethod
    def fit(cls, X):
        
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
        
        def get_cluster_centroids(X, labels):
            return np.array([
                X[labels == label].mean(axis=0).A.flatten()  # Convert mean to dense
                for label in np.unique(labels)
            ])    
        
        def merge_small_clusters(X, labels, min_cluster_size=10):
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

        merged_labels_list = merge_small_clusters(X, np.array(labels_list), min_cluster_size=10)
        
        counter_list = checkHowManyTimesALabelIsChosen(merged_labels_list)
        # print(counter_list, len(counter_list))
        
        print("Silhouette Score:", silhouette_score(X, merged_labels_list), "Number of Labels:", len(counter_list))
        
        return cls("Hierarchical KMeansCPU", tokenize_labels(merged_labels_list), get_cluster_centroids(X, np.array(labels_list)))
                    
