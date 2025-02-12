import os
import numpy as np
import pickle

from sklearn.preprocessing import normalize
from sklearn.metrics import silhouette_score, pairwise_distances_argmin_min

from collections import Counter

from src.machine_learning.cpu.ml import KMeansCPU
     
class DivisiveHierarchicalClustering():
    
    def __init__(self, clustering_model_type=None, labels=None, centroids=None, gpu_usage=False):
        """
        Initialize the clustering model, labels, and centroids
        """
        self.clustering_model_type = clustering_model_type
        self.labels = labels
        self.centroids = centroids
        self.gpu_usage = gpu_usage
        
    def save(self, directory):
        """
        Saves the trained model (clustering and linear models) to the specified directory.
        """
        os.makedirs(directory, exist_ok=True)
        model_data = {
            'clustering_model_type': self.clustering_model_type,
            'labels': self.labels,
            'centroids': self.centroids,
            'gpu_usage': self.gpu_usage
        }
        with open(directory, 'wb') as fout:
            pickle.dump(model_data, fout)

    @classmethod
    def load(cls, model_path):
        """
        Loads a previously saved model from the specified path.
        """
        assert os.path.exists(model_path), f"{model_path} does not exist"
        with open(model_path, 'rb') as fclu:
            data = pickle.load(fclu)
        return cls(
            clustering_model_type=data['clustering_model_type'], 
            labels=data['labels'],
            centroids=data['centroids'],
            gpu_usage=data['gpu_usage']
        )
    
    @classmethod
    def fit(cls, X, CLUSTERING_MODEL, depth=3, max_leaf_size=100, prefix="", random_state=0, spherical=True, gpu_usage=False):
        """
        Main method to perform divisive hierarchical clustering
        """
        def compute_iterations(num_classes):
            """
            Compute the number of iterations based on the number of classes
            """
            return 100 + (10 * num_classes)
            
        def compute_branching_factor(num_samples, min_clusters=2, max_clusters=16):
            """
            Compute branching factor based on the number of samples
            """
            return min(max(num_samples // 100, min_clusters), max_clusters)

        def encode_labels(labels_list):
            """Convert string labels into numeric labels"""
            label_to_idx = {}
            return np.array([label_to_idx.setdefault(label, len(label_to_idx)) for label in labels_list])
        
        def count_label_occurrences(labels_list):
            """
            Count occurrences of each label in the list
            """
            return list(Counter(labels_list).items())
        
        def calculate_centroids(X, labels):
            """
            Calculate the centroids of the clusters based on the labels
            """
            return np.array([
                X[labels == label].mean(axis=0) # Convert mean to dense array
                for label in np.unique(labels)
            ])  
        
        def merge_small_clusters(X, labels, min_cluster_size=10):
            """
            Merge clusters that are smaller than a given threshold by reassigning their points
            """
            unique_labels = np.unique(labels)
            valid_labels = [label for label in unique_labels if np.sum(labels == label) >= min_cluster_size]
            centroids = np.array([
                X[labels == label].mean(axis=0)
                for label in unique_labels
            ])
            
            updated_labels = labels.copy()

            for label in unique_labels:                
                cluster_indices = np.where(labels == label)[0]
                if len(cluster_indices) <= min_cluster_size:
                    # Reassign points to the nearest valid centroid
                    cluster_points = X[cluster_indices]
                    distances = pairwise_distances_argmin_min(cluster_points, centroids)[1]
                    closest_centroid_idx = np.argmin(distances)
                    updated_labels[labels == label] = valid_labels[closest_centroid_idx]
            
            return updated_labels
        
        # Depth = 1, 0.005757
        # Depth = 2, 0.008548
        # Depth = 3, 0.011572, clearing min=10, 0.012697098340191351 127
        # Depth = 4, 0.011774 
        def recursive_clustering(X, depth=2, max_leaf_size=100, prefix="", random_state=0, spherical=True):
            """
            Recursively perform the clustering process
            """
            
            if X.shape[0] < max_leaf_size or depth == 0:
                return {i: prefix for i in range(X.shape[0])}
            
            n_splits = compute_branching_factor(X.shape[0])
            n_iter = compute_iterations(n_splits)
            
            if spherical:
                X = normalize(X, norm="l2", axis=1)
            
            # Fiting Clustering Model
            kmeans_model = CLUSTERING_MODEL.create_model({'n_clusters': n_splits, 'max_iter': n_iter, 'random_state': random_state}).fit(X)
                
            cluster_labels = kmeans_model.labels_
            
            labels_dict = {}
            for i, cluster_id in enumerate(np.unique(cluster_labels)):
                cluster_indices = np.where(cluster_labels == cluster_id)[0]
                new_prefix = prefix + chr(67 + i)  # C is ASCII 67, used as a prefix for recursive clustering
                
                # Recursively cluster each subset of data
                sub_labels = recursive_clustering(X[cluster_indices], depth - 1, max_leaf_size, new_prefix, random_state, spherical)
                
                # Map local cluster labels to global indices
                for local_idx, global_idx in enumerate(cluster_indices):
                    labels_dict[global_idx] = sub_labels[local_idx]
                    
            return labels_dict            
        
        # Start recursive clustering
        labels_dict = recursive_clustering(X, depth, max_leaf_size, prefix, random_state, spherical)
        
        # Rebuild the labels list from the label dictionary
        final_labels = [None] * len(labels_dict)
        for idx, label in labels_dict.items():
            final_labels[int(idx)] = label
        
        # Merge small clusters and update the labels
        merged_labels = merge_small_clusters(X, np.array(final_labels), min_cluster_size=10)
        
        # Calculate silhouette score
        print("Silhouette Score:", silhouette_score(X, merged_labels))
        
        # Return the model with final labels and centroids
        
        labels = encode_labels(merged_labels)
        
        # print(labels)
        
        # label_occurence = count_label_occurrences(labels)
        
        # print(label_occurence, len(label_occurence))
        
        return cls(
            clustering_model_type=CLUSTERING_MODEL.model_type, 
            labels=labels, 
            centroids=calculate_centroids(X, np.array(final_labels)),
            gpu_usage=gpu_usage
        )
        
    def predict(self, CLUSTERING_MODEL, test_input):
        clustering_model = CLUSTERING_MODEL.create_model({'n_clusters': len(self.centroids), 'init': self.centroids, 'n_init': 1}).fit(test_input)
        return clustering_model.predict(test_input)
                    
