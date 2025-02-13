import os
import pickle
import numpy as np
from collections import Counter
from sklearn.preprocessing import normalize
from sklearn.metrics import silhouette_score, pairwise_distances_argmin_min

     
class DivisiveHierarchicalClustering():
    
    def __init__(self, clustering_model_type=None, labels=None, centroids=None, gpu_usage=False):
        """
        Initialize the clustering model, labels, and centroids
        """
        self.clustering_model_type = clustering_model_type
        self.labels = labels
        self.centroids = centroids
        self.gpu_usage = gpu_usage
        
    def save(self, hierarchical_folder):
        """Save the model and its parameters to a file """
        os.makedirs(os.path.dirname(hierarchical_folder), exist_ok=True)
        with open(hierarchical_folder, 'wb') as fout:
            pickle.dump(self.__dict__, fout)

    @classmethod
    def load(cls, hierarchical_folder):
        """Load a saved model from a file."""
        if not os.path.exists(hierarchical_folder):
            raise FileNotFoundError(f"Hierarchical folder {hierarchical_folder} does not exist.")
        
        with open(hierarchical_folder, 'rb') as fin:
            model_data = pickle.load(fin)
        model = cls()
        model.__dict__.update(model_data)
        return model
    
    @classmethod
    def fit(cls, X, clustering_model_factory, depth=3, max_leaf_size=100, prefix="", random_state=0, spherical=True, gpu_usage=False):
        """
        Main method to perform divisive hierarchical clustering
        """
    
        def recursive_clustering(X, depth=2, max_leaf_size=100, prefix="", random_state=0, spherical=True):
            """
            Recursively perform the clustering process
            """
            
            if X.shape[0] < max_leaf_size or depth == 0:
                return {i: prefix for i in range(X.shape[0])}
            
            n_splits = cls.__compute_branching_factor(X.shape[0])
            n_iter = 100 + (10 * n_splits) 
            
            if spherical:
                X = normalize(X, norm="l2", axis=1)
            
            # Fiting Clustering Model
            kmeans_model = clustering_model_factory.create_model({'n_clusters': n_splits, 'max_iter': n_iter, 'random_state': random_state}).fit(X)
                
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
        merged_labels = cls.__merge_small_clusters(X, np.array(final_labels), min_cluster_size=10)
        
        # Calculate silhouette score
        print("Silhouette Score:", silhouette_score(X, merged_labels))
        
        labels = cls.__encode_labels(merged_labels)
        
        return cls(
            clustering_model_type=clustering_model_factory.model_type, 
            labels=labels, 
            centroids=cls.__calculate_centroids(X, np.array(final_labels)),
            gpu_usage=gpu_usage
        )
        
    def predict(self, clustering_model_factory, test_input):
        
        # print(self.__count_label_occurrences(self.labels))
        
        clustering_model = clustering_model_factory.create_model({'n_clusters': len(self.centroids), 'init': self.centroids, 'n_init': 1}).fit(test_input)
        return clustering_model.predict(test_input)
    
    @staticmethod
    def __merge_small_clusters(X, labels, min_cluster_size=10):
        """Merges small clusters with larger clusters."""
        unique_labels = np.unique(labels)
        valid_labels = [label for label in unique_labels if np.sum(labels == label) >= min_cluster_size]
        centroids = np.array([X[labels == label].mean(axis=0) for label in unique_labels])
            
        updated_labels = labels.copy()
        for label in unique_labels:                
            cluster_indices = np.where(labels == label)[0]
            if len(cluster_indices) <= min_cluster_size:
                cluster_points = X[cluster_indices]
                _, closest_centroid_idx = pairwise_distances_argmin_min(cluster_points, centroids)
                
                updated_labels[labels == label] = valid_labels[int(closest_centroid_idx[0])]
                
        return updated_labels
    
    @staticmethod
    def __compute_branching_factor(num_samples, min_clusters=2, max_clusters=16):
        """
        Compute branching factor based on the number of samples
        """
        return min(max(num_samples // 100, min_clusters), max_clusters)

    @staticmethod
    def __encode_labels(labels_list):
        """Convert string labels into numeric labels"""
        label_to_idx = {}
        return np.array([label_to_idx.setdefault(label, len(label_to_idx)) for label in labels_list])
    
    @staticmethod    
    def __count_label_occurrences(labels_list):
        """
        Count occurrences of each label in the list
        """
        return list(Counter(labels_list).items())
        
    @staticmethod   
    def __calculate_centroids(X, labels):
        """
        Calculate the centroids of the clusters based on the labels
        """
        return np.array([X[labels == label].mean(axis=0) for label in np.unique(labels)])  
        
        
                    
