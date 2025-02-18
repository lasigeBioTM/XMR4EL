import os
import pickle
import numpy as np
from collections import Counter
from kneed import KneeLocator

from sklearn.preprocessing import normalize
from sklearn.metrics import silhouette_score, pairwise_distances_argmin_min, davies_bouldin_score, silhouette_samples

     
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
    def fit(cls, X, clustering_model_factory, config={}, gpu_usage=False):
        """
        Main method to perform divisive hierarchical clustering
        """
        
        DEFAULTS = {
            'n_splits': 2,
            'max_iter': 100,
            'depth': 1,
            'min_leaf_size':10,
            'max_leaf_size': 20,
            'variance': 0.1,
            'init': 'k-means++',
            'random_state': 0,
            'spherical': True,
            'prefix': ''
        }
        
        config = {**DEFAULTS, **config}
        
        n_splits = config['n_splits']
        max_iter = config['max_iter']
        depth = config['depth']
        min_leaf_size = config['min_leaf_size']
        max_leaf_size = config['max_leaf_size']
        variance = config['variance']
        init = config['init']
        random_state = config['random_state']
        spherical = config['spherical']
        prefix = config['prefix']

        def recursive_clustering(X, n_splits, max_iter, 
                                 depth, min_leaf_size, max_leaf_size, 
                                 variance, init, random_state, 
                                 spherical, prefix):
            """
            Recursively perform the clustering process
            
            Args:
                X (numpy.array): Embeddings
                n_splits (int): The number of clusters to form as well as the number of centroids to generate.
                max_iter (int): Maximum number of iterations of the k-means algorithm for a single run.
                depth (int) : Number of times the clustering should run
                min_leaf_size (int): Number of minimum datapoints an cluster can have 
                max_leaf_size (int): Number of maximum datapoints an cluster can have 
                random_state (int): Random State 
                spherical (bool): To normalize the data before the model runs
                prefix (char): Prefix of the label
            """
            
            if X.shape[0] < max_leaf_size or depth == 0:
                return {i: prefix for i in range(X.shape[0])}   
            
            if n_splits == 0:            
                n_splits = cls.__calculate_optimal_clusters(X, clustering_model_factory, 16, random_state)
            
            # Fiting Clustering Model
            kmeans_model = clustering_model_factory.create_model(
                {
                    'n_clusters': n_splits, 
                    'max_iter': max_iter, 
                    'random_state': random_state, 
                    'init':init
                 }
                ).fit(X)
                
            cluster_labels = kmeans_model.labels_
            
            # Get the clusters to refine
            
            """Metrics"""
            # print("Intercalar Cluster")
            # cls.__metrics(X, cluster_labels, n_splits)
            # print(cls.__count_label_occurrences(cluster_labels))
            
            labels_dict = {}
            for i, cluster_id in enumerate(np.unique(cluster_labels)):
                cluster_indices = np.where(cluster_labels == cluster_id)[0]
                # C is ASCII 67, used as a prefix for recursive clustering, A must be 65
                new_prefix = prefix + chr(65 + i)  
                
                # Recursively cluster each subset of data
                sub_labels = recursive_clustering(
                    X[cluster_indices], 
                    0, 
                    max_iter, 
                    depth - 1, 
                    min_leaf_size, 
                    max_leaf_size, 
                    variance, 
                    init, 
                    random_state, 
                    spherical, 
                    new_prefix
                )
                
                # Map local cluster labels to global indices
                for local_idx, global_idx in enumerate(cluster_indices):
                    labels_dict[global_idx] = sub_labels[local_idx]
                    
            return labels_dict      
        
        """Started the fit"""
        if spherical:
            X = normalize(X, norm='l2', axis=1)  
        
        # Start recursive clustering
        labels_dict = recursive_clustering(
            X, 
            n_splits, 
            max_iter, 
            depth, 
            min_leaf_size, 
            max_leaf_size, 
            variance, init, 
            random_state, 
            spherical, 
            prefix
        )
        
        # Rebuild the labels list from the label dictionary
        final_labels = [None] * len(labels_dict)
        for idx, label in labels_dict.items():
            final_labels[int(idx)] = label
        
        final_labels = cls.__encode_labels(final_labels)
        
        # print(cls.__count_label_occurrences(final_labels))    
    
        # Merge small clusters and update the labels
        merged_labels = cls.__merge_small_clusters(X, np.array(final_labels), min_cluster_size=min_leaf_size)
        # merged_labels = final_labels    
        
        # print(cls.__count_label_occurrences(merged_labels))
        
        n_splits = np.unique(merged_labels).shape[0]
        
        """Metrics"""
        cls.__metrics(X, merged_labels, n_splits)
        
        print(cls.__count_label_occurrences(merged_labels))
        
        return cls(
            clustering_model_type=clustering_model_factory.model_type, 
            labels=merged_labels, 
            centroids=cls.__calculate_centroids(X, np.array(final_labels)),
            gpu_usage=gpu_usage
        )
        
    def predict(self, clustering_model_factory, test_input):
        
        # print(self.__count_label_occurrences(self.labels))
        
        clustering_model = clustering_model_factory.create_model(
            {
                'n_clusters': len(self.centroids), 
                'init': self.centroids, 
                'n_init': 1
            }
            ).fit(test_input)
        
        return clustering_model.predict(test_input)
    
    @classmethod
    def __metrics(cls, X, cluster_labels, n_splits):
        
        """
        for idx in range(n_splits):
            intra_cluster_variance = cls.__intra_cluster_variance(X[cluster_labels == idx])
            print(f"Intra-Cluster Variance -> Cluster {idx}: {intra_cluster_variance}")
        
        cluster_variance = cls.__calculate_cluster_variance(X, cluster_labels, n_splits)
        print("\nCluster Variance:")
        for idx in range(n_splits):
            print(f"Cluster {idx} -> {cluster_variance[idx]}")
        
            
        max_distances = cls.__max_distance_from_centroid(X, cluster_labels, n_splits)
        print(f"\nMax Distance:")
        for idx in range(n_splits):
            print(f"Cluster {idx} -> {max_distances[idx]}")
        
        davies_bouldin_score = cls.__davies_bouldin(X, cluster_labels)            
        print(f"\nDavies Bouldin Score: {davies_bouldin_score}")
        """
        mean_silhouette_scores = cls.__mean_silhouette_for_splits(X, cluster_labels)
        print("\nMean Silhouette Score:")
        for idx in range(n_splits):
            print(f"Cluster {idx} -> {mean_silhouette_scores[idx]}")
        
        print(f"\nSilhouette Score: {silhouette_score(X, cluster_labels)}")

    @staticmethod
    def __calculate_optimal_clusters(X, clustering_model_factory, cluster_range, random_state=0):
        k_range = range(2, cluster_range)
        wcss = [clustering_model_factory.create_model(
            {'n_clusters': k, 
            'random_state':random_state}
            ).fit(X).inertia_ for k in k_range]

        knee = KneeLocator(k_range, wcss, curve='convex', direction='decreasing')
        optimal_clusters = knee.knee
        # print(f"Optimal number of clusters: {optimal_clusters}")
        return optimal_clusters
    
    @staticmethod
    def __mean_silhouette_for_splits(X, cluster_labels):    
        num_clusters = np.unique(cluster_labels).shape[0]
        sample_silhouette_values = silhouette_samples(X, cluster_labels)
        
        means_lst = []
        for label in range(num_clusters):
            means_lst.append(sample_silhouette_values[cluster_labels == label].mean())
            
        return means_lst
        
    
    @staticmethod
    def __intra_cluster_variance(cluster_embeddings):
        """
            Calculates the intra-cluster variance,
            Averaged variances
            It provides a measure of how spread out the 
            points are in the feature space for a given cluster. 
            
            Lower variance means tighter points.
            
            Quick measure of overall variance across all dimensions in the cluster.
        """
        variance = np.var(cluster_embeddings, axis=0).mean()
        return variance
        
    @staticmethod
    def __max_distance_from_centroid(X, labels, n_clusters, percentile=95):
        """
        Calculate the 95th percentile distance from the centroid for each cluster.
        
        This method calculates the maximum distance from the centroid 
        for each cluster, specifically the 95th percentile of all
        point-to-centroid distances in that cluster.
        """
        max_distances = []
        for i in range(n_clusters):
            cluster_points = X[labels == i]  # Get points in the cluster
            centroid = cluster_points.mean(axis=0)  # Compute cluster centroid
            distances = np.linalg.norm(cluster_points - centroid, axis=1)  # Euclidean distances
            max_distance = np.percentile(distances, percentile)  # 95th percentile
            max_distances.append(max_distance)
        return max_distances

    
    @staticmethod
    def __calculate_cluster_variance(X, labels, n_clusters):
        """
            Understand how much the distances from the centroid vary within the cluster
        """
        cluster_variance = []
        for i in range(n_clusters):
            cluster_points = X[labels == i]
            centroid = cluster_points.mean(axis=0)
            variance = np.var(np.linalg.norm(cluster_points - centroid, axis=1))
            cluster_variance.append(variance)
        return cluster_variance
    
    @staticmethod
    def __davies_bouldin(X, labels):
        db_score = davies_bouldin_score(X, labels)
        return db_score
    
    
    @staticmethod
    def __merge_small_clusters(X, labels, min_cluster_size=10):
        """
        Merges small clusters with larger clusters based on proximity to valid clusters.
        Ensures that all small clusters are merged.
        """
        unique_labels = np.unique(labels)
        
        # Find valid clusters and their centroids
        valid_labels = [label for label in unique_labels if np.sum(labels == label) >= min_cluster_size]
        valid_centroids = np.array([X[labels == label].mean(axis=0) for label in valid_labels])
        
        updated_labels = labels.copy()
        
        # Process small clusters
        for label in unique_labels:
            cluster_indices = np.where(labels == label)[0]
            if len(cluster_indices) < min_cluster_size:
                cluster_points = X[cluster_indices]
                
                # Find the closest valid centroid
                closest_valid_idx, _ = pairwise_distances_argmin_min(cluster_points, valid_centroids)
                
                # Assign all points in the small cluster to the closest valid cluster
                closest_label = valid_labels[int(closest_valid_idx[0])]
                updated_labels[cluster_indices] = closest_label
        
        return updated_labels


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
        labels_list = list(Counter(labels_list).items())
        sorted_data = sorted(labels_list, key=lambda x: x[1], reverse=True)
        out = ""
        for label, n_values in sorted_data:
            out += f"Cluster: {label} -> {n_values}\n"
        return out
        
    @staticmethod   
    def __calculate_centroids(X, labels):
        """
        Calculate the centroids of the clusters based on the labels
        """
        return np.array([X[labels == label].mean(axis=0) for label in np.unique(labels)])  
        
        
                    
