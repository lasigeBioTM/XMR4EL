import os
import pickle
import numpy as np

from collections import Counter
from kneed import KneeLocator
from sklearn.preprocessing import normalize
from sklearn.metrics import silhouette_score, pairwise_distances_argmin_min, silhouette_samples

from src.node_logic.tree_node import TreeNode
     
     
class DivisiveHierarchicalClustering():
    
    def __init__(self, clustering_model_type=None, tree_node=None, gpu_usage=False):
        """
        Initialize the clustering model, labels, and centroids
        """
        self.clustering_model_type = clustering_model_type
        self.tree_node = tree_node
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
            'init': 'k-means++',
            'random_state': 0,
            'spherical': True,
        }
        
        config = {**DEFAULTS, **config}
        
        n_splits = config['n_splits']
        max_iter = config['max_iter']
        depth = config['depth']
        min_leaf_size = config['min_leaf_size']
        init = config['init']
        random_state = config['random_state']
        spherical = config['spherical']

        def recursive_clustering(X, tree_node, n_splits, max_iter, depth, min_leaf_size, random_state, clustering_model_factory, init):
            # Base case: Stop recursion if depth reaches 0
            if depth == 0 or X.shape[0] <= min_leaf_size:
                return tree_node

            # Determine the number of splits if not provided
            if n_splits == 0:
                n_splits = cls.__calculate_optimal_clusters(
                    X,
                    clustering_model_factory,
                    cluster_range=16,
                    random_state=random_state
                )
                
            # Fit the clustering model
            clustering_model = clustering_model_factory.create_model(
                {
                    'n_clusters': n_splits,
                    'max_iter': max_iter,
                    'random_state': random_state,
                    'init': init
                }
            ).fit(X)
                    
            cluster_labels = clustering_model.labels_
            
            # Check if clustering_model.labels_ is valid
            if cluster_labels is None:
                raise ValueError("Clustering model did not generate any labels.")
            
            # Insert the Node into the TreeNode
            tree_node.set_cluster_node(clustering_model, cluster_labels, X)
            
            # Calculate silhouette_scores
            overall_silhouette_score, silhouette_scores = cls.__compute_silhouette_scores(X, cluster_labels)
            
            # Insert values in the tree_node
            tree_node.cluster_node.set_silhouette_scores(overall_silhouette_score, silhouette_scores)
            
            # Recursively process each cluster
            for cluster_label in np.unique(cluster_labels):
                # Get points belonging to the current cluster
                indices = cluster_labels == cluster_label
                cluster_points = X[indices]
                n_cluster_points = cluster_points.shape[0]
                
                if n_cluster_points > min_leaf_size:
                    # Create a new TreeNode for the sub-cluster, Parent Node
                    child_tree_node = TreeNode(depth=tree_node.depth + 1)
                    
                    tree_node.add_child(cluster_label, child_tree_node)
                    
                    # Recur for the sub-cluster
                    recursive_clustering(
                        cluster_points,
                        child_tree_node,
                        n_splits=0,  # Let the method recalculate optimal clusters
                        max_iter=max_iter,
                        depth=depth - 1,
                        min_leaf_size=min_leaf_size,
                        random_state=random_state,
                        clustering_model_factory=clustering_model_factory,
                        init=init
                    )
            
            return tree_node
                    
        """Started the fit"""
        if spherical:
            X = normalize(X, norm='l2', axis=1)  
        
        root_tree_node = TreeNode()
        
        final_tree = recursive_clustering(
            X=X, 
            tree_node=root_tree_node,
            n_splits=n_splits,
            max_iter=max_iter,
            depth=depth,
            min_leaf_size=min_leaf_size,
            random_state=random_state,
            clustering_model_factory=clustering_model_factory,
            init=init
        )

        # print(final_tree.print_tree())
        
        return cls(
            clustering_model_type=clustering_model_factory.model_type, 
            tree_node=final_tree,
            gpu_usage=gpu_usage
        )
        
    @staticmethod
    def __compute_silhouette_scores(X, cluster_labels):
        """
        Compute silhouette scores for clusters at this node.
        """
        silhouette_scores = {}
        
        # Calculate overall silhouette score
        overall_silhouette_score = silhouette_score(X, cluster_labels)
        
        # Compute silhouette scores for each sample
        sample_silhouette_values = silhouette_samples(X, cluster_labels)
        
        # Compute mean silhouette score for each cluster
        for cluster_idx in np.unique(cluster_labels):
            cluster_silhouette_values = sample_silhouette_values[cluster_labels == cluster_idx]
            if len(cluster_silhouette_values) > 0:
                silhouette_scores[cluster_idx] = np.mean(cluster_silhouette_values)
            else:
                silhouette_scores[cluster_idx] = 0.0
                
        return overall_silhouette_score, silhouette_scores
        
    def predict(self, clustering_model_factory, test_input):
        clustering_model = clustering_model_factory.create_model(
            {
                'n_clusters': len(self.centroids), 
                'init': self.centroids, 
                'n_init': 1
            }
            ).fit(test_input)
        
        return clustering_model.predict(test_input)

    @staticmethod
    def __calculate_optimal_clusters(X, clustering_model_factory, cluster_range, random_state=0):
        k_range = range(2, cluster_range)
        wcss = [clustering_model_factory.create_model(
            {'n_clusters': k, 
            'random_state':random_state}
            ).fit(X).inertia_ for k in k_range]

        knee = KneeLocator(k_range, wcss, curve='convex', direction='decreasing', online=True)
        optimal_clusters = knee.knee
        # print(f"Optimal number of clusters: {optimal_clusters}")
        return optimal_clusters
    
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
        
        
                    
