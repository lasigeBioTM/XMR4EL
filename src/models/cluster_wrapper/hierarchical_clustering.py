import logging
import os
import pickle
import numpy as np

from collections import Counter
from kneed import KneeLocator
from sklearn.preprocessing import normalize
from sklearn.metrics import silhouette_score, silhouette_samples

from src.models.hierarchical_node import TreeNode
from src.models.cluster_wrapper.clustering_model import ClusteringModel
     
     
LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

     
class HierarchicalClustering(ClusteringModel):
    
    def __init__(self, config, clustering_model_type=None, hierarchical_node=None):
        """
        Initialize the clustering model, labels, and centroids
        """
        self.config = config
        self.clustering_model_type = clustering_model_type
        self.hierarchical_node = hierarchical_node
        
    def save(self, save_dir):
        """Save trained Hierarchical Clustering model to disk.

        Args:
            save_dir (str): Folder to store serialized object in.
        """
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, "clustering.pkl"), "wb") as fout:
            pickle.dump(self.model, fout)

    @classmethod
    def load(cls, load_dir):
        """Load a saved Hierarchical Clustering model from disk.

        Args:
            load_dir (str): Folder inside which the model is loaded.

        Returns:
            SklearnAgglmerativeClustering: The loaded object.
        """
        
        LOGGER.info(f"Loading Agglomerative Clustering model from {load_dir}")
        clustering_path = os.path.join(load_dir, "clustering.pkl")
        assert os.path.exists(clustering_path), f"clustering path {clustering_path} does not exist"
        
        with open(clustering_path, 'rb') as fin:
            model_data = pickle.load(fin)
        model = cls()
        model.__dict__.update(model_data)
        return model
    
    @classmethod
    def fit(cls, X, cluster_type, config={}, spherical=True, gpu_usage=False):
        """
        Main method to perform divisive hierarchical clustering
        """
        
        DEFAULTS = {
            'n_splits': 2,
            'max_iter': 100,
            'depth': 1,
            'min_leaf_size':10,
            'min_clusters': 3,
            'random_state': 0,
        }
        
        config = {**DEFAULTS, **config}
        
        n_splits = config['n_splits']
        max_iter = config['max_iter']
        depth = config['depth']
        min_leaf_size = config['min_leaf_size']
        min_clusters = config['min_clusters']
        init = config['init']
        random_state = config['random_state']

        def recursive_clustering(X, tree_node, n_splits, max_iter, 
                                 depth, min_leaf_size, min_clusters, random_state, clustering_model_factory, init):
            """
            Recursively applies divisive clustering while handling cases where only one cluster remains.
            """
            # **Fix: Stop if all data points belong to one cluster**
            if X.shape[0] <= min_leaf_size:
                # print("Stopping: Only one cluster exists or too few points to split further.")
                tree_node.set_cluster_node(None, np.zeros(X.shape[0]), X)  # Assign all points to cluster 0
                return tree_node
            
            # Determine the number of splits if not provided
            if n_splits == 0:
                n_splits = cls.__calculate_optimal_clusters(
                    X, clustering_model_factory, cluster_range=16, random_state=random_state
                )
                print(f"Calculate Optimal Clusters: {n_splits}")
                if n_splits == None:
                    return 16

            merge_loop = True
            while merge_loop:
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

                # **Fix: Stop merging if only one cluster remains**
                if len(np.unique(cluster_labels)) <= min_clusters:
                    # print("Stopping: Minimun number of cluster detected after merging.")
                    return None

                # Merge small clusters
                merged_labels = cls.__merge_small_clusters(X, cluster_labels, min_leaf_size=min_leaf_size, min_clusters=min_clusters)

                new_n_splits = len(np.unique(merged_labels))
                if new_n_splits == n_splits:
                    merge_loop = False
                else:
                    n_splits = new_n_splits  
                    # print(f"Number of splits after merge: {n_splits}")

            # print(cls.__count_label_occurrences(cluster_labels))

            # Insert the Node into the Tree
            tree_node.set_cluster_node(clustering_model, cluster_labels, X)

            # Calculate silhouette_scores
            overall_silhouette_score, silhouette_scores = cls.__compute_silhouette_scores(X, cluster_labels)
            
            # Store silhouette scores in tree_node
            tree_node.cluster_node.set_silhouette_scores(overall_silhouette_score, silhouette_scores)

            # Recursively process each cluster
            for cluster_label in np.unique(cluster_labels):
                indices = cluster_labels == cluster_label
                cluster_points = X[indices]

                # **Fix: Stop clustering at leaf nodes**
                if cluster_points.shape[0] <= min_leaf_size:
                    # print(f"Stopping: Cluster {cluster_label} is a leaf node.")
                    continue  # Don't apply further clustering
                
                if depth - 1 == 0:
                    return tree_node
                
                # Create a new child node and recurse
                child_tree_node = TreeNode(depth=tree_node.depth + 1)
                
                new_child = recursive_clustering(
                    cluster_points, child_tree_node,
                    n_splits=0, max_iter=max_iter, depth=depth - 1,
                    min_leaf_size=min_leaf_size, min_clusters=min_clusters, random_state=random_state,
                    clustering_model_factory=clustering_model_factory, init=init
                )
                
                if new_child:  # Only add if valid
                    tree_node.add_child(cluster_label, new_child)

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
            depth=depth + 1,
            min_leaf_size=min_leaf_size,
            min_clusters=min_clusters,
            random_state=random_state,
            clustering_model_factory=clustering_model_factory,
            init=init
        )
        
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
    
    # Look to this method
    def predict(self, batch_embeddings):
        """ Predicts the cluster for input data X by traversing the hierarchical tree. """
        predictions = []
        
        for input_embedding in batch_embeddings:
            
            # print(f"Input Embedding: {input_embedding[0]}")
            
            current_node = self.tree_node  # Start at the root
            pred = []
            
            # print("Entered While Loop")
            while not current_node.is_leaf():
                if current_node.cluster_node is None or current_node.cluster_node.model is None:
                    # print("Cluster Model Is Null")
                    break
                
                # print(current_node)
                
                input_embedding = input_embedding.reshape(1, -1)
                cluster_predictions = current_node.cluster_node.model.predict(input_embedding)[0]
                
                # print(f"Cluster Predictions: {cluster_predictions}")
                
                # print(f"Predicted Cluster: {cluster_predictions}")
                # print(f"Children Keys: {list(current_node.children.keys())}")
                
                pred.append(int(cluster_predictions))
                
                if cluster_predictions not in current_node.children:
                    # print("Prediction not in children, stopping here.")
                    break
                
                current_node = current_node.children[int(cluster_predictions)]
            
            # print(f"Exiting Loop with predictions: {pred}")
            # print(f"Cluster Predictions: {pred}")
            predictions.append(pred)
            
            # exit()
            
        return predictions
    

    @staticmethod
    def __calculate_optimal_clusters(X, clustering_model_factory, cluster_range, random_state=0):
        max_clusters = min(len(X), cluster_range)
        if max_clusters < cluster_range:
            cluster_range = max_clusters
        
        print(f"Cluster Range : {cluster_range}")
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
    def __merge_small_clusters(X, labels, min_leaf_size, min_clusters):
        unique, counts = np.unique(labels, return_counts=True)
        cluster_sizes = dict(zip(unique, counts))

        # Identify small clusters
        small_clusters = {c for c, size in cluster_sizes.items() if size < min_leaf_size}
        
        if len(small_clusters) == 0:
            return labels  #  No small clusters, return original labels
        
        # print(f"Detected {len(small_clusters)} small clusters (<{min_leaf_size})")

        updated_labels = labels.copy()

        # Compute centroids of non-small clusters
        large_clusters = [c for c in np.unique(updated_labels) if c not in small_clusters]
        
        if len(large_clusters) == 0:
            # print("Warning: No large clusters to merge into. Returning labels unchanged.")
            return np.zeros_like(labels)  # Assign all points to cluster 0  #  Avoid breaking if no valid clusters remain
        
        # Fix: If only one cluster exists, return it unchanged
        if len(unique) <= min_clusters:
            # print("Warning: Cluster does not have the minimum of clusters. Returning labels unchanged.")
            return labels
        
        centroids = np.array([X[updated_labels == c].mean(axis=0) for c in large_clusters])

        for cluster in small_clusters:
            indices = np.where(updated_labels == cluster)[0]  # Points in the small cluster
            
            if len(indices) == 0:
                continue  # Skip empty clusters
            
            # Compute distances from each point to large cluster centroids
            distances = np.linalg.norm(X[indices, None, :] - centroids, axis=2)  

            # Assign each point to the nearest large cluster
            nearest_cluster = np.argmin(distances, axis=1)
            updated_labels[indices] = np.array(large_clusters)[nearest_cluster]  # Correct reassignment

        return updated_labels  # Always return modified labels


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
        # return [n_values for _, n_values in sorted_data]
        
        
                    
