import gc

import numpy as np

from typing import Counter

from xmr4el.xmr.skeleton import Skeleton
from xmr4el.models.cluster_wrapper.clustering_model import ClusteringModel


class SkeletonConstruction():
    """
    Hierarchical clustering constructor for building the XMR tree skeleton.
    
    This class handles the recursive construction of the hierarchical tree structure
    using clustering algorithms. Key features:
    - Dynamic determination of optimal cluster count
    - Recursive tree building with depth control
    - Validation of cluster quality and size constraints
    - Memory-efficient processing of embeddings
    
    Attributes:
        max_n_clusters (int): Maximum number of clusters per node
        min_n_clusters (int): Minimum number of clusters per node
        min_leaf_size (int): Minimum samples required per leaf cluster
        dtype (np.dtype): Data type for numerical operations
    """
    
    def __init__(self, 
                 min_leaf_size, 
                 dtype=np.float32):
        """
        Initializes the SkeletonConstruction with clustering parameters.
        
        Args:
            max_n_clusters (int): Maximum number of clusters to consider per node
            min_n_clusters (int): Minimum number of clusters to consider per node
            min_leaf_size (int): Minimum size for a cluster to be valid
            dtype (np.dtype): Data type for computations. Defaults to np.float32.
        """
        # Configs
        self.min_leaf_size = min_leaf_size
        
        # Type
        self.dtype = dtype
    
    @staticmethod
    def _train_clustering(trn_corpus, config, dtype=np.float32):
        """Trains the clustering model with the training data

        Args:
            trn_corpus (np.array): Trainign data as a Dense Array
            config (dict): Configurations of the clustering model
            dtype (np.float): Type of the data inside the array

        Return:
            ClusteringModel (ClusteringModel): Trained Clustering Model
        """
        # Delegate training to ClusteringModel class
        return ClusteringModel.train(trn_corpus, config, dtype)    

    def execute(self, htree, comb_emb_idx, depth, clustering_config, root=False):
        gc.collect()

        if depth < 0:
            return htree

        n_clusters = clustering_config["kwargs"]["n_clusters"]
        indices = sorted(comb_emb_idx.keys())
        text_emb_array = np.array([comb_emb_idx[idx] for idx in indices])

        htree.set_kb_indices(indices)

        min_clusterable_size = max(n_clusters, self.min_leaf_size)
        if len(text_emb_array) <= min_clusterable_size:
            return htree

        # Train clustering
        clustering_model = self._train_clustering(text_emb_array, clustering_config, self.dtype)
        cluster_labels = clustering_model.labels().flatten()
        cluster_counts = Counter(cluster_labels)

        # Separate valid and small clusters
        valid_clusters = []
        fallback_indices = []
        for cluster_id in cluster_counts:
            if cluster_counts[cluster_id] >= self.min_leaf_size:
                valid_clusters.append(cluster_id)
            else:
                fallback_indices.extend([
                    idx for idx, lbl in zip(indices, cluster_labels) if lbl == cluster_id
                ])

        # If all clusters are invalid and this is the root, raise
        if not valid_clusters and fallback_indices and root:
            raise Exception("All clusters are too small at root. Try reducing n_clusters or min_leaf_size.")

        htree.set_clustering_model(clustering_model)
        htree.set_text_embeddings(comb_emb_idx)

        print(Counter(cluster_labels), valid_clusters)

        # Process valid clusters
        for cluster in valid_clusters:
            cluster_indices = [idx for idx, lbl in zip(indices, cluster_labels) if lbl == cluster]
            child_comb_dict = {idx: comb_emb_idx[idx] for idx in cluster_indices}
            child_htree = Skeleton(depth=htree.depth + 1)

            child_subtree = self.execute(
                child_htree,
                child_comb_dict,
                depth - 1,
                clustering_config,
            )

            if not child_subtree.is_empty():
                htree.set_children(int(cluster), child_subtree)

        # Fallback leaf
        if fallback_indices:
            fallback_dict = {idx: comb_emb_idx[idx] for idx in fallback_indices}
            fallback_htree = Skeleton(depth=htree.depth + 1)
            fallback_subtree = self.execute(
                fallback_htree,
                fallback_dict,
                depth - 1,
                clustering_config,
            )

            if not fallback_subtree.is_empty():
                fallback_cluster_id = max(valid_clusters) + 1 if valid_clusters else 0
                htree.set_children(fallback_cluster_id, fallback_subtree)  # Use -1 for fallback child, trying 999999

        return htree