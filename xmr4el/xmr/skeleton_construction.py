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
                 htree, 
                 Z,
                 clustering_config,
                 min_leaf_size, 
                 depth,
                 dtype=np.float32):
        """
        Initializes the SkeletonConstruction with clustering parameters.
        
        Args:
            max_n_clusters (int): Maximum number of clusters to consider per node
            min_n_clusters (int): Minimum number of clusters to consider per node
            min_leaf_size (int): Minimum size for a cluster to be valid
            dtype (np.dtype): Data type for computations. Defaults to np.float32.
        """
        self.htree = htree
        self.Z = Z
        
        self.indices = [idx for idx in range(self.Z.shape[0])]
        
        self.clustering_config = clustering_config
        self.min_leaf_size = min_leaf_size
        self.depth = depth
        self.dtype = dtype
    
    @staticmethod
    def _train_clustering(trn_corpus, config, dtype=np.float32):
        """Trains the clustering model with the training data.

        Args:
            trn_corpus (np.array): Training data as a Dense Array
            config (dict): Configurations of the clustering model
            dtype (np.float): Type of the data inside the array

        Returns:
            ClusteringModel: Trained Clustering Model
        """
        return ClusteringModel.train(trn_corpus, config, dtype)    

    def execute(self):
        return self._gen_cluster(self.htree, self.indices, self.Z, self.depth, root=True)

    def _gen_cluster(self, htree, indices, Z, depth, root=False):
        """Executes the hierarchical clustering process.
        
        Args:
            htree: The hierarchical tree structure
            comb_emb_idx: Dictionary of indices to embeddings
            depth: Current depth of the tree
            clustering_config: Configuration for clustering
            root: Whether this is the root node (default: False)
            
        Returns:
            The constructed hierarchical tree
        """
        if depth < 0:
            return htree

        gc.collect()

        n_labels = Z.shape[0]
        n_clusters = self.clustering_config["kwargs"]["n_clusters"]

        min_clusterable_size = max(n_clusters, self.min_leaf_size)
        if n_labels <= min_clusterable_size:
            return htree

        # Train clustering
        clustering_model = self._train_clustering(Z, self.clustering_config, self.dtype)
        cluster_labels = clustering_model.labels()
        cluster_counts = Counter(cluster_labels)
        
        # print(cluster_labels, cluster_counts)

        # Separate valid and small clusters
        valid_clusters = []
        fallback_indices = []
        
        for cluster_id, count in cluster_counts.items():
            if count >= self.min_leaf_size:
                valid_clusters.append(cluster_id)
            else:
                fallback_indices.extend(
                    idx for idx, lbl in zip(self.indices, cluster_labels) if lbl == cluster_id
                )

        # Validate root clusters
        if not valid_clusters and fallback_indices and root:
            raise ValueError(
                "All clusters are too small at root. Try reducing n_clusters or min_leaf_size."
            )

        # htree.set_clustering_model(clustering_model)
        htree.set_Z(Z)
        htree.set_kb_indices(indices)

        # Process valid clusters
        for cluster in valid_clusters:
            cluster_indices = [idx for idx, lbl in zip(self.indices, cluster_labels) if lbl == cluster]
            child_htree = Skeleton(depth=htree.depth + 1)

            child_Z = Z[cluster_indices]

            child_subtree = self._gen_cluster(
                child_htree,
                cluster_indices, 
                child_Z,
                depth - 1,
            )
            
            if not child_subtree.is_empty():
                htree.set_children(int(cluster), child_subtree)

        # Process fallback cluster
        if fallback_indices:
            fallback_htree = Skeleton(depth=htree.depth + 1)
            
            child_Z = Z[fallback_indices]
            
            fallback_subtree = self._gen_cluster(
                fallback_htree,
                fallback_indices,
                child_Z, # Z
                depth - 1,
            )

            if not fallback_subtree.is_empty():
                fallback_cluster_id = max(valid_clusters) + 1 if valid_clusters else 0
                htree.set_children(int(fallback_cluster_id), fallback_subtree)
                
                for idx in fallback_indices:
                    cluster_labels[idx] = fallback_cluster_id
                
                
        C = np.zeros((Z.shape[0], n_clusters), dtype=int)
        for label_idx, cluster_id in enumerate(cluster_labels):
            C[label_idx, cluster_id] = 1

        htree.set_cluster_labels(cluster_labels)
        htree.set_C(C)

        return htree