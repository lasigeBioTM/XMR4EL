import gc

import numpy as np

from collections import Counter

from scipy.sparse import csr_matrix

from typing import Dict

from xmr4el.models.cluster_wrapper.clustering_model import ClusteringModel
from xmr4el.xmr.skeleton import Skeleton

class SkeletonConstruction:
    """
    Optimized hierarchical clustering constructor for building the XMR tree skeleton.
    
    Improvements made:
    1. Better memory management
    2. Type hints for better code clarity
    3. More efficient cluster processing
    4. Better validation and error handling
    5. Optimized array operations
    """
    
    def __init__(self, 
                 htree, 
                 Z: np.ndarray,
                 clustering_config: Dict,
                 min_leaf_size: int, 
                 depth: int,
                 dtype: np.dtype = np.float32):
        """
        Initialize with validated parameters.
        
        Args:
            htree: The hierarchical tree structure
            Z: Input embeddings matrix (n_samples, n_features)
            clustering_config: Configuration for clustering algorithm
            min_leaf_size: Minimum samples required per leaf cluster
            depth: Maximum depth of the tree
            dtype: Data type for computations
        """
        self.htree = htree
        self.Z = Z.astype(dtype)  # Ensure correct dtype
        self.indices = np.arange(Z.shape[0])  # More efficient than list(range())
        self.clustering_config = clustering_config
        self.min_leaf_size = max(1, min_leaf_size)  # Ensure at least 1
        self.depth = depth
        self.dtype = dtype
        
        # Validate inputs
        if Z.shape[0] == 0:
            raise ValueError("Input matrix Z cannot be empty")
        if depth < 0:
            raise ValueError("Depth must be non-negative")

    @staticmethod
    def _train_clustering(trn_corpus: np.ndarray, 
                         config: Dict, 
                         dtype: np.dtype = np.float32) -> 'ClusteringModel':
        """Train clustering model with memory efficiency."""
        return ClusteringModel.train(trn_corpus.astype(dtype), config, dtype)

    def execute(self) -> 'Skeleton':
        """Entry point for tree construction."""
        return self._gen_cluster(self.htree, self.indices, self.Z, self.depth, root=True)

    def _gen_cluster(self, 
                    htree: 'Skeleton', 
                    indices: np.ndarray, 
                    Z: np.ndarray, 
                    depth: int, 
                    root: bool = False) -> 'Skeleton':
        """
        Recursive cluster generation with optimizations:
        - Vectorized operations
        - Memory-efficient processing
        - Better cluster validation
        """
        if depth <= 0 or Z.shape[0] == 0:
            return htree

        gc.collect()

        n_samples = Z.shape[0]
        n_clusters = self.clustering_config["kwargs"]["n_clusters"]
        min_clusterable_size = max(n_clusters, self.min_leaf_size)

        # Base case: stop if too few samples
        if n_samples <= min_clusterable_size:
            return htree

        # Train clustering
        clustering_model = self._train_clustering(Z, self.clustering_config, self.dtype)
        cluster_labels = clustering_model.labels()
        cluster_counts = Counter(cluster_labels)
        
        del clustering_model
        
        print(cluster_counts)

        valid_clusters = []
        invalid_clusters = []
        
        for key, value in cluster_counts.items():
            if value < self.min_leaf_size:
                invalid_clusters.append(key)
            else:
                valid_clusters.append(key)

        # Handle root node special case
        if root and invalid_clusters:
            raise ValueError(
                f"All clusters are too small at root (min_leaf_size={self.min_leaf_size}). "
                f"Consider reducing n_clusters ({n_clusters}) or min_leaf_size."
            )
        elif invalid_clusters:
            return htree
            

        # Store node information
        htree.set_Z(Z)
        htree.set_kb_indices(indices)

        # Process valid clusters
        for cluster_id in valid_clusters:
            cluster_mask = (cluster_labels == cluster_id)
            cluster_indices = indices[cluster_mask]
            child_Z = Z[cluster_mask]
            
            child_htree = Skeleton(depth=htree.depth + 1)
            child_subtree = self._gen_cluster(
                child_htree,
                cluster_indices,
                child_Z,
                depth - 1
            )
            
            if not child_subtree.is_empty():
                htree.set_children(int(cluster_id), child_subtree)
            
        # Create cluster assignment matrix C
        n_clusters = len(np.unique(cluster_labels))
        C = np.zeros((Z.shape[0], n_clusters), dtype=self.dtype)
        C[np.arange(Z.shape[0]), cluster_labels] = 1

        htree.set_cluster_labels(cluster_labels)
        htree.set_C(csr_matrix(C))
        
        return htree