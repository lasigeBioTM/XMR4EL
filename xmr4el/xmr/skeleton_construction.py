import gc
import logging

import numpy as np

from typing import Counter

from xmr4el.xmr.skeleton import Skeleton
from xmr4el.xmr.tuner import XMRTuner 
from xmr4el.models.cluster_wrapper.clustering_model import ClusteringModel


LOGGER = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class SkeletonConstruction():
    
    def __init__(self, 
                 max_n_clusters, 
                 min_n_clusters,
                 min_leaf_size, 
                 dtype=np.float32):
        
        """
        Init:
            max_n_clusters (int): Maximum number of clusters to consider
            min_n_clusters (int): Minimum number of clusters to consider
            min_leaf_size (int): Minimum size for a cluster to be valid
            dtype: Data type for computations
        """
        # Configs
        self.max_n_clusters = max_n_clusters
        self.min_n_clusters = min_n_clusters
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

    def execute(self, htree, comb_emb_idx, emb_idx, depth, clustering_config):
        """
        Recursively builds hierarchical clustering tree structure using text embeddings
        
        Args:
            htree (XMRTree): Current tree node
            comb_emb_idx (dict): Indexed combined embeddings (text + PIFA)
            emb_idx (dict): Indexed text embeddings
            depth (int): Remaining recursion depth
            clustering_config (dict): Configuration for clustering
            
        Returns:
            XMRTree: Constructed hierarchical tree
        """
        gc.collect()
        
        # Base case: stop recursion if depth exhausted
        if depth < 0:
            return htree
        
        # Convert indexed embeddings to array (sorted by index)
        indices = sorted(comb_emb_idx.keys())
        text_emb_array = np.array([comb_emb_idx[idx] for idx in indices])

        # Base case: stop if too few points to cluster meaningfully
        if len(text_emb_array) <= self.min_n_clusters:
            return htree

        # Determine optimal number of clusters  
        k_range = (self.min_n_clusters, self.max_n_clusters)
        optimal_k, _ = XMRTuner.tune_k(text_emb_array, clustering_config, self.dtype, k_range=k_range)
        
        n_clusters = optimal_k
        clustering_config["kwargs"]["n_clusters"] = n_clusters

        while True:
            clustering_model = self._train_clustering(
                text_emb_array, clustering_config, self.dtype
            )  
            cluster_labels = clustering_model.model.labels()
            cluster_counts = Counter(cluster_labels)

            # Check if any cluster has fewer than 2 samples (to avoid classifier errors)
            if min(cluster_counts.values()) < 2:  # <-- NEW CHECK (prevents singleton clusters)
                if n_clusters == self.min_n_clusters:
                    if htree.depth == 0:
                        LOGGER.warning("Cannot split further: Some clusters have < 2 samples.")
                        break
                    return htree
                
                # Try with fewer clusters
                n_clusters -= 1
                clustering_config["kwargs"]["n_clusters"] = n_clusters
                continue

            break  # Valid clustering found
        
        LOGGER.info(f"Saving Clustering Model at depth {htree.depth}, with {n_clusters} clusters")
        
        htree.set_clustering_model(clustering_model)
        htree.set_text_embeddings(emb_idx)

        # Process each cluster recursively
        unique_labels = np.unique(cluster_labels)
        for cluster in unique_labels:
            cluster_indices = [idx for idx, label in zip(indices, cluster_labels) if label == cluster]
            
            # Skip if cluster has fewer than 2 samples (cannot train classifier)
            if len(cluster_indices) < 2:
                LOGGER.warning(f"Skipping cluster {cluster}: Only 1 sample.")
                continue  # <-- Skip instead of recursing
            
            filt_combined_dict = {idx: comb_emb_idx[idx] for idx in cluster_indices}
            filt_text_dict = {idx: emb_idx[idx] for idx in cluster_indices}
            
            new_child_htree_instance = Skeleton(depth=htree.depth + 1)
            new_child_htree = self.execute(
                new_child_htree_instance,
                filt_combined_dict,
                filt_text_dict,
                depth - 1,
                clustering_config,
            )

            if not new_child_htree.is_empty():
                htree.set_children(int(cluster), new_child_htree)

        return htree