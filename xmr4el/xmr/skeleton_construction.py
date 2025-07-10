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
                 max_n_clusters, 
                 min_n_clusters,
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

    def execute(self, htree, comb_emb_idx, emb_idx, depth, clustering_config, root=False):
        """
        Recursively constructs the hierarchical clustering tree structure.
        
        The method:
        1. Validates clusterability of current node
        2. Determines optimal cluster count
        3. Trains clustering model
        4. Validates cluster quality
        5. Recursively processes valid clusters
        
        Args:
            htree (XMRTree): Current tree node being processed
            comb_emb_idx (dict): Indexed combined embeddings {index: embedding}
            emb_idx (dict): Indexed text embeddings {index: embedding}
            depth (int): Remaining recursion depth
            clustering_config (dict): Configuration for clustering algorithm
            
        Returns:
            XMRTree: Constructed hierarchical tree with children nodes
            
        Note:
            - Automatically handles memory cleanup via gc.collect()
            - Validates cluster sizes against min_leaf_size
            - Dynamically adjusts cluster count when validation fails
        """
        gc.collect()
        
        # Base case: stop recursion if depth exhausted
        if depth < 0:
            return htree
        
        # Convert indexed embeddings to array (sorted by index)
        indices = sorted(comb_emb_idx.keys())
        text_emb_array = np.array([comb_emb_idx[idx] for idx in indices])

        min_clusterable_size = max(self.min_n_clusters, self.min_leaf_size)

        if len(text_emb_array) <= min_clusterable_size:
            return htree
        
        # Changing Config of model 
        n_clusters = clustering_config["kwargs"]["n_clusters"]

        clustering_model = self._train_clustering(text_emb_array, clustering_config, self.dtype)  
            
        # FAISS-specific label extraction
        _, cluster_labels = clustering_model.model.model.index.search(text_emb_array, 1)
        cluster_labels = cluster_labels.flatten()
        
        cluster_counts = Counter(cluster_labels)
        
        for c in cluster_counts:
            if cluster_counts[c] < self.min_leaf_size:
                # Min leaf size not reached
                LOGGER.info("Removing Clustering, Minimal Leaf Size reached")
                if root:
                    LOGGER.info(cluster_counts)
                    raise Exception(f"Exiting Training, model has too many clusters being formed, mininal leaf size being reached")
                return htree
            
        LOGGER.info(f"Saving Clustering Model at depth {htree.depth}, with {n_clusters} clusters")
        htree.set_clustering_model(clustering_model)
        htree.set_text_embeddings(emb_idx)

        # Process each cluster recursively
        unique_labels = np.unique(cluster_labels)
        for cluster in unique_labels:
            cluster_indices = [idx for idx, label in zip(indices, cluster_labels) if label == cluster]
            
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