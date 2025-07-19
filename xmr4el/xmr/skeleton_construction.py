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

        # 1) Depth cutoff
        if depth < 0:
            return htree

        indices = sorted(comb_emb_idx.keys())
        N = len(indices)
        htree.set_kb_indices(indices)

        # 2) Compute how many clusters we can actually form
        requested = clustering_config["kwargs"]["n_clusters"]
        max_clusters = N // self.min_leaf_size
        k = min(requested, max_clusters)

        # If we can’t form at least 2 clusters of size >= min_leaf_size, stop splitting
        if k < 2:
            return htree

        # 3) Prepare data & train clustering with adjusted k
        clustering_config["kwargs"]["n_clusters"] = k
        text_emb_array = np.array([comb_emb_idx[idx] for idx in indices])
        clustering_model = self._train_clustering(text_emb_array, clustering_config, self.dtype)
        cluster_labels = clustering_model.labels().flatten()
        cluster_counts = Counter(cluster_labels)

        # 4) If clustering didn’t actually split (say all points in one cluster), stop
        if len(cluster_counts) < 2:
            return htree

        htree.set_clustering_model(clustering_model)
        htree.set_text_embeddings(comb_emb_idx)

        # print(f"Requested={requested}, used={k}, counts={cluster_counts}")

        # 5) Separate valid vs fallback
        valid_clusters, fallback_indices = [], []
        for cid, count in cluster_counts.items():
            if count >= self.min_leaf_size:
                valid_clusters.append(cid)
            else:
                fallback_indices.extend(
                    idx for idx, lbl in zip(indices, cluster_labels) if lbl == cid
                )

        if root and not valid_clusters and fallback_indices:
            raise Exception("All clusters are too small at root.")

        # 6) Recurse on each valid cluster
        for cid in valid_clusters:
            cluster_indices = [idx for idx, lbl in zip(indices, cluster_labels) if lbl == cid]
            child_dict = {idx: comb_emb_idx[idx] for idx in cluster_indices}
            child = Skeleton(depth=htree.depth + 1)
            subtree = self.execute(child, child_dict, depth-1, clustering_config)
            if not subtree.is_empty():
                htree.set_children(int(cid), subtree)

        # 7) Handle fallback as one leaf (if any)
        if fallback_indices:
            fb_dict = {idx: comb_emb_idx[idx] for idx in fallback_indices}
            fb_child = Skeleton(depth=htree.depth + 1)
            fb_subtree = self.execute(fb_child, fb_dict, depth-1, clustering_config)
            if not fb_subtree.is_empty():
                fb_id = (max(valid_clusters) + 1) if valid_clusters else 0
                htree.set_children(fb_id, fb_subtree)

        return htree