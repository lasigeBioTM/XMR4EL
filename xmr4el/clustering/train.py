import gc
import numpy as np

from typing import Counter

from xmr4el.models.cluster_wrapper.clustering_model import ClusteringModel


class ClusteringTrainer():

    @staticmethod
    def train(Z, config, min_leaf_size=20, dtype=np.float32):
        """
        Recursive cluster generation with optimizations:
        - Vectorized operations
        - Memory-efficient processing
        - Better cluster validation
        """
        gc.collect()

        n_clusters = config["kwargs"]["n_clusters"]

        # Train clustering
        clustering_model = ClusteringModel.train(Z, config, dtype)
        cluster_labels = clustering_model.labels()
        cluster_counts = Counter(cluster_labels)

        valid_clusters = []
        invalid_clusters = []
        
        for key, value in cluster_counts.items():
            if value < min_leaf_size:
                invalid_clusters.append(key)
            else:
                valid_clusters.append(key)
                
        if len(invalid_clusters) > 0:
            print("Invalid clusters found")
            return None, None, None
        
        # Save Z and kb_indices
            
        # Create cluster assignment matrix C
        C = np.zeros((Z.shape[0], n_clusters), dtype=dtype)
        C[np.arange(Z.shape[0]), cluster_labels] = 1

        # Set cluster labels and C <- csr_matrix
        
        return Z, C, clustering_model