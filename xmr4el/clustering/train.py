import gc
import numpy as np

from typing import Any, Dict, Optional, Tuple, Counter

from scipy.sparse import csr_matrix

from xmr4el.models.cluster_wrapper.clustering_model import ClusteringModel


class ClusteringTrainer:
    """Utility class providing the clustering training routine."""
    
    @staticmethod
    def train(
        Z: np.ndarray,
        config: Dict[str, Any],
        min_leaf_size: int = 20,
        max_leaf_size: Optional[int] = None,
        dtype: Any = np.float32,
    ) -> Tuple[Optional[csr_matrix], Optional[ClusteringModel]]:
        """Train clustering without recursive partitioning."""
        
        n_points = Z.shape[0]
        n_clusters = config["kwargs"]["n_clusters"]

        if n_points <= min_leaf_size:
            print(f"Too few points ({n_points}), stopping clustering.")
            return None, None

        if max_leaf_size is None:
            avg_cluster_size = n_points / n_clusters
            max_leaf_size = int(2 * avg_cluster_size)

        # Train clustering model
        clustering_model = ClusteringModel.train(Z, config, dtype)
        cluster_labels = clustering_model.labels()
        cluster_counts = Counter(cluster_labels)
        print(f"Cluster sizes: {cluster_counts}")

        # Filter out invalid clusters
        valid_clusters = [cid for cid, cnt in cluster_counts.items() if cnt >= min_leaf_size]
        if len(valid_clusters) <= 1:
            print(f"Only {len(valid_clusters)} valid clusters after pruning, stopping clustering.")
            return None, None

        # Build C_node
        rows, cols = [], []
        cluster_id_offset = 0
        for cluster_id in valid_clusters:
            indices = np.where(cluster_labels == cluster_id)[0]
            for i in indices:
                rows.append(i)
                cols.append(cluster_id_offset)
            cluster_id_offset += 1

        if len(rows) == 0:
            return None, None

        C_node = csr_matrix((np.ones(len(rows), dtype=dtype), (rows, cols)), shape=(n_points, cluster_id_offset))
        return C_node, clustering_model