import gc
import numpy as np

from typing import Counter

from scipy.sparse import csr_matrix

from xmr4el.models.cluster_wrapper.clustering_model import ClusteringModel


class ClusteringTrainer:

    @staticmethod
    def train(Z, config, min_leaf_size=20, max_leaf_size=None, depth=0, dtype=np.float32, partition=False):
        """
        Trains a (recursive) clustering tree unless partition=False.

        Args:
            Z: np.ndarray of shape (n_samples, n_features)
            config: dict with clustering config, including 'kwargs'->'n_clusters'
            min_leaf_size: int, min size for a cluster to be accepted
            max_leaf_size: int, optional max size for a leaf cluster
            depth: current recursion depth
            dtype: type used for csr_matrix
            partition: if False, disables recursive splitting
        Returns:
            Z: input data
            C: csr_matrix of shape (n_samples, n_clusters)
            clustering_model: fitted clustering model
        """
        gc.collect()

        if depth > 10:
            print(f"Limit Depth Reached")
            return None, None, None

        n_clusters = config["kwargs"]["n_clusters"]
        n_points = Z.shape[0]
        
        if n_points <= min_leaf_size:
            print(f"[Depth {depth}] Too few points ({n_points}), stopping recursion.")
            return None, None, None

        if max_leaf_size is None:
            avg_cluster_size = n_points / n_clusters
            max_leaf_size = int(2 * avg_cluster_size)

        # Train clustering
        clustering_model = ClusteringModel.train(Z, config, dtype)
        cluster_labels = clustering_model.labels()
        cluster_counts = Counter(cluster_labels)
        print(f"[Depth {depth}] Cluster sizes: {cluster_counts}")

        invalid_clusters = [cid for cid, cnt in cluster_counts.items() if cnt < min_leaf_size]
        if len(invalid_clusters) > 0:
            print(f"[Depth {depth}] {len(invalid_clusters)} invalid clusters, stopping clustering.")
            return None, None, None

        valid_clusters = [cid for cid, cnt in cluster_counts.items() if cnt >= min_leaf_size]
        if len(valid_clusters) <= 1:
            print(f"[Depth {depth}] Only {len(valid_clusters)} valid clusters after pruning, refusing singleton cluster.")
            return None, None, None

        cluster_id_offset = 0
        rows = []
        cols = []

        for cluster_id in valid_clusters:
            count = cluster_counts[cluster_id]
            indices = np.where(cluster_labels == cluster_id)[0]
            Z_sub = Z[indices]

            if partition and count > max_leaf_size:
                # Recursive partitioning
                sub_config = config.copy()
                sub_config["kwargs"] = sub_config["kwargs"].copy()
                sub_config["kwargs"]["n_clusters"] = 2
                print(f"[Depth {depth}] Reclustering large cluster {cluster_id} with {count} points into 2 clusters")

                _, C_sub, _ = ClusteringTrainer.train(Z_sub, sub_config, min_leaf_size, max_leaf_size, depth + 1, dtype, partition)

                if C_sub is None:
                    # Fallback to treating as a single cluster
                    for i in indices:
                        rows.append(i)
                        cols.append(cluster_id_offset)
                    cluster_id_offset += 1
                else:
                    C_sub = C_sub.tocsr()
                    for row_local, cluster_local in zip(*C_sub.nonzero()):
                        rows.append(indices[row_local])
                        cols.append(cluster_id_offset + cluster_local)
                    cluster_id_offset += C_sub.shape[1]
            else:
                # Cluster is small enough or partitioning is disabled
                for i in indices:
                    rows.append(i)
                    cols.append(cluster_id_offset)
                cluster_id_offset += 1

        if len(rows) == 0:
            print(f"[Depth {depth}] No valid clusters found, returning None")
            return None, None, None

        C = csr_matrix((np.ones(len(rows), dtype=dtype), (rows, cols)), shape=(n_points, cluster_id_offset))   
        return Z, C, clustering_model