import gc
import numpy as np

from typing import Counter

from scipy.sparse import csr_matrix

from xmr4el.models.cluster_wrapper.clustering_model import ClusteringModel


class ClusteringTrainer():

    @staticmethod
    def train(Z, config, min_leaf_size=20, depth=0, max_depth=10, dtype=np.float32):
        """
        Recursive cluster generation with optimizations:
        - Vectorized operations
        - Memory-efficient processing
        - Better cluster validation
        """
        gc.collect()

        n_clusters = config["kwargs"]["n_clusters"]
        n_points = Z.shape[0]
        
        if n_points <= min_leaf_size or depth >= max_depth or n_clusters < 2:
            print(f"[Depth {depth}] Base case reached with {n_points} points, assigning single cluster")
            rows = np.arange(n_points)
            cols = np.zeros(n_points, dtype=int)
            C = csr_matrix((np.ones(n_points, dtype=dtype), (rows, cols)), shape=(n_points, 1))
            return None, None, None

        # Train clustering
        clustering_model = ClusteringModel.train(Z, config, dtype)
        cluster_labels = clustering_model.labels()
        cluster_counts = Counter(cluster_labels)
        print(f"[Depth {depth}] Cluster sizes: {cluster_counts}")

        valid_clusters = [cid for cid, cnt in cluster_counts.items() if cnt >= min_leaf_size]
    
        if len(valid_clusters) <= 1:
            print(f"[Depth {depth}] Only {len(valid_clusters)} valid clusters after pruning, refusing singleton cluster.")
            return None, None, None  # fail clustering here, no single cluster allowed

        avg_cluster_size = n_points / n_clusters
        cluster_id_offset = 0
        rows = []
        cols = []
        
        for cluster_id in valid_clusters:            
            count = cluster_counts[cluster_id]
            indices = np.where(cluster_labels == cluster_id)[0]
            Z_sub = Z[indices]
            
            if count > 2 * avg_cluster_size:
                sub_n_clusters = max(2, count // min_leaf_size)
                print(f"[Depth {depth}] Reclustering large cluster {cluster_id} with {count} points into {sub_n_clusters} clusters")
                
                sub_config = config.copy()
                sub_config["kwargs"] = sub_config.get("kwargs", {}).copy()
                sub_config["kwargs"]["n_clusters"] = sub_n_clusters
                
                _, C_sub, _ = ClusteringTrainer.train(
                    Z_sub, sub_config, min_leaf_size, depth + 1, dtype
                )
                if C_sub is None:
                    print(f"[Depth {depth}] Reclustering failed or empty, skipping cluster {cluster_id}")
                    continue
                
                C_sub = C_sub.tocsr()
                for row_local, cluster_local in zip(*C_sub.nonzero()):
                    rows.append(indices[row_local])
                    cols.append(cluster_id_offset + cluster_local)
                cluster_id_offset += C_sub.shape[1]
                
            else:
                print(f"[Depth {depth}] Accepting cluster {cluster_id} with {count} points")
                for i in indices:
                    rows.append(i)
                    cols.append(cluster_id_offset)
                cluster_id_offset += 1 
            
        if len(rows) == 0:
            print(f"[Depth {depth}] No valid clusters found, returning None")
            return None, None, None
        
        C = csr_matrix(
            (np.ones(len(rows), dtype=dtype), (rows, cols)), shape=(n_points, cluster_id_offset)
        )   
            
        return Z, C, clustering_model