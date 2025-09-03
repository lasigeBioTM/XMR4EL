import os
import gc
import shutil
import tempfile

import numpy as np

from pathlib import Path
from joblib import Parallel, delayed
from scipy.sparse import hstack, csr_matrix

from typing import Dict, Optional, Tuple

from xmr4el.models.classifier_wrapper.classifier_model import ClassifierModel

ranker_dir = Path(tempfile.mkdtemp(prefix="rankers_store_"))
 
class RankerTrainer:
    """Training routines for per-label rankers."""
    
    @staticmethod
    def save_ranker_temp(model: ClassifierModel, label: int) -> str:
        """Persist a temporary model for the given label."""
        sub_dir = ranker_dir / str(label)
        sub_dir.mkdir(parents=True, exist_ok=True)
        model.save(str(sub_dir))
        return str(sub_dir)

    @staticmethod
    def delete_ranker_temp() -> None:
        """Remove all temporary ranker models from disk."""
        shutil.rmtree(ranker_dir)

    @staticmethod
    def process_label(
        global_idx: int,
        X_cluster: np.ndarray,
        Y_col: np.ndarray,
        label_emb: np.ndarray,
        candidate_indices: np.ndarray,
        config: Dict,
    ) -> Tuple[int, Optional[str]]:
        """Train a ranker for a single label inside a cluster."""
        positive_global = Y_col.nonzero()[0]
        if positive_global.size == 0:
            return (global_idx, None)

        pos_mask = np.isin(candidate_indices, positive_global)
        positive_positions = np.where(pos_mask)[0]
        n_pos = positive_positions.size
        if n_pos < 2:
            print(f"[PID {os.getpid()}] SKIP label {global_idx}: only {n_pos} positives in cluster")
            return (global_idx, None)

        neg_mask = ~pos_mask
        n_neg_total = neg_mask.sum()
        if n_neg_total == 0:
            print(f"[PID {os.getpid()}] SKIP label {global_idx}: no negatives in cluster")
            return (global_idx, None)

        # --- ensure dot uses base embedding of correct width ---
        n_feats = X_cluster.shape[1]
        base_emb = (
            label_emb[:n_feats]
            if label_emb.shape[0] >= n_feats
            else np.pad(label_emb, (0, n_feats - label_emb.shape[0]))
        )
        scores = X_cluster.dot(base_emb).ravel()

        max_neg = n_pos * 15
        k = min(max_neg, n_neg_total)
        if k <= 0:
            print(f"[PID {os.getpid()}] SKIP label {global_idx}: k==0 negatives")
            return (global_idx, None)

        neg_positions_all = np.where(neg_mask)[0]
        neg_scores = scores[neg_mask]
        if k < neg_scores.size:
            top_k_idx = np.argpartition(-neg_scores, k)[:k]
            neg_positions = neg_positions_all[top_k_idx]
        else:
            neg_positions = neg_positions_all

        n_neg_selected = neg_positions.size
        if n_neg_selected < 2:
            print(f"[PID {os.getpid()}] SKIP label {global_idx}: only {n_neg_selected} negatives selected")
            return (global_idx, None)

        selected_positions = np.concatenate([positive_positions, neg_positions])
        Y_valid = np.zeros(selected_positions.size, dtype=np.int8)
        Y_valid[:n_pos] = 1

        X_valid = X_cluster[selected_positions]

        n = X_valid.shape[0]
        label_tile = csr_matrix((np.ones(n), (np.arange(n), np.zeros(n))), shape=(n, 1)).dot(
            csr_matrix(label_emb.reshape(1, -1))
        )

        X_combined = hstack([X_valid, label_tile])

        print(
            f"[PID {os.getpid()}] Training label {global_idx} with {n_pos} positives, {n_neg_selected} negatives "
            f"(cluster_size={X_cluster.shape[0]})"
        )

        if np.unique(Y_valid).size == 1:
            print(f"[PID {os.getpid()}] SKIP label {global_idx}: only one class in Y_valid after hard-neg selection")
            return (global_idx, None)
        
        model = ClassifierModel.train(X_combined, Y_valid, config, onevsrest=False)
        path = RankerTrainer.save_ranker_temp(model, global_idx)
        del model
        
        return (global_idx, path)


    @staticmethod
    def train(
        X: np.ndarray,
        Y: np.ndarray,
        Z: np.ndarray,
        M_TFN: np.ndarray,
        M_MAN: Optional[np.ndarray],
        cluster_labels: np.ndarray,
        config: Dict,
        local_to_global_idx: np.ndarray,
        n_label_workers: int = -1,
        parallel_backend: str = "threading",
    ) -> Dict[int, ClassifierModel]:
        """Train ranker models for all labels."""
        
        if M_MAN is None:
            M_bar = (M_TFN > 0).astype(int)
        else:
            M_bar = ((M_TFN + M_MAN) > 0).astype(int)
            
        ranker_models: Dict[int, ClassifierModel] = {}

        labels_by_cluster: Dict[int, list] = {}
        
        for local_idx, cluster_idx in enumerate(cluster_labels):
            labels_by_cluster.setdefault(cluster_idx, []).append(local_idx)

        cluster_mentions = {c: M_bar[:, c].nonzero()[0] for c in labels_by_cluster.keys()}

        # Build tasks
        tasks = []
        for cluster_idx, label_list in labels_by_cluster.items():
            candidate_indices = cluster_mentions.get(cluster_idx, np.array([], dtype=int))
            if candidate_indices.size == 0:
                continue
            X_cluster = X[candidate_indices]
            for local_idx in label_list:
                global_idx = int(local_to_global_idx[local_idx])
                label_emb = Z[local_idx]
                Y_col = Y[:, local_idx]
                tasks.append((global_idx, X_cluster, Y_col, label_emb, candidate_indices))

        results = Parallel(n_jobs=n_label_workers, backend=parallel_backend, verbose=10)(
            delayed(RankerTrainer.process_label)(
                global_idx, X_cluster, Y_col, label_emb, candidate_indices, config
            )
            for (global_idx, X_cluster, Y_col, label_emb, candidate_indices) in tasks
        )

        for global_idx, path in results:
            if path is not None:
                ranker_models[global_idx] = ClassifierModel.load(path)
            
        RankerTrainer.delete_ranker_temp()

        return ranker_models