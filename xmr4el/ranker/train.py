import os
import gc
import psutil

import numpy as np

from joblib import Parallel, delayed

from scipy.sparse import hstack, csr_matrix

from xmr4el.models.classifier_wrapper.classifier_model import ClassifierModel


class ReRankerTrainer:

    @staticmethod
    def process_label(global_idx, X_cluster, Y_col, label_emb, candidate_indices, config):
        """
        X_cluster: CSR (n_cluster_mentions, n_features)  (rows in same order as candidate_indices)
        Y_col:    global mention indices for positives for this label (sparse col or array)
        label_emb: 1D dense array
        candidate_indices: global mention indices for this cluster
        """
        # global positives for this label
        positive_global = Y_col.nonzero()[0]
        if positive_global.size == 0:
            return (global_idx, None)

        # map positives into cluster local positions
        pos_mask = np.isin(candidate_indices, positive_global)  # len == n_cluster_mentions
        positive_positions = np.where(pos_mask)[0]             # positions inside X_cluster
        n_pos = positive_positions.size
        if n_pos < 2:
            # Not enough positives inside this cluster to safely train (skipping)
            print(f"[PID {os.getpid()}] SKIP label {global_idx}: only {n_pos} positives in cluster")
            return (global_idx, None)

        neg_mask = ~pos_mask
        n_neg_total = neg_mask.sum()
        if n_neg_total == 0:
            print(f"[PID {os.getpid()}] SKIP label {global_idx}: no negatives in cluster")
            return (global_idx, None)

        # compute dot-product scores for mentions in cluster (robustly convert to 1d array)
        scores_raw = X_cluster.dot(label_emb)
        # convert matrix/array-like to 1d numpy array
        try:
            scores = np.asarray(scores_raw).ravel()
        except Exception:
            # fallback
            scores = np.array(scores_raw).ravel()

        # select hard negatives: top k by score
        max_neg = n_pos * 15
        k = min(max_neg, n_neg_total)
        if k <= 0:
            print(f"[PID {os.getpid()}] SKIP label {global_idx}: k==0 negatives")
            return (global_idx, None)

        neg_positions_all = np.where(neg_mask)[0]  # positions inside cluster of all negatives
        neg_scores = scores[neg_mask]

        if k < neg_scores.size:
            top_k_idx = np.argpartition(-neg_scores, k)[:k]  # indices into neg_scores
            neg_positions = neg_positions_all[top_k_idx]     # positions inside cluster
        else:
            neg_positions = neg_positions_all

        n_neg_selected = neg_positions.size
        # require at least 2 examples for each class to avoid sklearn validation-split errors
        if n_neg_selected < 2:
            print(f"[PID {os.getpid()}] SKIP label {global_idx}: only {n_neg_selected} negatives selected")
            return (global_idx, None)

        # combine positives + negatives (positions relative to X_cluster)
        selected_positions = np.concatenate([positive_positions, neg_positions])
        # labels: first n_pos are positives
        Y_valid = np.zeros(selected_positions.size, dtype=np.int8)
        Y_valid[:n_pos] = 1

        # slice cluster matrix for training
        X_valid = X_cluster[selected_positions]

        # build label tile (dense repeated -> convert to sparse)
        label_tile = csr_matrix(np.repeat(label_emb[np.newaxis, :], X_valid.shape[0], axis=0))

        # concatenate [mention_features | label_embedding]
        X_combined = hstack([X_valid, label_tile])
        X_combined = csr_matrix(X_combined)

        print(f"[PID {os.getpid()}] Training label {global_idx} with {n_pos} positives, {n_neg_selected} negatives "
              f"(cluster_size={X_cluster.shape[0]})")

        model = ClassifierModel.train(X_combined, Y_valid, config, onevsrest=False)
        return (global_idx, model)

    @staticmethod
    def train(X, Y, Z, M_TFN, M_MAN, cluster_labels, config, local_to_global_idx,
              layer=None, n_label_workers=4, parallel_backend="threading"):
        """
        Pre-slice mentions by cluster and run per-label reranker training in parallel.

        - X: full mention-feature matrix (only used here to create X_cluster slices)
        - Y: mention x label matrix (sparse or dense)
        - Z: label embeddings (num_labels, emb_dim)
        - M_TFN, M_MAN: mention-label matrices used to compute M_bar
        - cluster_labels: array mapping local label idx -> cluster idx
        - local_to_global_idx: map local label idx -> global label id
        """
        M_bar = ((M_TFN + M_MAN) > 0).astype(int)
        reranker_models = {}
        num_labels = Y.shape[1]

        if layer and layer > 0:
            Z = np.hstack([Z, np.zeros((num_labels, 3 * layer), dtype=Z.dtype)])

        # Group labels by cluster
        labels_by_cluster = {}
        for local_idx, cluster_idx in enumerate(cluster_labels):
            labels_by_cluster.setdefault(cluster_idx, []).append(local_idx)

        # Precompute mention (global) indices for each cluster (only once)
        cluster_mentions = {
            c: M_bar[:, c].nonzero()[0] for c in labels_by_cluster.keys()
        }

        # Build tasks: (global_idx, X_cluster, Y_col, label_emb, candidate_indices)
        tasks = []
        for cluster_idx, label_list in labels_by_cluster.items():
            candidate_indices = cluster_mentions.get(cluster_idx, np.array([], dtype=int))
            if candidate_indices.size == 0:
                continue
            X_cluster = X[candidate_indices]  # slice once per cluster
            for local_idx in label_list:
                global_idx = int(local_to_global_idx[local_idx])
                label_emb = Z[local_idx]
                Y_col = Y[:, local_idx]
                tasks.append((global_idx, X_cluster, Y_col, label_emb, candidate_indices))

        # Run parallel jobs (ensure tasks unpacking matches process_label signature)
        results = Parallel(n_jobs=n_label_workers, backend=parallel_backend, verbose=10)(
            delayed(ReRankerTrainer.process_label)(
                global_idx, X_cluster, Y_col, label_emb, candidate_indices, config
            )
            for (global_idx, X_cluster, Y_col, label_emb, candidate_indices) in tasks
        )

        # Collect trained rerankers
        for global_idx, model in results:
            if model is not None:
                reranker_models[global_idx] = model
                del model

        return reranker_models