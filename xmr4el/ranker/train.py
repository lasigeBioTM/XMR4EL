import os

import numpy as np

from joblib import Parallel, delayed

from scipy.sparse import hstack

from xmr4el.models.classifier_wrapper.classifier_model import ClassifierModel


class ReRankerTrainer():
    
    @staticmethod
    def process_label(global_idx, local_idx, X, Z, Y_col, M_bar, M_mentions_col, cluster_idx, config):
        positive_indices = Y_col.nonzero()[0]
        if len(positive_indices) <= 1:
            return (global_idx, None)

        candidate_indices = M_mentions_col.nonzero()[0]
        if len(candidate_indices) == 0:
            return (global_idx, None)

        scores_sparse = M_bar[candidate_indices, cluster_idx]
        scores = scores_sparse.toarray().flatten()

        cluster_positive_mask = np.isin(candidate_indices, positive_indices)
        cluster_negative_mask = ~cluster_positive_mask
        if cluster_negative_mask.sum() == 0:
            return (global_idx, None)

        negative_scores = scores[cluster_negative_mask]
        negative_indices_sorted = np.argsort(-negative_scores)
        negative_candidates = candidate_indices[cluster_negative_mask][negative_indices_sorted]

        selected_indices = np.concatenate([positive_indices, negative_candidates])
        Y_valid = np.zeros(len(selected_indices), dtype=np.int8)
        Y_valid[:len(positive_indices)] = 1

        X_valid = X[selected_indices]
        label_emb = Z[local_idx]
        label_tile = np.tile(label_emb, (X_valid.shape[0], 1))
        X_combined = hstack([X_valid, label_tile])

        print(f"[PID {os.getpid()}] Training label {global_idx} with {len(positive_indices)} positives, {len(negative_candidates)} hard negatives")
        model = ClassifierModel.train(X_combined, Y_valid, config, onevsrest=True)
        return (global_idx, model)


    @staticmethod
    def train(X, Y, Z, M_TFN, M_MAN, cluster_labels, config, local_to_global_idx, n_label_workers=4):
        M_bar = ((M_TFN + M_MAN) > 0).astype(int)
        reranker_models = {}
        num_labels = Y.shape[1]

        results = Parallel(n_jobs=n_label_workers, backend="loky", verbose=10)(
            delayed(ReRankerTrainer.process_label)(
                int(local_to_global_idx[local_idx]),
                local_idx,
                X,
                Z,
                Y[:, local_idx],
                M_bar,
                M_TFN[:, cluster_labels[local_idx]],
                cluster_labels[local_idx],
                config
            )
            for local_idx in range(num_labels)
        )

        for global_idx, model in results:
            if model is not None:
                reranker_models[global_idx] = model
                del model

        return reranker_models