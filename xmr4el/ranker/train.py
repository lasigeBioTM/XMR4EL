import os

import numpy as np

from multiprocessing import get_context
from joblib import Parallel, delayed

from scipy.sparse import hstack

from xmr4el.models.classifier_wrapper.classifier_model import ClassifierModel


class ReRankerTrainer():
    
    @staticmethod 
    def process_label(global_idx, local_idx, X, Y, Z, M_bar, M_mentions, cluster_idx, config):  
            # Extract the column for this label from Y (sparse)          
            Y_col = Y[:, local_idx]
            positive_indices = Y_col.nonzero()[0]
            if len(positive_indices) <= 1:
                return (global_idx, None)

            # Step 1: Restrict to cluster members only (C is [mention x cluster])
            # Step 1: Get mention indices assigned to this cluster from M_mentions (sparse)
            candidate_indices = M_mentions[:, cluster_idx].nonzero()[0]
            if len(candidate_indices) == 0:
                return (global_idx, None)

            # Step 2: Weak matcher scores only on cluster members
            scores_sparse = M_bar[candidate_indices, :][:, cluster_idx]
            
            scores = scores_sparse.toarray().flatten()

            # Step 3: Identify positives and negatives from cluster
            cluster_positive_mask = np.isin(candidate_indices, positive_indices)
            cluster_negative_mask = ~cluster_positive_mask
            if cluster_negative_mask.sum() == 0:
                return (global_idx, None)

            # Step 4: Select top-K negatives from candidates
            negative_scores = scores[cluster_negative_mask]
            negative_indices_sorted = np.argsort(-negative_scores)
            negative_candidates = candidate_indices[cluster_negative_mask][negative_indices_sorted]# [:200]

            # Step 5: Combine positives + top-K negatives
            selected_indices = np.concatenate([positive_indices, negative_candidates])
            Y_valid = np.zeros(len(selected_indices), dtype=np.int8)
            Y_valid[:len(positive_indices)] = 1

            # Step 6: Build reranker input features
            X_valid = X[selected_indices]
            label_emb = Z[local_idx]
            label_tile = np.tile(label_emb, (X_valid.shape[0], 1))
            X_combined = hstack([X_valid, label_tile])

            print(f"[PID {os.getpid()}] Training label {global_idx} with {len(positive_indices)} positives, {len(negative_candidates)} hard negatives")

            model = ClassifierModel.train(X_combined, Y_valid, config, onevsrest=True)
            return (global_idx, model)
    
    # Mmentions and M TFN is the same thing
    @staticmethod
    def train(X, Y, Z, M_TFN, M_MAN, cluster_labels, config, local_to_global_idx, n_label_workers=4):
        M_bar = ((M_TFN + M_MAN) > 0).astype(int)
        num_labels = Y.shape[1]
        reranker_models = {}

        results = Parallel(n_jobs=n_label_workers, backend="loky", verbose=10)(
            delayed(ReRankerTrainer.process_label)(
                int(local_to_global_idx[local_idx]),
                local_idx,
                X,
                Y,
                Z,
                M_bar,
                M_TFN,
                cluster_labels[local_idx],
                config
            )
            for local_idx in range(num_labels)
        )

        for global_idx, model in results:
            if model is not None:
                reranker_models[global_idx] = model
                print(f"Stored model for label {global_idx}/{num_labels})")
                del model  # Free memory

        return reranker_models