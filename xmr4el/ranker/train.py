import os

import numpy as np

from multiprocessing import Pool
from scipy.sparse import hstack

from xmr4el.models.classifier_wrapper.classifier_model import ClassifierModel


class ReRankerTrainer():
    
    @staticmethod 
    def process_label(args):
            label_idx, X, Y, Z, M_bar, M_mentions, cluster_idx, config = args
            
            Y_col = Y[:, label_idx]
            positive_indices = Y_col.nonzero()[0]
            if len(positive_indices) <= 1:
                return (label_idx, None)

            # Step 1: Restrict to cluster members only (C is [mention x cluster])
            # Get cluster id of the label (assumes labels aligned with cluster columns)
            mention_in_cluster = M_mentions[:, cluster_idx].ravel().astype(bool) # toarray
            # print(label_cluster)
            candidate_indices = np.where(mention_in_cluster)[0]
            # print(candidate_indices.shape, candidate_indices)
            if len(candidate_indices) == 0:
                return (label_idx, None)

            # Step 2: Weak matcher scores only on cluster members
            scores = M_bar[candidate_indices, cluster_idx].ravel() # toarray

            # Step 3: Identify positives and negatives from cluster
            cluster_positive_mask = np.isin(candidate_indices, positive_indices)
            cluster_negative_mask = ~cluster_positive_mask

            if cluster_negative_mask.sum() == 0:
                return (label_idx, None)

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
            label_emb = Z[label_idx]
            label_tile = np.tile(label_emb, (X_valid.shape[0], 1))
            X_combined = hstack([X_valid, label_tile])

            print(f"[PID {os.getpid()}] Training label {label_idx} with {len(positive_indices)} positives, {len(negative_candidates)} hard negatives")

            model = ClassifierModel.train(X_combined, Y_valid, config, onevsrest=True)
            return (label_idx, model)
    
    # Mmentions and M TFN is the same thing
    @staticmethod
    def train(X, Y, Z, M_TFN, M_MAN, cluster_labels, config, n_label_workers=4):
        M_bar = ((M_TFN + M_MAN) > 0).astype(int)
        num_labels = Y.shape[1]
        reranker_models = {}
        
        tasks = []
        for label_idx in range(num_labels):
            cluster_idx = cluster_labels[label_idx] # C was below
            tasks.append((label_idx, X, Y, Z, M_bar, M_TFN, cluster_idx, config))
            
        # Process in parallel with dynamic batching
        with Pool(processes=n_label_workers, maxtasksperchild=5) as pool:
            # Use imap_unordered with chunksize=1 for optimal load balancing
            for result in pool.imap_unordered(ReRankerTrainer.process_label, tasks, chunksize=1):
                label_idx, model = result
                if model is not None:
                    reranker_models[label_idx] = model
                    print(f"Stored model for label {label_idx}/{num_labels})")
                    # Immediately release memory after processing
                    del model
                    
        # M_MAN -> self._predict_classifier(leaf.classifier, X_node) NEED MATCHER
        return reranker_models