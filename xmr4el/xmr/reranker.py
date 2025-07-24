from collections import Counter
import os
import numpy as np

from multiprocessing import Pool
from scipy.sparse import hstack

from xmr4el.models.classifier_wrapper.classifier_model import ClassifierModel

class SkeletonReranker():
    def __init__(self, labels, label_to_indices, reranker_config, n_label_workers=16):
        self.labels = labels
        self.label_to_indices = label_to_indices
        self.reranker_config = reranker_config.copy()
        self.n_label_workers = min(n_label_workers, os.cpu_count())
        # Disable inner parallelism since we're parallelizing at label level
        self.reranker_config["kwargs"]["n_jobs"] = 1  

    @staticmethod
    def _train_classifier(X, Y, config):
        return ClassifierModel.train(X, Y, config, onevsrest=True)
    
    def _predict_classifier(self, model, X):
        return ClassifierModel.predict(model, X)

    # [PID 1993] Training label 0 with 100 positives, 65 hard negatives
    # [PID 1994] Training label 1 with 100 positives, 411 hard negatives
    def _process_label(self, args):
        """Process one label with top-K hard negatives from same cluster."""
        label_idx, X, Y, label_embs, M_bar, M_mentions, cluster_idx = args
        
        Y_col = Y[:, label_idx]
        positive_indices = Y_col.nonzero()[0]
        if len(positive_indices) <= 1:
            return (label_idx, None)
        
        # print(len(positive_indices), positive_indices)
        # print(C)
        # print(label_idx)
        # print(Y_col)

        # Step 1: Restrict to cluster members only (C is [mention x cluster])
        # Get cluster id of the label (assumes labels aligned with cluster columns)
        mention_in_cluster = M_mentions[:, cluster_idx].toarray().ravel().astype(bool)
        # print(label_cluster)
        candidate_indices = np.where(mention_in_cluster)[0]
        # print(candidate_indices.shape, candidate_indices)
        if len(candidate_indices) == 0:
            return (label_idx, None)

        # Step 2: Weak matcher scores only on cluster members
        scores = M_bar[candidate_indices, cluster_idx].toarray().ravel()
        
        # print(scores)

        # Step 3: Identify positives and negatives from cluster
        cluster_positive_mask = np.isin(candidate_indices, positive_indices)
        cluster_negative_mask = ~cluster_positive_mask

        if cluster_negative_mask.sum() == 0:
            return (label_idx, None)

        # Step 4: Select top-K negatives from candidates
        negative_scores = scores[cluster_negative_mask]
        negative_indices_sorted = np.argsort(-negative_scores)
        negative_candidates = candidate_indices[cluster_negative_mask][negative_indices_sorted]

        # Step 5: Combine positives + top-K negatives
        selected_indices = np.concatenate([positive_indices, negative_candidates])
        Y_valid = np.zeros(len(selected_indices), dtype=np.int8)
        Y_valid[:len(positive_indices)] = 1

        # Step 6: Build reranker input features
        X_valid = X[selected_indices]
        label_emb = label_embs[label_idx]
        label_tile = np.tile(label_emb, (X_valid.shape[0], 1))
        X_combined = hstack([X_valid, label_tile])

        print(f"[PID {os.getpid()}] Training label {label_idx} with {len(positive_indices)} positives, {len(negative_candidates)} hard negatives")

        model = self._train_classifier(X_combined, Y_valid, self.reranker_config)
        return (label_idx, model)


    def _build_dataset_and_train(self, X, Y, label_embs, C, M_TFN, M_MAN, cluster_labels):
        M_bar = ((M_TFN + M_MAN) > 0).astype(int)
        num_labels = Y.shape[1]
        reranker_models = {}

        # Compute mention-to-cluster matrix: M_mentions = (Y @ C) > 0
        # Y: mention x label, C: label x cluster
        M_mentions = (Y @ C) > 0

        # Create all tasks first (memory efficient)
        tasks = []
        for label_idx in range(num_labels):
            cluster_idx = cluster_labels[label_idx] # C was below
            tasks.append((label_idx, X, Y, label_embs, M_bar, M_mentions, cluster_idx))
        
        # Process in parallel with dynamic batching
        with Pool(processes=self.n_label_workers) as pool:
            # Use imap_unordered with chunksize=1 for optimal load balancing
            for result in pool.imap_unordered(self._process_label, tasks, chunksize=1):
                label_idx, model = result
                if model is not None:
                    reranker_models[label_idx] = model
                    # Immediately release memory after processing
                    del model
        
        return reranker_models

    def execute(self, htree):
        X_node = htree.X
        Y_node = htree.Y
        C = htree.C
        M = htree.M
        Z = htree.Z
        cluster_labels = htree.cluster_labels

        reranker_models = self._build_dataset_and_train(
            X_node, Y_node, Z, C, M, self._predict_classifier(htree.classifier, X_node), cluster_labels
        )

        htree.set_reranker(reranker_models)

        for child in htree.children.values():
            self.execute(child)