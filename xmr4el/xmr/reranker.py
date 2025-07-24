from collections import Counter
import os
import numpy as np

from multiprocessing import Pool
from scipy.sparse import hstack

from xmr4el.models.classifier_wrapper.classifier_model import ClassifierModel

class SkeletonReranker():
    def __init__(self, labels, label_to_indices, reranker_config, n_label_workers=1):
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

    def _process_label(self, args):
        """Process one label in isolation"""
        label_idx, X, Y, label_embs, C, M_bar = args
        
        # Calculate label-specific data
        label_cluster_mask = C.T
        mention_in_label_cluster = M_bar @ label_cluster_mask
        valid_mention_mask = mention_in_label_cluster[:, label_idx].toarray().ravel().astype(bool)
        valid_indices = np.where(valid_mention_mask)[0]
        
        if len(valid_indices) == 0:
            return (label_idx, None)

        X_valid = X[valid_indices]
        Y_col = Y[:, label_idx]
        Y_valid_sparse = Y_col[valid_indices] # .toarray().ravel().astype(np.int8)
        
        Y_valid = np.zeros(len(valid_indices), dtype=np.int8)
        Y_valid[Y_valid_sparse.nonzero()[0]] = 1
        
        if Y_valid.sum() <= 1:
            return (label_idx, None)

        # Prepare features
        label_emb = label_embs[label_idx]
        label_tile = np.tile(label_emb, (X_valid.shape[0], 1))
        X_combined = hstack([X_valid, label_tile])

        print(f"[PID {os.getpid()}] Training label {label_idx}")
        print(label_idx, X_combined.shape, Y_valid.shape, Counter(Y_valid))
        # print(Y_valid.sum())
        model = self._train_classifier(X_combined, Y_valid, self.reranker_config)
        return (label_idx, model)

    def _build_dataset_and_train(self, X, Y, label_embs, C, M_TFN, M_MAN):
        M_bar = ((M_TFN + M_MAN) > 0).astype(int)
        num_labels = Y.shape[1]
        reranker_models = {}

        # Create all tasks first (memory efficient)
        tasks = []
        for label_idx in range(num_labels):
            tasks.append((label_idx, X, Y, label_embs, C, M_bar))
        
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

        reranker_models = self._build_dataset_and_train(
            X_node, Y_node, Z, C, M, self._predict_classifier(htree.classifier, X_node)
        )

        htree.set_reranker(reranker_models)

        for child in htree.children.values():
            self.execute(child)