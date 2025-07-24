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
        Y_valid = Y_col[valid_indices].toarray().ravel().astype(np.int8)

        if Y_valid.sum() == 0:
            return (label_idx, None)

        # Prepare features
        label_emb = label_embs[label_idx]
        rows = X_valid.shape[0]
        label_tile = np.tile(label_emb, (rows, 1))
        X_combined = hstack([X_valid, label_tile])

        print(f"[PID {os.getpid()}] Training label {label_idx}")
        model = self._train_classifier(X_combined, Y_valid, self.reranker_config)
        return (label_idx, model)

    def _build_dataset_and_train(self, X, Y, label_embs, C, M_TFN, M_MAN):
        M_bar = ((M_TFN + M_MAN) > 0).astype(int)
        num_labels = Y.shape[1]
        reranker_models = {}

        # Process in batches to limit memory usage
        for batch_start in range(0, num_labels, self.n_label_workers):
            batch_end = min(batch_start + self.n_label_workers, num_labels)
            print(f"Processing labels {batch_start}-{batch_end-1}")

            # Prepare arguments for this batch
            args = [
                (label_idx, X, Y, label_embs, C, M_bar)
                for label_idx in range(batch_start, batch_end)
            ]

            # Process batch in parallel
            with Pool(processes=self.n_label_workers) as pool:
                batch_results = pool.map(self._process_label, args)

            # Store results
            for label_idx, model in batch_results:
                if model is not None:
                    reranker_models[label_idx] = model

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