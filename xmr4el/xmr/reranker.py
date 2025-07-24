import os
import numpy as np

from multiprocessing import Pool
from scipy.sparse import hstack

from xmr4el.models.classifier_wrapper.classifier_model import ClassifierModel

class SkeletonReranker():
    def __init__(self, labels, label_to_indices, reranker_config, n_label_workers=8):
        self.labels = labels
        self.label_to_indices = label_to_indices
        self.reranker_config = reranker_config
        self.n_label_workers = n_label_workers  # Number of parallel label trainers

    @staticmethod
    def _train_classifier(X, Y, config):
        """Train a classifier with the given config"""
        return ClassifierModel.train(X, Y, config, onevsrest=True)
    
    def _predict_classifier(self, model, X):
        """Predict using a trained classifier"""
        return ClassifierModel.predict(model, X)
    
    def _prepare_label_data(self, label_idx, X, Y, label_embs, mention_in_label_cluster, max_neg_per_pos=None):
        """Prepare data for a single label (used in parallel)"""
        valid_mention_mask = mention_in_label_cluster[:, label_idx].toarray().ravel().astype(bool)
        valid_indices = np.where(valid_mention_mask)[0]
        if len(valid_indices) == 0:
            return None

        X_valid = X[valid_indices]
        Y_col = Y[:, label_idx]
        Y_valid_sparse = Y_col[valid_indices]

        Y_valid = np.zeros(len(valid_indices), dtype=np.int8)
        Y_valid[Y_valid_sparse.nonzero()[0]] = 1

        if Y_valid.sum() == 0:
            return None

        if max_neg_per_pos is not None:
            pos_mask = Y_valid == 1
            num_pos = pos_mask.sum()
            if num_pos == 0:
                return None
            
            neg_indices = np.where(~pos_mask)[0]
            max_neg = max_neg_per_pos * num_pos
            
            if len(neg_indices) > max_neg:
                np.random.shuffle(neg_indices)
                neg_keep = neg_indices[:max_neg]
                keep_mask = pos_mask.copy()
                keep_mask[neg_keep] = True
                X_valid = X_valid[keep_mask]
                Y_valid = Y_valid[keep_mask]

        label_emb = label_embs[label_idx]
        rows = X_valid.shape[0]
        label_tile = np.tile(label_emb, (rows, 1))
        X_combined = hstack([X_valid, label_tile])

        return (label_idx, X_combined, Y_valid)

    def _train_label_parallel(self, args):
        """Wrapper for parallel training"""
        label_idx, X_label, Y_label = args
        if X_label is None:
            return (label_idx, None)
            
        print(f"[PID {os.getpid()}] Training reranker {label_idx} (32 threads)")
        self.reranker_config["n_jobs"] = 32  # Each reranker uses 32 threads
        model = self._train_classifier(X_label, Y_label, self.reranker_config)
        return (label_idx, model)

    def _build_dataset_and_train(self, X, Y, label_embs, C, M_TFN, M_MAN, max_neg_per_pos=None):
        M_bar = ((M_TFN + M_MAN) > 0).astype(int)
        label_cluster_mask = C.T
        mention_in_label_cluster = M_bar @ label_cluster_mask
        num_labels = Y.shape[1]

        # Prepare all label data first
        label_args = [
            self._prepare_label_data(
                label_idx, X, Y, label_embs, mention_in_label_cluster, max_neg_per_pos
            )
            for label_idx in range(num_labels)
        ]
        
        # Filter out None results (invalid labels)
        label_args = [x for x in label_args if x is not None]

        # Train in parallel (8 labels at once)
        with Pool(processes=self.n_label_workers) as pool:
            results = pool.map(self._train_label_parallel, label_args)

        return {label_idx: model for label_idx, model in results if model is not None}

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