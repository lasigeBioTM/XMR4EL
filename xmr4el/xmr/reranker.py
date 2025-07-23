import os
import numpy as np

from itertools import islice
from scipy.sparse import csr_matrix, hstack
from joblib import Parallel, delayed, parallel_backend

from tqdm import tqdm

from xmr4el.models.classifier_wrapper.classifier_model import ClassifierModel


class SkeletonReranker():
    
    def __init__(self, labels, label_to_indices, reranker_config):
        self.labels = labels
        self.label_to_indices = label_to_indices
        
        self.reranker_config = reranker_config
    
    def _train_classifier(self, X, Y):
        """
        Train a node-level classifier to route to its children.

        Args:
            X_node (np.ndarray): Embeddings of samples at this node, shape (N,d).
            true_ids (List[Any]): Corresponding gold entity IDs, length N.
            children (List[XMRTree]): Node's immediate children.
            htree: Current tree node (for setters).
        """
        """One VS Rest Classifier"""
        return ClassifierModel.train(X, Y, self.reranker_config, onevsrest=True)
    
    def _predict_classifier(self, model, X):
        """
        Train a node-level classifier to route to its children.

        Args:
            X_node (np.ndarray): Embeddings of samples at this node, shape (N,d).
            true_ids (List[Any]): Corresponding gold entity IDs, length N.
            children (List[XMRTree]): Node's immediate children.
            htree: Current tree node (for setters).
        """
        """One VS Rest Classifier"""
        return ClassifierModel.predict(model, X)
    
    def train_one_label(self, label_idx, X_label, Y_label):
        print(f"Ranker number {label_idx}")
        model = self._train_classifier(X_label, Y_label)
        return label_idx, model

    def _train_labelwise_classifiers(self, dataset_stream, buffer_size=4):
        rerankers = {}
        while True:
            batch = list(islice(dataset_stream, buffer_size))
            if not batch:
                break

            # Train in parallel
            results = Parallel(n_jobs=buffer_size, prefer="processes")(
                delayed(self.train_one_label)(label_idx, X_label, Y_label)
                for label_idx, (X_label, Y_label) in batch
            )

            for label_idx, model in results:
                rerankers[label_idx] = model
        return rerankers
    
    def _build_dataset_parallel_streamed(self, X, Y, label_embs, C, M_TFN, M_MAN, max_neg_per_pos=None, n_jobs=-1):
        M_bar = ((M_TFN + M_MAN) > 0).astype(int)
        label_cluster_mask = C.T
        mention_in_label_cluster = M_bar @ label_cluster_mask
        num_labels = Y.shape[1]

        def process_label(label_idx):
            valid_mention_mask = mention_in_label_cluster[:, label_idx].astype(bool)
            valid_indices = np.where(valid_mention_mask)[0]
            if len(valid_indices) == 0:
                return None

            X_valid = X[valid_indices]
            Y_col = Y[:, label_idx]
            Y_valid_sparse = Y_col[valid_indices]

            Y_valid = np.zeros(len(valid_indices), dtype=np.int8)
            if isinstance(Y_valid_sparse, csr_matrix):
                Y_valid[Y_valid_sparse.nonzero()[0]] = 1
            else:
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
            label_tile_sparse = csr_matrix(np.broadcast_to(label_emb, (rows, label_emb.shape[0])))
            X_combined = hstack([X_valid, label_tile_sparse])

            return (label_idx, (X_combined, Y_valid))

        # Submit jobs in parallel and consume results lazily
        label_indices = list(range(num_labels))
        with parallel_backend("loky", n_jobs=n_jobs):
            for result in tqdm(
                Parallel()(delayed(process_label)(idx) for idx in label_indices),
                total=num_labels,
                desc="Building reranker dataset"
            ):
                if result is not None:
                    yield result
        
    def execute(self, htree):
        X_node = htree.X #  mention/input embeddings
        Y_node = htree.Y #  Multi label binary matrix
        C = csr_matrix(htree.C) # Set Label to cluster matrix for this node.
        M = htree.M # Mention to cluster labels
        Z = htree.Z # Embedding matrix of all labels under the current node.
        
        label_embs = Z
        
        # Teacher Forcing Negatives (TFN), ground-truth input-to-cluster assignment for the input 
        M_TFN = M 
        # Matcher Aware Negatives (MAN),  matcher-aware hard negatives for each training instance.
        M_MAN = self._predict_classifier(htree.classifier, X_node) 
        
        dataset_stream = self._build_dataset_parallel_streamed(
            X_node, Y_node, label_embs, C, M_TFN, M_MAN,
            max_neg_per_pos=None,
            n_jobs=4
        )
        reranker_models = self._train_labelwise_classifiers(dataset_stream)
        
        htree.set_reranker(reranker_models)
        
        for child in htree.children.values():
            self.execute(child)
        
        