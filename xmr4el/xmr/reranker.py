import os
import numpy as np

from scipy.sparse import csr_matrix

from collections import defaultdict
from joblib import Parallel, delayed

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
        model = self._train_classifier(X_label, Y_label)
        return label_idx, model

    def _train_labelwise_classifiers(self, datasets):
        results = Parallel(n_jobs=4)(
            delayed(self.train_one_label)(label_idx, *datasets[label_idx])
            for label_idx in datasets
        )
        return {label_idx: model for label_idx, model in results}
    
    def _build_dataset(self, X, Y, label_embs, C, M_TFN, M_MAN, max_neg_per_pos=None, n_jobs=-1):
        """
        Build reranker dataset using hard negatives from relevant clusters only.

        Optimizations:
        - Vectorized operations instead of loops.
        - Batch negative sampling.
        - Precompute cluster assignments.
        - Avoid repeated list appends.

        Args:
            X: np.ndarray, (N, d) mention embeddings
            Y: csr_matrix, (N, L) multi-label binarized labels for current node
            label_embs: np.ndarray, (L, d) embeddings for labels in this node
            C: csr_matrix, (L, K) label-to-cluster assignment for this node
            M_TFN: np.ndarray, (N, K) TFN cluster indicator matrix
            M_MAN: np.ndarray, (N, K) MAN cluster indicator matrix
            max_neg_per_pos: int, max negatives per positive (None for no capping)

        Returns:
            datasets: dict {label_idx: (X_rerank, y_rerank)}
        """
        M_bar = ((M_TFN + M_MAN) > 0).astype(int)
        num_mentions, num_labels = Y.shape
        # datasets = defaultdict(lambda: ([], []))

        # Precompute which mentions are in each label's clusters
        label_cluster_mask = C.T # (K, L) -> cluster-to-label
        mention_in_label_cluster = M_bar @ label_cluster_mask  # (N, L)
        
        def process_label(label_idx):

            # Get mentions in this label's clusters
            valid_mention_mask = mention_in_label_cluster[:, label_idx].astype(bool)
            valid_indices = np.where(valid_mention_mask)[0]

            if len(valid_indices) == 0:
                return None  # Skip labels with no valid mentions

            # Get input embeddings for valid mentions
            X_valid = X[valid_indices]

            # Get sparse column of Y for this label
            Y_col = Y[:, label_idx]
            Y_valid_sparse = Y_col[valid_indices]

            # Create binary labels
            Y_valid = np.zeros(len(valid_indices), dtype=np.int8)
            if isinstance(Y_valid_sparse, csr_matrix):
                Y_valid[Y_valid_sparse.nonzero()[0]] = 1
            else:
                Y_valid[Y_valid_sparse.nonzero()[0]] = 1

            if Y_valid.sum() == 0:
                return None

            # Combine mention and label embeddings
            label_emb = label_embs[label_idx]
            label_tile = np.tile(label_emb, (len(X_valid), 1))
            X_combined = np.concatenate([X_valid, label_tile], axis=1)
            
            # Apply negative capping if specified
            if max_neg_per_pos is not None:
                pos_mask = Y_valid == 1
                num_pos = pos_mask.sum()
                if num_pos == 0:
                    return None  # No positives for this label

                neg_indices = np.where(~pos_mask)[0]
                max_neg = max_neg_per_pos * num_pos

                if len(neg_indices) > max_neg:
                    # Randomly sample negatives
                    np.random.shuffle(neg_indices) # Put a seed
                    neg_keep = neg_indices[:max_neg]
                    
                    keep_mask = pos_mask.copy()
                    keep_mask[neg_keep] = True
                    
                    X_combined = X_combined[keep_mask]
                    Y_valid = Y_valid[keep_mask]

            return (label_idx, (X_combined, Y_valid))

        results = Parallel(n_jobs=n_jobs, prefer="processes")(
            delayed(process_label)(label_idx) for label_idx in range(num_labels)
        )
        
        # Filter out None entries
        return dict(filter(None, results))
        
        
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
        
        datasets = self._build_dataset(X_node, Y_node, label_embs, C, M_TFN, M_MAN, max_neg_per_pos=50, n_jobs=-1)
        classifiers = self._train_labelwise_classifiers(datasets)
        
        htree.set_reranker(classifiers)
        
        for child in htree.children.values():
            self.execute(child)
        
        