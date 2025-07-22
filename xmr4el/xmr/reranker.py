import numpy as np

from collections import defaultdict

from xmr4el.models.classifier_wrapper.classifier_model import ClassifierModel


class SkeletonReranker():
    
    def __init__(self, labels, label_to_indices, classifier_config):
        self.labels = labels
        self.label_to_indices = label_to_indices
        
        self.classifier_config = classifier_config
    
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
        return ClassifierModel.train(X, Y, self.classifier_config, onevsrest=True)
    
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
    
    def _train_labelwise_classifiers(self, datasets):
        """
        Train a separate binary classifier for each label.

        Args:
            datasets: dict[label_idx] = (X_label, y_label)

        Returns:
            dict[label_idx] = trained_model
        """
        classifiers = {}

        for label_idx, (X_label, Y_label) in datasets.items():
            # Initialize your classifier, e.g., logistic regression, linear SVM
            model = self._train_classifier(X_label, Y_label)
            classifiers[label_idx] = model

        return classifiers
    
    def _build_dataset(self, X, Y, label_embs, C, M_TFN, M_MAN, max_neg_per_pos=None):
        """
        Build reranker dataset using hard negatives from relevant clusters only.

        Optimizations:
        - Vectorized operations instead of loops.
        - Batch negative sampling.
        - Precompute cluster assignments.
        - Avoid repeated list appends.

        Args:
            X: np.ndarray, (N, d) mention embeddings
            Y: np.ndarray, (N, L) multi-label binarized labels for current node
            label_embs: np.ndarray, (L, d) embeddings for labels in this node
            C: np.ndarray, (L, K) label-to-cluster assignment for this node
            M_TFN: np.ndarray, (N, K) TFN cluster indicator matrix
            M_MAN: np.ndarray, (N, K) MAN cluster indicator matrix
            max_neg_per_pos: int, max negatives per positive (None for no capping)

        Returns:
            datasets: dict {label_idx: (X_rerank, y_rerank)}
        """
        M_bar = ((M_TFN + M_MAN) > 0).astype(int)
        num_mentions, num_labels = Y.shape
        datasets = defaultdict(lambda: ([], []))

        # Precompute which mentions are in each label's clusters
        label_cluster_mask = C.T  # (K, L) -> cluster-to-label
        mention_in_label_cluster = M_bar @ label_cluster_mask  # (N, L)

        for label_idx in range(num_labels):
            # Get mentions in this label's clusters
            valid_mention_mask = mention_in_label_cluster[:, label_idx].astype(bool)
            valid_indices = np.where(valid_mention_mask)[0]

            if len(valid_indices) == 0:
                continue  # Skip labels with no valid mentions

            # Get embeddings for valid mentions
            X_valid = X[valid_indices]
            Y_valid = Y[valid_indices, label_idx]

            # Combine mention and label embeddings
            label_emb = label_embs[label_idx]
            label_tile = np.tile(label_emb, (len(X_valid), 1))
            X_combined = np.concatenate([X_valid, label_tile], axis=1)
            
            # Apply negative capping if specified
            if max_neg_per_pos is not None:
                pos_mask = Y_valid == 1
                num_pos = pos_mask.sum()
                if num_pos == 0:
                    continue  # No positives for this label

                neg_mask = ~pos_mask
                neg_indices = np.where(neg_mask)[0]
                num_neg = len(neg_indices)
                max_neg = max_neg_per_pos * num_pos

                if num_neg > max_neg:
                    # Randomly sample negatives
                    np.random.shuffle(neg_indices) # Put a seed
                    neg_keep = neg_indices[:max_neg]
                    keep_mask = pos_mask | np.isin(np.arange(len(Y_valid)), neg_keep)
                    X_combined = X_combined[keep_mask]
                    Y_valid = Y_valid[keep_mask]

            if len(X_combined) > 0:
                datasets[label_idx] = (X_combined, Y_valid)

        return dict(datasets)
    
    def execute(self, htree):
        X_node = htree.X #  mention/input embeddings
        Y_node = htree.Y #  Multi label binary matrix
        C = htree.C # Set Label to cluster matrix for this node.
        M = htree.M # Mention to cluster labels
        Z = htree.Z # Embedding matrix of all labels under the current node.
        
        label_embs = Z
        
        # Teacher Forcing Negatives (TFN), ground-truth input-to-cluster assignment for the input 
        M_TFN = M 
        # Matcher Aware Negatives (MAN),  matcher-aware hard negatives for each training instance.
        M_MAN = self._predict_classifier(htree.classifier, X_node) 
        
        datasets = self._build_dataset(X_node, Y_node, label_embs, C, M_TFN, M_MAN, max_neg_per_pos=None)
        classifiers = self._train_labelwise_classifiers(datasets)
        
        htree.set_reranker(classifiers)
        
        for child in htree.children.values():
            self.execute(child)
        
        