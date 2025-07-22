import numpy as np
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
        return ClassifierModel.train(X, Y, self.classifier_config)
    
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
    
    def _build_dataset(self, X, Y, label_embs, C, M_TFN, M_MAN, max_neg_per_pos=None, shuffle_negatives=True):
        """
        Build reranker dataset using hard negatives from relevant clusters only.

        Args:
            X: np.ndarray, (N, d) mention embeddings
            Y: np.ndarray, (N, L) multi-label binarized labels for current node
            label_embs: np.ndarray, (L, d) embeddings for labels in this node
            C: np.ndarray, (L, K) label-to-cluster assignment for this node
            M: np.ndarray, (N, K) input-to-cluster indicator matrix for this node

        Returns:
            X_rerank: np.ndarray, concatenated mention-label embeddings (num_samples, 2*d)
            y_rerank: np.ndarray, binary labels (num_samples,)
        """
        
        X_rerank = []
        Y_rerank = []
        
        num_mentions, num_labels = Y.shape
        num_clusters = C.shape[1]

        M_bar = ((M_TFN + M_MAN) > 0).astype(int)

        # print(num_mentions)

        cluster_to_labels = {k: set(np.where(C[:, k] == 1)[0]) for k in range(num_clusters)}
        
        for i in range(num_mentions):
            mention_emb = X[i]
            relevant_clusters = np.where(M_bar[i] == 1)[0]
            
            candidate_labels = set()
            for cluster_id in relevant_clusters:
                candidate_labels |= cluster_to_labels[cluster_id]
            candidate_labels = list(candidate_labels)
            
            # Positives and negatives in candidate labels
            pos_labels = [l for l in candidate_labels if Y[i, l] == 1]
            neg_labels = [l for l in candidate_labels if Y[i, l] == 0]
            
            
            if shuffle_negatives:
                np.random.shuffle(neg_labels)

            # Cap negatives if needed
            if max_neg_per_pos is not None and len(pos_labels) > 0:
                neg_cap = max_neg_per_pos * len(pos_labels)
                neg_labels = neg_labels[:neg_cap]
            
            # Positive samples
            for pos in pos_labels:
                pos_emb = label_embs[pos]
                combined_emb = np.concatenate([mention_emb, pos_emb])
                X_rerank.append(combined_emb)
                Y_rerank.append(1)
            
            # Negative samples
            for neg in neg_labels:
                neg_emb = label_embs[neg]
                combined_emb = np.concatenate([mention_emb, neg_emb])
                X_rerank.append(combined_emb)
                Y_rerank.append(0)
            
        X_rerank = np.array(X_rerank)
        Y_rerank = np.array(Y_rerank)
        return X_rerank, Y_rerank
    
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
        
        X_rerank, Y_rerank = self._build_dataset(X_node, Y_node, label_embs, C, M_TFN, M_MAN, max_neg_per_pos=10)
        
        print(X_rerank.shape, Y_rerank.shape)
        
        model = self._train_classifier(X_rerank, Y_rerank)
        
        htree.set_reranker(model)
        
        for child in htree.children.values():
            self.execute(child)
        
        