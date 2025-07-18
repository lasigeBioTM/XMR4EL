import gc
import logging
import random

import numpy as np

from scipy.sparse import csr_matrix

from collections import Counter, defaultdict

from sklearn.preprocessing import normalize
from sklearn.decomposition import TruncatedSVD

from sklearn.random_projection import SparseRandomProjection

from xmr4el.featurization.transformers import Transformer
from xmr4el.featurization.vectorizers import Vectorizer
from xmr4el.ranker.candidate_retrieval import CandidateRetrieval
from xmr4el.ranker.cross_encoder import CrossEncoder


# LOGGER = logging.getLogger(__name__)
# logging.basicConfig(
#     level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
# )


class Predict():
    """
    Extreme Multi-label Ranking (XMR) prediction system that handles:
    - Feature transformation and dimensionality reduction
    - Hierarchical tree traversal for efficient prediction
    - Two-stage candidate retrieval and ranking
    - Parallel processing for high throughput
    
    The prediction pipeline consists of:
    1. Text embedding generation using trained vectorizer
    2. Transformer embedding generation
    3. Dimensionality reduction and normalization
    4. Hierarchical tree traversal for candidate selection
    5. Two-stage ranking (retrieval + cross-encoder)
    6. Sparse output generation
    """
        
    @staticmethod
    def _reduce_dimensionality(emb, n_features, random_state=0):
        """Reduces the dimensionality of embeddings

        Args:
            emb (np.array): Embeddings to reduce
            n_features (int): Target number of features
            random_state (int): Random seed for reproducibility
            
        Returns:
            np.array: Reduced embeddings with shape (n_samples, n_features)
                     or original if reduction not possible
        """
        n_samples, n_features_emb = emb.shape
        effective_dim = min(n_features, emb.shape[0])
        
        if n_features_emb >= n_features_emb or effective_dim <= 0:
            # LOGGER.info("Maintaining original number of features," 
            #             f"impossible to reduce, min({n_samples}, {n_features})={n_features_emb}")
            return emb  # Return original if reduction not possible
        
        # Perform PCA with effective dimension
        svd = TruncatedSVD(n_components=n_features, random_state=random_state)
        dense_emb = svd.fit_transform(emb) # turns it into dense auto
        
        return dense_emb
    
    @staticmethod
    def _train_vectorizer(trn_corpus, config, dtype=np.float32):
        """Trains the vectorizer model with the training data

        Args:
            trn_corpus (np.array): Training Data, sparse or dense array
            config (dict): Configurations of the vectorizer model
            dtype (np.float): Type of the data inside the array

        Return:
            TextVectorizer (Vectorizer): Trained Vectorizer
        """
        # Delegate training to Vectorizer class with given configuration
        return Vectorizer.train(trn_corpus, config, dtype)
    
    @staticmethod
    def _predict_vectorizer(text_vec, corpus):
        """Predicts the training data with the Vectorizer model

        Args:
            text_vec (Vectorizer): The Vectorizer Model
            corpus (np.array or sparse ?): The data array to be predicted

        Return:
            Transformed text embeddings (nd.array or scipy.sparse)
        """
        # Use trained vectorizer to transform input corpus
        return text_vec.predict(corpus)
    
    @staticmethod
    def _predict_transformer(trn_corpus, config, dtype=np.float32):
        """Predicts the training data with the transformer model

        Args:
            trn_corpus (np.array): Training Data, sparse or dense array
            config (dict): Configurations of the vectorizer model
            dtype (np.float): Type of the data inside the array

        Return:
            Transformed Embeddings (np.array): Predicted Embeddings
        """
        # Delegate training to Transformer class with given configuration
        return Transformer.train(trn_corpus, config, dtype)
    
    @staticmethod
    def _predict_proba_classifier(classifier_model, data_points):
        """
        Predicts class probabilities using a trained classifier.
        
        Args:
            classifier_model: Trained classifier model
            data_points: Input features to classify
            
        Returns:
            np.array: Class probabilities
        """
        return classifier_model.predict_proba(data_points)
    
    @staticmethod
    def _convert_predictions_into_csr(predictions, num_labels=None):
        """
        Converts ranked predictions into CSR sparse matrix.
        
        Args:
            predictions (list): List of (label_indices, scores) tuples where:
                            - label_indices: Array of top-k label indices
                            - scores: Array of corresponding reranker scores
            num_labels (int, optional): Total number of possible labels.
                                    If None, uses max index found + 1.
                                    
        Returns:
            csr_matrix: Sparse matrix of shape (n_instances, num_labels)
                    with reranker scores at top-k positions
        """
        if not predictions:
            return csr_matrix((0, num_labels or 0), dtype=np.float32)

        # Prepare CSR components
        rows, cols, vals = [], [], []
        
        for row_idx, (label_indices, scores) in enumerate(predictions):
            if len(label_indices) > 0:  # Handle empty predictions
                rows.extend([row_idx] * len(label_indices))
                cols.extend(label_indices)
                vals.extend(np.asarray(scores, dtype=np.float32))
        
        # Determine matrix shape
        if num_labels is None:
            num_labels = max(cols) + 1 if cols else 0
        
        return csr_matrix(
            (vals, (rows, cols)),
            shape=(len(predictions), num_labels),
            dtype=np.float32
        )
    
    @classmethod
    def _predict_inference(cls, htree, all_kb_ids, batch_conc_input, batch_labels, candidates=100):
        """
        Predict with guaranteed alignment between inputs and golden labels.
        
        Args:
            htree: Trained hierarchical tree
            all_kb_ids: List of all possible KB IDs
            batch_conc_input: Input embeddings (n_samples x n_features)
            batch_labels: Golden labels (n_samples) - each can be single ID or list
            candidates: Number of candidates to return
            
        Returns:
            tuple: (predictions, hit_indicators)
                predictions: List of (kb_indices, scores) tuples
                hit_indicators: List indicating if golden label was found (1/0)
        """
        n_samples = batch_conc_input.shape[0]
        results = [None] * n_samples
        hit_indicators = [0] * n_samples  # 1 if golden label found, else 0
        
        current_nodes = [htree] * n_samples
        active_indices = list(range(n_samples))
        
        while active_indices:
            node_groups = defaultdict(list)
            for idx in active_indices:
                node_groups[current_nodes[idx]].append(idx)
            
            new_active_indices = []
            for node, group_indices in node_groups.items():
                group_inputs = batch_conc_input[group_indices]
                
                # Batch predict probabilities
                probs = cls._predict_proba_classifier(node.tree_classifier, group_inputs)
                top_labels = np.argmax(probs, axis=1)
                
                for i, idx in enumerate(group_indices):
                    top_label = top_labels[i]
                    golden_label = batch_labels[idx]  # Get corresponding label(s)
                    
                    if top_label in node.children:
                        current_nodes[idx] = node.children[top_label]
                        new_active_indices.append(idx)
                    else:
                        # Get candidates at leaf node
                        mask = node.clustering_model.labels() == top_label
                        cand_kb_indices = np.array(node.kb_indices)[mask]
                        candidate_ids = np.array(all_kb_ids)[cand_kb_indices]
                        
                        # Check if golden label is in candidates (handles both str and list)
                        if isinstance(golden_label, list):
                            hit_indicators[idx] = int(any(label in candidate_ids for label in golden_label))
                        else:
                            hit_indicators[idx] = int(golden_label in candidate_ids)
                        
                        # Rerank candidates
                        input_emb = batch_conc_input[idx]
                        rerank_input = np.vstack([
                            np.hstack((input_emb, node.entity_centroids[eid]))
                            for eid in candidate_ids
                        ])
                        
                        scores = cls._predict_proba_classifier(node.reranker, rerank_input)[:, 1]
                        top_k_indices = np.argsort(scores)[-candidates:][::-1]
                        
                        results[idx] = (
                            cand_kb_indices[top_k_indices],
                            scores[top_k_indices]
                        )
            
            active_indices = new_active_indices
        
        print(hit_indicators)
        
        return results, hit_indicators
        
    @classmethod
    def inference(cls, htree, labels, input_text, k=5, dtype=np.float32):
        """
        End-to-end prediction pipeline for XMR system with multi-label support.
        
        Args:
            htree (XMRTree): Trained hierarchical tree model
            labels: List of lists of label IDs (e.g., [['D002294', 'D002583'], ['D000223'], ...])
            input_text: Input text(s) to predict
            k: Number of predictions to return per input
            dtype: Data type for embeddings
            
        Returns:
            tuple: (csr_matrix, list) where:
                - csr_matrix: Sparse prediction matrix
                - list: List of label hit ratios for each input
        """
        # LOGGER.info("Started inference")
        
        # 1. Generate embeddings
        # vec = htree.vectorizer
        # text_emb = cls._predict_vectorizer(vec, input_text)
        # svd = htree.dimension_model
        # dense_text_emb = svd.transform(text_emb)
        # dense_text_emb = normalize(dense_text_emb, norm='l2', axis=1)
        
        transformer_model = cls._predict_transformer(
            input_text, 
            htree.transformer_config, 
            dtype
        )
        transformer_emb = normalize(transformer_model.model.embeddings, norm="l2", axis=1)
        
        # concat_emb = np.hstack((
        #      transformer_emb.astype(dtype),
        #     dense_text_emb.astype(dtype)
        # ))

        gc.collect()

        concat_emb = transformer_emb

        # 2. Get predictions
        predictions, hit_ratios = cls._predict_inference(
            htree, 
            htree.labels, 
            concat_emb, 
            labels, 
            candidates=k*3  # Get more candidates for multi-label case
        )
        
        print(Counter(hit_ratios))
        
        # 3. Convert to CSR format
        return cls._convert_predictions_into_csr(predictions)

        