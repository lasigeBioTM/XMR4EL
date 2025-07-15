import gc
import logging
import random

import numpy as np

from scipy.sparse import csr_matrix

from collections import defaultdict

from sklearn.preprocessing import normalize
from sklearn.decomposition import TruncatedSVD

from sklearn.random_projection import SparseRandomProjection

from xmr4el.featurization.transformers import Transformer
from xmr4el.featurization.vectorizers import Vectorizer
from xmr4el.ranker.candidate_retrieval import CandidateRetrieval
from xmr4el.ranker.cross_encoder import CrossEncoder


LOGGER = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


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
            LOGGER.info("Maintaining original number of features," 
                        f"impossible to reduce, min({n_samples}, {n_features})={n_features_emb}")
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
    def _rank(cls, predictions, train_data, input_texts, encoder_config, candidates=100, k=5):
        """
        Simplified two-stage ranking pipeline without batching
        
        Args:
            predictions (list): List of (kb_indices, conc_input, conc_emb) tuples
            train_data (dict): {index: text} mapping
            input_texts (list): Original input texts
            candidates (int): Number of candidates to retrieve
            config (dict, optional): Configuration parameters
            
        Returns:
            list: Ranked predictions as (label_ids, scores) tuples
        """
        candidate_retrieval = CandidateRetrieval()
        cross_encoder = CrossEncoder(encoder_config)
        trn_corpus = list(train_data.values())

        LOGGER.info(f"Ranking {len(predictions)} predictions")

        # --- Phase 1: Candidate Retrieval ---
        LOGGER.info("First Stage Retrieval")
        indices_list = []

        for kb_indices, conc_input, conc_emb in predictions:
            # Ensure proper input shape for FAISS
            query = np.atleast_2d(conc_input)
            candidates_emb = np.atleast_2d(conc_emb)

            # Get candidates
            scores, indices = candidate_retrieval.retrival(query, candidates_emb, candidates)

            # 30 candidates,
            candidates_indices = min(k, len(indices))
            indices = indices[indices != -1].flatten()[:candidates_indices]

            indices_list.append(indices)


        # --- Phase 2: Cross-Encoder Scoring ---
        LOGGER.info("Second Stage Cross-Encoder")

        text_pairs = []
        entity_indices = []  # Tracks entity index for each pair (no alias details)

        for i, indices in enumerate(indices_list):
            query = input_texts[i]
            for idx in indices:
                aliases = trn_corpus[int(idx)]
                text_pairs.extend([(query, alias) for alias in aliases])
                entity_indices.extend([idx] * len(aliases))  # Repeat entity index

        # Batch scoring
        entity_scores_dict, _ = cross_encoder.predict(text_pairs, entity_indices)

        # Recombine scores per query
        results = []
        for i, (kb_indices, _, _) in enumerate(predictions):
            # Get scores for this query's candidates
            query_entity_scores = {
                idx: entity_scores_dict[idx] 
                for idx in indices_list[i] 
                if idx in entity_scores_dict
            }

            # Sort entities by score (descending)
            sorted_entities = sorted(
                query_entity_scores.items(),
                key=lambda x: -x[1]
            )

            # Map to final labels
            label_ids = [kb_indices[int(idx)] for idx, _ in sorted_entities]
            scores = [score for _, score in sorted_entities]

            results.append((label_ids, scores))

        return results 
    
    @classmethod
    def _predict(cls, htree, all_kb_ids, batch_conc_input, candidates=100):
        """ 
        Batch version of _predict_input that processes multiple inputs simultaneously.
        
        Args:
            htree (XMRTree): Root of hierarchical tree
            batch_conc_input (np.ndarray): Batch of concatenated input embeddings (n_samples x n_features)
            candidates (int): Number of candidates to retrieve per input
            
        Returns:
            list: List of tuples (kb_indices, conc_input, conc_emb) for each input
        """
        # Initialize results and current nodes for all inputs
        n_samples = batch_conc_input.shape[0]
        results = [None] * n_samples
        current_nodes = [htree] * n_samples
        active_indices = list(range(n_samples))  # Track which inputs still need processing
        
        while active_indices:
            # Group inputs by their current node for batch processing
            node_groups = {}
            for idx in active_indices:
                node = current_nodes[idx]
                if node not in node_groups:
                    node_groups[node] = []
                node_groups[node].append(idx)
            
            # Process each node group in batch
            new_active_indices = []
            for node, group_indices in node_groups.items():
                group_inputs = batch_conc_input[group_indices]
                # Batch predict probabilities
                classifier = node.classifier_model
                classifier.model.model.n_jobs = -1
                probs = cls._predict_proba_classifier(classifier, group_inputs)
                
                # Get top labels for each input
                top_labels = np.argmax(probs, axis=1)
                
                # Process each input in the group
                for i, idx in enumerate(group_indices):
                    top_label = top_labels[i]
                    
                    if top_label in node.children:
                        # Move to child node for next iteration
                        current_nodes[idx] = node.children[top_label]
                        new_active_indices.append(idx)
                    else:
                        # Reached leaf node - store results
                        entity_centroids = node.entity_centroids
                        mask = node.clustering_model.labels() == top_label
                        # cand_embs = node.concatenated_embeddings[mask]
                        cand_kb_indices = np.array(node.kb_indices)[mask]
                        unique_kb_ids = np.array(all_kb_ids)[cand_kb_indices]
                        
                        mention_emb = batch_conc_input[idx]
                        
                        rerank_X = np.vstack([
                            np.hstack((mention_emb, entity_centroids[eid]))
                            for eid in unique_kb_ids
                        ])
                        
                        # print(rerank_X)
                        
                        scores = cls._predict_proba_classifier(node.reranker, rerank_X)[:, 1] # Get positive class scores

                        top_k_indices = np.argsort(scores)[-candidates:][::-1]
                        
                        results[idx] = (
                            cand_kb_indices[top_k_indices],
                            scores[top_k_indices]
                        )
            
            active_indices = new_active_indices
        
        return results
    
    @classmethod
    def _predict_inference(cls, htree, all_kb_ids, batch_conc_input, batch_labels, candidates=100):
        """
        Modified to handle multiple labels per input.
        """
        n_samples = batch_conc_input.shape[0]
        results = [None] * n_samples
        current_nodes = [htree] * n_samples
        active_indices = list(range(n_samples))
        
        # Track hits for each label of each input
        label_hits = [defaultdict(int) for _ in range(n_samples)]
        label_counts = [defaultdict(int) for _ in range(n_samples)]
        
        while active_indices:
            node_groups = defaultdict(list)
            for idx in active_indices:
                node_groups[current_nodes[idx]].append(idx)
            
            new_active_indices = []
            for node, group_indices in node_groups.items():
                group_inputs = batch_conc_input[group_indices]
                
                classifier = node.tree_classifier
                classifier.model.model.n_jobs = -1
                probs = cls._predict_proba_classifier(classifier, group_inputs)
                top_labels = np.argmax(probs, axis=1)
                
                for i, idx in enumerate(group_indices):
                    top_label = top_labels[i]
                    current_labels = batch_labels[idx]  # List of labels for this input
                    
                    if top_label in node.children:
                        current_nodes[idx] = node.children[top_label]
                        new_active_indices.append(idx)
                    else:
                        mask = node.clustering_model.labels() == top_label
                        cand_kb_indices = np.array(node.kb_indices)[mask]
                        unique_kb_ids = np.array(all_kb_ids)[cand_kb_indices]
                        
                        # Check each label for this input
                        for label in current_labels:
                            label_hits[idx][label] += int(label in unique_kb_ids)
                            label_counts[idx][label] += 1
                        
                        # Reranking
                        mention_emb = batch_conc_input[idx]
                        rerank_X = np.vstack([
                            np.hstack((mention_emb, node.entity_centroids[eid]))
                            for eid in unique_kb_ids
                        ])
                        
                        scores = cls._predict_proba_classifier(node.reranker, rerank_X)[:, 1]
                        top_k_indices = np.argsort(scores)[-candidates:][::-1]
                        
                        results[idx] = (
                            cand_kb_indices[top_k_indices],
                            scores[top_k_indices]
                        )
            
            active_indices = new_active_indices
        
        # Calculate hit ratios per input (average across all its labels)
        hit_ratios = []
        for i in range(n_samples):
            if not label_counts[i]:
                hit_ratios.append(0.0)
            else:
                total_hits = sum(label_hits[i].values())
                total_checks = sum(label_counts[i].values())
                hit_ratios.append(total_hits / total_checks)
        
        return results, label_hits, hit_ratios
    
    @classmethod
    def _predict_batch_memopt(cls, htree, all_kb_ids, batch_conc_input, candidates=100, batch_size=640000):
        """
        Memory-optimized batch prediction that processes inputs in chunks.
        
        Args:
            htree (XMRTree): Root of hierarchical tree
            batch_conc_input (np.ndarray): Batch of concatenated input embeddings
            candidates (int): Number of candidates to retrieve per input
            batch_size (int): Number of inputs to process at once
            
        Returns:
            list: List of tuples (kb_indices, conc_input, conc_emb) for each input
        """
        n_samples = batch_conc_input.shape[0]
        results = []
        
        
        
        # Process in chunks to control memory usage
        for i in range(0, n_samples, batch_size):
            chunk = batch_conc_input[i:i+batch_size]
            chunk_results = cls._predict(htree, all_kb_ids, chunk, candidates)
            results.extend(chunk_results)
            
            # Explicit cleanup if needed
            if i % (10 * batch_size) == 0:
                gc.collect()
        
        return results
    
    @classmethod
    def _predict_batch_memopt_inference(cls, htree, all_kb_ids, batch_conc_input, batch_labels, candidates=100, batch_size=640000):
        """
        Memory-optimized batch prediction with multi-label support.
        
        Args:
            batch_labels: List of lists of label IDs
            ... other params same as before ...
            
        Returns:
            tuple: (predictions, hit_ratios)
        """
        n_samples = batch_conc_input.shape[0]
        all_predictions = []
        all_hit_ratios = []
        
        for i in range(0, n_samples, batch_size):
            chunk = batch_conc_input[i:i+batch_size]
            chunk_labels = batch_labels[i:i+batch_size]
            
            # Process each input's labels separately
            chunk_predictions = []
            chunk_hit_ratios = []
            
            for j in range(len(chunk)):
                # For each input, check all its labels
                input_labels = chunk_labels[j]
                input_emb = chunk[j:j+1]  # Keep 2D shape
                
                # Get predictions for this single input
                predictions, _, hit_ratio = cls._predict_inference(
                    htree,
                    all_kb_ids,
                    input_emb,
                    input_labels,  # Pass all labels for this input
                    candidates
                )
                
                chunk_predictions.extend(predictions)
                chunk_hit_ratios.append(hit_ratio)
            
            all_predictions.extend(chunk_predictions)
            all_hit_ratios.extend(chunk_hit_ratios)
            
            if i % (10 * batch_size) == 0:
                gc.collect()
        
        return all_predictions, all_hit_ratios
                
    @classmethod
    def predict(
        cls, htree, input_text, k=3, dtype=np.float32
    ):
        """
        End-to-end prediction pipeline for XMR system.
        
        Args:
            htree (XMRTree): Trained hierarchical tree model
            input_text (iterable): Input text(s) to predict
            transformer_config (dict): Transformer configuration
            k (int): Number of predictions to return per input. Defaults to 3.
            dtype (np.dtype): Data type for embeddings. Defaults to np.float32.
            
        Returns:
            csr_matrix: Sparse prediction matrix of shape (n_inputs * n_labels)
        """
        LOGGER.info(f"Started inference")
        # Step 1: Generate text embeddings using stored vectorizer
        vec = htree.vectorizer
        text_emb = cls._predict_vectorizer(vec, input_text)
        
        
        LOGGER.info(f"Truncating text_embedding")
        svd = htree.dimension_model
        dense_text_emb = svd.transform(text_emb) # turns it into dense auto

        # Normalize text embeddings (handling sparse)
        dense_text_emb = normalize(dense_text_emb, norm='l2', axis=1)
        
        # Step 2: Generate transformer embeddings with memory management
        transformer_model = cls._predict_transformer(
            input_text, 
            htree.transformer_config, 
            dtype
        )
        
        transformer_emb = transformer_model.model.embeddings

        transformer_emb = normalize(transformer_emb, norm="l2", axis=1)
        
        concat_emb = np.hstack((
            transformer_emb.astype(dtype),
            dense_text_emb.astype(dtype)
        ))

        del transformer_emb, dense_text_emb
        gc.collect()

        predictions = cls._predict_batch_memopt(htree, htree.labels, concat_emb, candidates=10)
        
        return cls._convert_predictions_into_csr(predictions)
        
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
        LOGGER.info("Started inference")
        
        # 1. Generate embeddings
        vec = htree.vectorizer
        text_emb = cls._predict_vectorizer(vec, input_text)
        svd = htree.dimension_model
        dense_text_emb = svd.transform(text_emb)
        dense_text_emb = normalize(dense_text_emb, norm='l2', axis=1)
        
        transformer_model = cls._predict_transformer(
            input_text, 
            htree.transformer_config, 
            dtype
        )
        transformer_emb = normalize(transformer_model.model.embeddings, norm="l2", axis=1)
        
        concat_emb = np.hstack((
            transformer_emb.astype(dtype),
            dense_text_emb.astype(dtype)
        ))

        del transformer_emb, dense_text_emb
        gc.collect()

        # 2. Get predictions
        predictions, hit_ratios = cls._predict_batch_memopt_inference(
            htree, 
            htree.labels, 
            concat_emb, 
            labels, 
            candidates=k*3  # Get more candidates for multi-label case
        )
        
        # 3. Convert to CSR format
        return cls._convert_predictions_into_csr(predictions), hit_ratios

        