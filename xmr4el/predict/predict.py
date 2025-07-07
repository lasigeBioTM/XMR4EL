import gc
import os
import logging
import tempfile
from typing import Counter

import numpy as np

from scipy.sparse import csr_matrix

from sklearn.preprocessing import normalize
from sklearn.decomposition import TruncatedSVD

from sklearn.random_projection import SparseRandomProjection

from joblib import Parallel, delayed, load, dump
from tqdm import tqdm

from xmr4el.featurization.transformers import Transformer
from xmr4el.ranker.candidate_retrieval import CandidateRetrieval
from xmr4el.ranker.cross_encoder import CrossEncoderMP


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
        Converts prediction results into a sparse CSR matrix.
        
        Args:
            predictions (list): List of (label_indices, scores) tuples where:
                             - label_indices: Array of label indices
                             - scores: Array of corresponding scores
            num_labels (int, optional): Total number of possible labels.
                                      If None, inferred from data.
                                      
        Returns:
            csr_matrix: Sparse matrix of shape (n_instances, num_labels)
                       with prediction scores
        """
        rows, cols, vals = [], [], []

        for row_idx, (label_indices, scores) in enumerate(predictions):
            for col, score in zip(label_indices, scores):
                rows.append(row_idx)
                cols.append(col)
                vals.append(np.float32(score))

        if num_labels is None:
            num_labels = max(cols) + 1  # infer if not provided

        return csr_matrix((vals, (rows, cols)), shape=(len(predictions), num_labels), dtype=np.float32)
            
    @classmethod
    def _rank(cls, predictions, train_data, input_texts, candidates=100):
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
        cross_encoder = CrossEncoderMP()
        trn_corpus = list(train_data.values())
        
        LOGGER.info(f"Ranking {len(predictions)} predictions")

        # --- Phase 1: Candidate Retrieval ---
        LOGGER.info("First Stage Retrieval")
        indices_list = []
        
        all_scores = []
        all_indices = []
        
        for kb_indices, conc_input, conc_emb in predictions:
            # Ensure proper input shape for FAISS
            query = np.atleast_2d(conc_input)
            candidates_emb = np.atleast_2d(conc_emb)
            
            # Get candidates
            scores, indices = candidate_retrieval.retrival(query, candidates_emb, candidates)
            
            candidates_indices = min(50, len(indices))
            indices = indices[indices != -1].flatten()[:candidates_indices]
            
            indices_list.append(indices)
            
        
        # --- Phase 2: Cross-Encoder Scoring ---
        LOGGER.info("Second Stage Cross-Encoder")
        
        # Prepare all text pairs at once
        text_pairs = [
            (input_texts[i], ["[SEP]".join(trn_corpus[int(idx)]) for idx in indices])
            for i, indices in enumerate(indices_list)
        ]
        
        # Get all matches at once
        matches = cross_encoder.predict(text_pairs)
        
        # Format final results
        results = []
        for (kb_indices, _, _), (match_indices, match_scores) in zip(predictions, matches):
            label_ids = [kb_indices[i] for i in match_indices]
            results.append((label_ids, list(match_scores)))
        
        return results
           
    @classmethod
    def _predict_input(cls, htree, conc_input):
        """
        Recursively traverses hierarchical tree to find relevant candidates.
        
        Args:
            htree (XMRTree): Current hierarchical tree node
            conc_input (np.ndarray): Concatenated input embeddings
            candidates (int): Number of candidates to retrieve. Defaults to 100.
            
        Returns:
            tuple: (kb_indices, conc_input, conc_emb) where:
                 - kb_indices: Knowledge base indices
                 - conc_input: Original input embeddings
                 - conc_emb: Candidate embeddings for ranking
        """
        current_htree = htree

        while True:
            # Get classifier and number of possible labels at current level
            classifier = current_htree.classifier_model
            classifier.model.model.n_jobs = -1
            n_labels = len(current_htree.clustering_model.labels())
            
            probs = cls._predict_proba_classifier(classifier, conc_input)[0]
            
            top_label = np.argmax(probs)

            if top_label in current_htree.children:
                # Continue down the tree hierarchy
                current_htree = current_htree.children[top_label]
                continue
                
            else: 
                # Reached leaf node - perform reranking
                mask = current_htree.clustering_model.labels() == top_label
                conc_emb = current_htree.concatenated_embeddings[mask]
                kb_indices = np.array(current_htree.kb_indices)[mask]
                
                return (kb_indices, conc_input, conc_emb)
    
    @classmethod
    def _predict(cls, htree, batch_conc_input, candidates=100):
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
                        mask = node.clustering_model.labels() == top_label
                        conc_emb = node.concatenated_embeddings[mask]
                        kb_indices = np.array(node.kb_indices)[mask]
                        
                        # Ensure we don't return more than candidates
                        if len(kb_indices) > candidates:
                            random_indices = np.random.choice(
                                len(kb_indices), 
                                candidates, 
                                replace=False
                            )
                            kb_indices = kb_indices[random_indices]
                            conc_emb = conc_emb[random_indices]
                        
                        results[idx] = (kb_indices, batch_conc_input[idx], conc_emb)
            
            active_indices = new_active_indices
        
        return results
    
    @classmethod
    def _predict_batch_memopt(cls, htree, batch_conc_input, candidates=100, batch_size=1000):
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
            chunk_results = cls._predict(htree, chunk, candidates)
            results.extend(chunk_results)
            
            # Explicit cleanup if needed
            if i % (10 * batch_size) == 0:
                gc.collect()
        
        return results
                
    @classmethod
    def inference(
        cls, htree, input_text, transformer_config, k=3, dtype=np.float32
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
        text_emb = cls._predict_vectorizer(htree.vectorizer, input_text)
        
        # print(htree.transformer_embeddings, type(htree.transformer_embeddings), htree.transformer_embeddings.shape)
        
        # Problemas de features, pq logistic regression é treinado com muitos mais features, inventar features ?     
        n_features = htree.text_features
        
        LOGGER.info(f"Truncating text_embeddings to {n_features} n features")
        # svd = TruncatedSVD(n_components=n_features, random_state=0)
        # dense_text_emb = svd.fit_transform(text_emb) # turns it into dense auto
        
        rp = SparseRandomProjection(n_components=n_features, random_state=42)
        dense_text_emb = rp.fit_transform(text_emb).toarray()
        
        # print(dense_text_emb.shape)

        # Normalize text embeddings (handling sparse)
        dense_text_emb = normalize(dense_text_emb, norm='l2', axis=1)
        
        # Step 2: Generate transformer embeddings with memory management
        transformer_model = cls._predict_transformer(
            input_text, 
            transformer_config, 
            dtype
        )
        
        transformer_emb = transformer_model.model.embeddings
        
        # Step 3: Ensure dimensional compatibility with training data
        transformer_n_features = htree.transformer_embeddings.shape[1]
        LOGGER.info(f"Truncating transformer embeddings to {transformer_n_features} n features")
        if transformer_emb.shape[1] != transformer_n_features:
            transformer_emb = cls._reduce_dimensionality(
                transformer_emb, 
                transformer_n_features
            )

        transformer_emb = normalize(transformer_emb, norm="l2", axis=1)
        
        concat_emb = np.hstack((
            transformer_emb.astype(dtype),
            dense_text_emb.astype(dtype)
        ))

        del transformer_emb, dense_text_emb, rp
        gc.collect()

        predictions = cls._predict_batch_memopt(htree, concat_emb)
        
        results = cls._rank(predictions, htree.train_data, input_text, candidates=100)

        return cls._convert_predictions_into_csr(results)