import os
import logging
import torch
import gc

import numpy as np

from scipy.sparse import csr_matrix, issparse
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from joblib import Parallel, delayed
from tqdm import tqdm


from xmr4el.featurization.transformers import Transformer
from xmr4el.ranker.reranker import ReRanker


LOGGER = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class Predict():
    """
    Prediction class for Extreme Multi-label Ranking (XMR) system that handles:
    - Feature transformation and dimensionality reduction
    - Hierarchical tree traversal for prediction
    - Reranking of candidate labels
    - Efficient batch prediction
    """
    
    @staticmethod
    def __reduce_dimensionality(emb, n_features, random_state=0):
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
        # PCA cannot have more components than min(n_samples, n_features)
        max_possible = min(n_samples-1, n_features_emb)  # PCA limitation
        effective_dim = min(n_features, max_possible)
        
        if effective_dim <= 0:
            LOGGER.info("Maintaining original number of features," 
                        f"impossible to reduce, min({n_samples}, {n_features})={max_possible}")
            return emb  # Return original if reduction not possible
        
        # Perform PCA with effective dimension
        pca = PCA(n_components=effective_dim, random_state=random_state)
        return pca.fit_transform(emb)
    
    @staticmethod
    def __predict_vectorizer(text_vec, corpus):
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
    def __predict_transformer(trn_corpus, config, dtype=np.float32):
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
    def __predict_proba_classifier(classifier_model, data_points):
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
    def __reranker(input_vec, labels_vec, k=5, candidates=100):
        """
        Reranks candidate labels based on similarity to input.
        
        Args:
            input_vec: Input embedding to match against
            labels_vec: Candidate label embeddings
            k: Number of top matches to return
            candidates: Number of candidates to consider
            
        Returns:
            tuple: (indices of top matches, similarity scores)
        """
        rr = ReRanker(
            embed_dim=labels_vec.shape[1],
            hidden_dim=128, 
            batch_size=400,
            alpha=0
            )

        top_indices, top_scores = rr.match(
            input_vec=input_vec,
            label_vecs=labels_vec,
            top_k=k,
            candidates=candidates,
        )
        
        return top_indices, top_scores
    
    @staticmethod
    def __convert_predictions_into_csr(data, num_labels=None):
        """
        Converts prediction results into sparse matrix format.
        
        Args:
            data: List of predictions in format [(label_index, score), ...]
            num_labels: Total number of possible labels
            
        Returns:
            csr_matrix: Sparse matrix of predictions
        """
        rows, cols, vals = [], [], []

        for row_idx, instance in enumerate(data):
            for col_idx, score in instance:
                rows.append(row_idx)
                cols.append(col_idx)
                vals.append(np.float32(score))

        if num_labels is None:
            num_labels = max(cols) + 1  # infer if not provided
        
        return csr_matrix((vals, (rows, cols)), 
                          shape=(len(data), num_labels), 
                          dtype=np.float32)
    
    @classmethod
    def _rank_indices(cls, kb_indices, conc_input, conc_emb, k=10):
        # Get top-k matches from candidates in this cluster
        indices, scores = cls.__reranker(
            conc_input, 
            conc_emb, 
            k=k, 
            candidates=min(100, len(conc_emb))
        )
                
        return [
            (kb_indices[idx], float(score))
            for idx, score in zip(indices, scores)
        ]    
    
    @classmethod
    def _predict_input(cls, htree, conc_input, k=10):
        """
        Recursively traverses the hierarchical tree to make predictions.
        
        Args:
            htree (XMRTree): Current tree node
            conc_input: Concatenated input embeddings
            k: Number of predictions to return
            
        Returns:
            list: Predicted (kb_index, score) pairs
        """
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        current_htree = htree
        input_tensor = torch.FloatTensor(conc_input).to(device)

        while True:
            # Get classifier and number of possible labels at current level
            classifier = current_htree.classifier_model
            classifier.model.model.n_jobs = 1
            n_labels = len(current_htree.clustering_model.labels())
            k = min(k, n_labels) # Ensure k doesn't exceed available labels
            
            # Get class probabilities
            with torch.no_grad():
                if torch.is_tensor(input_tensor):
                    input_cpu = input_tensor.cpu().numpy()
                else:
                    input_cpu = input_tensor
                probs = cls.__predict_proba_classifier(classifier, input_cpu)[0]
            
            top_label = np.argmax(probs)

            if top_label in current_htree.children:
                # Continue down the tree hierarchy
                current_htree = current_htree.children[top_label]
                
            else: 
                # Reached leaf node - perform reranking
                mask = current_htree.clustering_model.labels() == top_label
                conc_emb = current_htree.concatenated_embeddings[mask]
                
                return (current_htree.kb_indices, conc_input, conc_emb)
            
    @classmethod
    def inference(
        cls, htree, input_text, transformer_config, k=3, dtype=np.float32, n_workers=None
    ):
        """
        Main prediction pipeline for XMR system.
        
        Args:
            htree (XMRTree): Trained hierarchical tree model
            input_text: Input text(s) to predict
            transformer_config: Configuration for transformer model
            k: Number of predictions to return per input
            dtype: Data type for embeddings
            
        Returns:
            csr_matrix: Sparse matrix of predictions (n_inputs * n_labels)
        """
        
        # Step 1: Generate text embeddings using stored vectorizer
        text_emb = cls.__predict_vectorizer(htree.vectorizer, input_text)

        # Normalize text embeddings (handling both sparse and dense cases)
        if hasattr(text_emb, 'data'):
            text_emb.data = normalize(text_emb.data.reshape(1, -1)).ravel()
        else: # Dense matrix
            text_emb = normalize(text_emb, norm='l2', axis=1)
        
        # Step 2: Generate transformer embeddings with memory management
        transformer_model = cls.__predict_transformer(
            input_text, 
            transformer_config, 
            dtype
        )
        transformer_emb = transformer_model.model.embeddings
        
        # Step 3: Ensure dimensional compatibility with training data
        transfomer_n_features = htree.transformer_embeddings.shape[1]
        if transformer_emb.shape[1] != transfomer_n_features:
            transformer_emb = cls.__reduce_dimensionality(
                transformer_emb, 
                transfomer_n_features
            )

        transformer_emb = normalize(transformer_emb, norm="l2", axis=1)

        # Step 4: Concatenate features efficiently
        if issparse(text_emb):
            text_emb = text_emb.toarray()
            
        concat_emb = np.hstack((
            transformer_emb.astype(dtype),
            text_emb.astype(dtype)
        ))

        concat_emb = [emb.reshape(1, -1) for emb in concat_emb]
        
        transformer_emb = [emb.reshape(1, -1) for emb in transformer_emb]

        def task(emb):
            kb_indices, conc_input, conc_emb = cls._predict_input(htree, emb, k)
            return cls._rank_indices(kb_indices, conc_input, conc_emb)

        # Use threads instead of processes
        predictions = Parallel(n_jobs=-1, prefer="threads", batch_size=1)(
            delayed(task)(emb) for emb in tqdm(transformer_emb) # transformer_emb.astype(dtype), concat_emb
        )
        
        gc.collect()
        
        return cls.__convert_predictions_into_csr(predictions)