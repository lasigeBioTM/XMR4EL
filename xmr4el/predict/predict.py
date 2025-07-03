import gc
import os
import logging
import tempfile

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
    def _rank(cls, predictions, train_data, input_texts, candidates=100, config=None):
        """
        Two-stage ranking pipeline:
        1. Fast candidate retrieval using FAISS
        2. Precise reranking using CrossEncoder
        
        Args:
            predictions (list): List of (kb_indices, conc_input, conc_emb) tuples
            train_data (dict): {index: text} mapping of training corpus
            input_texts (list): Original input texts for cross-encoder
            candidates (int): Number of candidates to retrieve. Defaults to 100.
            config (dict, optional): Configuration including:
                                   - n_jobs: Number of parallel jobs
                                   
        Returns:
            list: Ranked predictions as (label_ids, scores) tuples
        """
        # Initialize components (once)
        candidate_retrieval = CandidateRetrieval()
        cross_encoder = CrossEncoderMP()
        trn_corpus = list(train_data.values())  # Precompute corpus
        
        # Use all cores unless specified
        n_jobs = config.get("n_jobs", -1) if config else -1
        
        LOGGER.info(f"Ranking {len(predictions)} predictions with {n_jobs} cores")
        
        LOGGER.info(f"First Stage Retrieval")
        # --- Phase 1: Parallel Candidate Retrieval ---
        def _retrieve(pred):
            conc_input, conc_emb = pred[1], pred[2]
            _, indices = candidate_retrieval.retrival(conc_input, conc_emb, candidates) # scores
            # print(f"Indices: {indices}")
            return indices[indices != -1].flatten()
        
        # Parallelize retrieval (FAISS is single-threaded, but we batch queries)
        with Parallel(n_jobs=n_jobs, backend="threading") as parallel:
            indices_list = parallel(delayed(_retrieve)(pred) for pred in predictions)
        
        LOGGER.info(f"Second Stage Cross-Enconder")
        
        # --- Phase 2: Batch Cross-Encoder Scoring ---
        # Prepare all (input_text, candidate_text) pairs
        text_pairs = [
            (input_texts[i], ["[SEP]".join(trn_corpus[int(idx)]) for idx in indices])
            for i, indices in enumerate(indices_list)
        ]
        
        # Batch predict (cross_encoder should handle parallelism internally)
        matches = cross_encoder.predict(text_pairs)
        
        # matches = top_k_indices, scores
        
        results = []
        for (kb_indices, _, _), (match_indices, match_scores) in zip(predictions, matches):
            label_ids = [kb_indices[i] for i in match_indices]
            scores = list(match_scores)
            results.append((label_ids, scores))
        
        return results
           
    @classmethod
    def _predict_input(cls, htree, conc_input, candidates=100):
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
        # device = 'cuda' if torch.cuda.is_available() else 'cpu'
        current_htree = htree

        while True:
            # Get classifier and number of possible labels at current level
            classifier = current_htree.classifier_model
            classifier.model.model.n_jobs = 1
            n_labels = len(current_htree.clustering_model.labels())
            candidates = min(candidates, n_labels) # Ensure k doesn't exceed available labels
            
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

        concat_emb = [emb.reshape(1, -1) for emb in concat_emb]
        
        LOGGER.info(f"Starting main predict task")
        
        # Before parallel section:
        with tempfile.NamedTemporaryFile(delete=False) as f:
            dump(htree, f.name)
            htree_path = f.name
        
        def task(emb, model_path):
            htree = load(model_path)
            kb_indices, conc_input, conc_emb = cls._predict_input(htree, emb, k)
            del htree
            return kb_indices, conc_input, conc_emb

        del transformer_emb
        del dense_text_emb
        del rp
        
        gc.collect()

        try:
            # Use threads instead of processes
            predictions = Parallel(n_jobs=min(8, os.cpu_count()), prefer="processes", batch_size=20)(
                delayed(task)(emb, htree_path) for emb in tqdm(concat_emb)
            )
        finally:
            os.unlink(htree_path)
        
        gc.collect()
        
        results = cls._rank(predictions, htree.train_data, input_text, candidates=200)

        return cls._convert_predictions_into_csr(results)