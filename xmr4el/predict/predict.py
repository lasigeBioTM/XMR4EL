import logging
import torch
import gc

import numpy as np

from scipy.sparse import issparse

from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA

from joblib import Parallel, delayed
from tqdm import tqdm

from xmr4el.featurization.transformers import Transformer
from xmr4el.ranker.candidate_retrieval import CandidateRetrieval
from xmr4el.ranker.cross_enconder import CrossEncoderMP
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
    def __convert_predictions_into_csr(data, num_labels=None):
        """
        Converts prediction results into sparse matrix format.
        
        Args:
            data: List of predictions in format [(label_index, score), ...]
            num_labels: Total number of possible labels
            
        Returns:
            csr_matrix: Sparse matrix of predictions
        """
        return ReRanker.convert_predictions_into_csr(data, num_labels)
            
    @classmethod
    def _rank(cls, predictions, train_data, input_texts, candidates=100, config=None):
        """Optimized parallel candidate retrieval and ranking.
        
        Args:
            predictions: List of tuples (_, conc_input, conc_emb).
            train_data: Dict {idx: text} for training corpus.
            input_texts: List of input texts to match against.
            candidates: Top-k candidates to retrieve per query.
            config: Optional config dict (e.g., for n_jobs).
        
        Returns:
            List of ranked matches.
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
            scores, indices = candidate_retrieval.retrival(conc_input, conc_emb, candidates)
            print(f"Indices: {indices}")
            return indices[indices != -1].flatten()
        
        # Parallelize retrieval (FAISS is single-threaded, but we batch queries)
        with Parallel(n_jobs=n_jobs, backend="threading") as parallel:
            indices_list = parallel(delayed(_retrieve)(pred) for pred in predictions)
        
        LOGGER.info(f"Second Stage Cross-Enconder")
        
        # --- Phase 2: Batch Cross-Encoder Scoring ---
        # Prepare all (input_text, candidate_text) pairs
        text_pairs = [
            (input_texts[i], [trn_corpus[int(idx)] for idx in indices[:5]])
            for i, indices in enumerate(indices_list)
        ]
        
        """
        # Smart concatenation (e.g., join with separator)
        trn_corpus_hybrid = {
            0: "[SEP]".join(["text1_v1", "text1_v2", ...]),  # [SEP] helps models distinguish
        }
        """
        
        # Batch predict (cross_encoder should handle parallelism internally)
        matches = cross_encoder.predict(text_pairs)
        
        return matches
           
    @classmethod
    def _predict_input(cls, htree, conc_input, candidates=100):
        """
        Recursively traverses the hierarchical tree to make predictions.
        
        Args:
            htree (XMRTree): Current tree node
            conc_input: Concatenated input embeddings
            k: Number of predictions to return
            
        Returns:
            list: Predicted (kb_index, score) pairs
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
        text_emb = cls._predict_vectorizer(htree.vectorizer, input_text)

        # Normalize text embeddings (handling both sparse and dense cases)
        if hasattr(text_emb, 'data'):
            text_emb.data = normalize(text_emb.data.reshape(1, -1)).ravel()
        else: # Dense matrix
            text_emb = normalize(text_emb, norm='l2', axis=1)
        
        # Step 2: Generate transformer embeddings with memory management
        transformer_model = cls._predict_transformer(
            input_text, 
            transformer_config, 
            dtype
        )
        transformer_emb = transformer_model.model.embeddings
        
        # Step 3: Ensure dimensional compatibility with training data
        transfomer_n_features = htree.transformer_embeddings.shape[1]
        if transformer_emb.shape[1] != transfomer_n_features:
            transformer_emb = cls._reduce_dimensionality(
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
        
        # transformer_emb = transformer_emb.astype(dtype)
        
        # trans_emb = [emb.reshape(1, -1) for emb in transformer_emb]
        
        def task(emb):
            kb_indices, conc_input, conc_emb = cls._predict_input(htree, emb, k) # didnt add k
            return kb_indices, conc_input, conc_emb

        # Use threads instead of processes
        predictions = Parallel(n_jobs=-1, prefer="threads", batch_size=1)(
            delayed(task)(emb) for emb in tqdm(concat_emb) # trans_emb
        )
        
        gc.collect()
        
        cls._rank(predictions, htree.train_data, input_text, candidates=50)

        return cls.__convert_predictions_into_csr(predictions)