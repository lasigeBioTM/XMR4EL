import gc


import numpy as np

from scipy.sparse import csr_matrix

from collections import Counter, defaultdict

from sklearn.preprocessing import normalize
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

from xmr4el.featurization.transformers import Transformer
from xmr4el.featurization.vectorizers import Vectorizer
from xmr4el.ranker.candidate_retrieval import CandidateRetrieval
from xmr4el.ranker.cross_encoder import CrossEncoder


class SkeletonPredict():
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
       
    def __init__(self, htree, all_kb_ids):
        self.htree = htree
        self.all_kb_ids = all_kb_ids
        
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

    def _batch_predict_proba_classifier(self, classifiers, input_embs):
            """
            Batch predict probabilities from multiple classifiers.
            
            Args:
                classifiers: List of classifiers (one per input)
                input_embs: Batch of input embeddings
                
            Returns:
                List of probability arrays
            """
            # Group by unique classifiers to avoid redundant predictions
            unique_classifiers = {}
            for i, clf in enumerate(classifiers):
                clf_id = id(clf)
                if clf_id not in unique_classifiers:
                    unique_classifiers[clf_id] = {'clf': clf, 'indices': []}
                unique_classifiers[clf_id]['indices'].append(i)
            
            # Initialize results
            all_probs = [None] * len(classifiers)
            
            # Process each unique classifier
            for clf_info in unique_classifiers.values():
                if not clf_info['indices']:
                    continue
                    
                batch_embs = input_embs[clf_info['indices']]
                try:
                    batch_probs = clf_info['clf'].predict_proba(batch_embs)
                except AttributeError:
                    # Handle case where classifier is not properly initialized
                    batch_probs = np.ones((len(batch_embs), 1))  # Dummy probabilities
                    
                for idx, prob in zip(clf_info['indices'], batch_probs):
                    all_probs[idx] = prob
            
            return all_probs
        
    def vectorize_input_text(self, input_text):
        # 1. Generate embeddings
        vec = self.htree.vectorizer
        text_emb = self._predict_vectorizer(vec, input_text)
        svd = self.htree.dimension_model
        dense_text_emb = svd.transform(text_emb)
        dense_text_emb = normalize(dense_text_emb, norm='l2', axis=1)
        return dense_text_emb
    
    def transform_input_text(self, input_text):
        transformer_model = self._predict_transformer(
            input_text, 
            self.htree.transformer_config, 
        )
        transformer_emb = normalize(transformer_model.model.embeddings, norm="l2", axis=1)

        return transformer_emb
    
    def generate_input_embeddigns(self, input_text):
        text_emb = self.vectorize_input_text(input_text)
        transformer_emb = self.transform_input_text(input_text)
        
        concat_emb = np.hstack((
             transformer_emb,
             text_emb
        ))
        
        concat_emb = normalize(concat_emb, norm="l2", axis=1)
        
        return concat_emb

    def predict(self, input_embs, k=5, beam_size=3, batch_size=4000):
        """
        Optimized batch prediction with reduced checks and vectorized operations.
        
        Args:
            input_embs: 2D numpy array of input embeddings
            k: Number of top results to return
            beam_size: Beam search width
            batch_size: Processing batch size
            
        Returns:
            List of top-k (entity_id, score) tuples for each input
        """
        all_results = []
        
        # Process in batches
        for batch_start in range(0, len(input_embs), batch_size):
            batch_end = min(batch_start + batch_size, len(input_embs))
            current_batch = input_embs[batch_start:batch_end]
            batch_size = len(current_batch)
            
            # Initialize beams - (active, completed)
            beams = [([(self.htree, 1.0)], []) for _ in range(batch_size)]
            
            # ---- Optimized Beam Search ----
            while True:
                # Find samples that still have active beams
                active_mask = [bool(b[0]) for b in beams]
                if not any(active_mask):
                    break
                
                # Pre-allocate updates
                beam_updates = [[] for _ in range(batch_size)]
                
                # Process active beams in parallel
                for i in np.where(active_mask)[0]:
                    active_beams, completed = beams[i]
                    new_active = []
                    new_completed = completed.copy()
                    
                    for node, path_score in active_beams:
                        # Leaf node detection
                        if not node.children:
                            if hasattr(node, 'kb_indices') and node.kb_indices:
                                new_completed.append((node, path_score))
                            continue
                        
                        # Classifier prediction
                        if hasattr(node, 'tree_classifier'):
                            try:
                                probs = node.tree_classifier.predict_proba(current_batch[i:i+1])[0]
                                children = []
                                for class_idx, prob in enumerate(probs):
                                    child_id = node.tree_classifier.model.model.classes_[class_idx]
                                    if child_id in node.children:
                                        children.append((node.children[child_id], path_score * prob))
                                
                                # Keep top beam_size children
                                children.sort(key=lambda x: x[1], reverse=True)
                                new_active.extend(children[:beam_size])
                            except:
                                # Fallback if classifier fails
                                children = [(child, path_score * (1.0/len(node.children))) 
                                        for child in node.children.values()]
                                new_active.extend(children[:beam_size])
                        else:
                            # Uniform distribution if no classifier
                            children = [(child, path_score * (1.0/len(node.children))) 
                                    for child in node.children.values()]
                            new_active.extend(children[:beam_size])
                    
                    # Update beams
                    new_active.sort(key=lambda x: x[1], reverse=True)
                    beam_updates[i] = (new_active[:beam_size], new_completed)
                
                # Apply updates
                for i in np.where(active_mask)[0]:
                    beams[i] = beam_updates[i]
            
            # ---- Vectorized Reranking ----
            batch_candidates = [[] for _ in range(batch_size)]
            all_entity_centroids = self.htree.entity_centroids
            
            for i, (_, completed) in enumerate(beams):
                if not completed:
                    continue
                    
                # Collect all candidate entities from completed beams
                candidates = {}
                for node, path_score in completed:
                    if not hasattr(node, 'kb_indices') or not node.kb_indices:
                        continue
                        
                    for idx in node.kb_indices:
                        eid = self.htree.labels[idx]
                        if eid in all_entity_centroids:
                            candidates[eid] = path_score  # Will be multiplied by reranker score
                
                if not candidates:
                    continue
                    
                # Prepare reranker inputs
                eids = list(candidates.keys())
                centroids = np.array([all_entity_centroids[eid] for eid in eids])
                input_emb = current_batch[i]
                
                # Vectorized similarity calculation
                if hasattr(completed[0][0], 'reranker') and completed[0][0].reranker:
                    X_pairs = np.hstack((
                        np.tile(input_emb, (len(eids), 1)),
                        centroids
                    ))
                    scores = completed[0][0].reranker.model.predict_proba(X_pairs)[:, 1]
                else:
                    # Fallback to cosine similarity
                    scores = cosine_similarity(input_emb.reshape(1, -1), centroids)[0]
                
                # Combine scores
                for eid, score in zip(eids, scores):
                    batch_candidates[i].append((eid, float(score * candidates[eid])))
                
                # Sort and keep top-k
                batch_candidates[i].sort(key=lambda x: x[1], reverse=True)
                all_results.append(batch_candidates[i][:k])
        
        return all_results

    def batch_predict(self, input_embs, labels, k=5, candidates=100, batch_size=32000):
        """
        Optimized batch prediction with evaluation.
        Returns CSR matrix of predictions and hit indicators.
        """
        n = len(input_embs)
        preds = self.predict(input_embs, k=k, beam_size=max(1, round(candidates/k)), batch_size=batch_size)
        
        # Convert to CSR format
        rows, cols, data = [], [], []
        hits = np.zeros(n, dtype=int)
        
        for i, pred in enumerate(preds):
            # Handle hits
            gold = labels[i]
            cand_ids = {eid for eid, _ in pred}
            
            if isinstance(gold, list):
                hits[i] = int(any(g in cand_ids for g in gold))
            else:
                hits[i] = int(gold in cand_ids)
            
            # Add to sparse matrix
            for eid, score in pred:
                try:
                    j = self.all_kb_ids.index(eid)
                    rows.append(i)
                    cols.append(j)
                    data.append(score)
                except ValueError:
                    continue
        
        csr = csr_matrix((data, (rows, cols)), shape=(n, len(self.all_kb_ids)))
        return csr, hits.tolist()