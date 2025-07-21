import numpy as np

from scipy.sparse import csr_matrix

from sklearn.preprocessing import normalize

from xmr4el.featurization.transformers import Transformer



class SkeletonInference:
    
    def __init__(self, htree, all_kb_ids):
        self.htree = htree
        self.all_kb_ids = all_kb_ids

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
    
    def inference(self, input_emb, k=5, beam_size=3):
        """
        Fixed version that properly reaches leaf nodes.
        Key changes:
        1. Proper leaf node detection
        2. Correct beam search termination
        3. Handles cases where reranker might be missing
        """
        beams = [(self.htree, 1.0)]  # (node, cumulative_score)

        # ---- Improved Beam Search ----
        while beams:
            next_beams = []
            
            for node, path_score in beams:
                # True leaf nodes have no children AND contain kb_indices
                is_leaf = not node.children and hasattr(node, 'kb_indices') and node.kb_indices
                
                if is_leaf:
                    next_beams.append((node, path_score))
                    continue

                if not node.children:  # Empty non-leaf node
                    continue

                # Get classifier predictions
                try:
                    probs = self._predict_proba_classifier(node.tree_classifier, input_emb.reshape(1, -1))[0]
                except AttributeError:
                    # Handle nodes without classifiers
                    continue

                # Match classifier classes to actual children
                valid_children = []
                for class_idx, prob in enumerate(probs):
                    child_id = node.tree_classifier.model.model.classes_[class_idx]
                    if child_id in node.children:
                        valid_children.append((node.children[child_id], prob))

                # Keep top beam_size children
                valid_children.sort(key=lambda x: x[1], reverse=True)
                for child, prob in valid_children[:beam_size]:
                    next_beams.append((child, path_score * prob))

            # Stop if all beams are leaves or no progress
            if not next_beams or all(not node.children 
                                     and hasattr(node, 'kb_indices') 
                                     and node.kb_indices for node, _ in next_beams):
                beams = next_beams
                break
                
            beams = next_beams
        
        # print(beams)

        # ---- Reranking ----
        final_candidates = []
        all_entity_centroids = self.htree.entity_centroids
        
        for node, path_score in beams:
            if not hasattr(node, 'kb_indices') or not node.kb_indices:
                continue

            kb_indices = node.kb_indices
            kb_ids = [self.htree.labels[i] for i in kb_indices]
            
            # Skip if no reranker (use centroid similarity as fallback)
            if not hasattr(node, 'reranker') or not node.reranker:
                continue

            # Use reranker if available
            X_pairs = np.array([
                np.hstack((input_emb, all_entity_centroids[eid]))
                for eid in kb_ids if eid in all_entity_centroids
            ])
            scores = node.reranker.model.predict_proba(X_pairs)[:, 1] if len(X_pairs) > 0 else []

            # print(scores)

            # Combine scores with path probability
            valid_pairs = [
                (eid, float(score * path_score))
                for eid, score in zip(kb_ids, scores)
                if eid in all_entity_centroids
            ]
            final_candidates.extend(valid_pairs)

        # Return top-k results
        final_candidates.sort(key=lambda x: x[1], reverse=True)
        # print(final_candidates)
        return final_candidates

    
    def batch_inference(self, input_embs, labels, k=5, candidates=15):
        """
        End-to-End batch prediction with hit-rate evaluation.
        Returns sparse prediction and hit indicators
        """
        
        n = input_embs.shape[0]
        results = [None] * n
        hits = [0] * n
        # for each sample 
        for i in range(n):
            emb = input_embs[i]
            pred = self.inference(emb, k=k, beam_size=round(candidates/k))
            results[i] = ([self.all_kb_ids.index(eid) for eid, _ in pred], [score for _, score in pred])
            gold = labels[i]
            cand_ids = [eid for eid, _ in pred]
            # print(gold, cand_ids)
            # print(gold, pred)
            # exit()
            
            # exit()
            if isinstance(gold, list):
                hits[i] = int(any(g in cand_ids for g in gold))
            else:
                hits[i] = int(gold in cand_ids)
        # convert to CSR
        rows, cols, data = [], [], []
        for i, (idxs, scores) in enumerate(results):
            for j, score in zip(idxs, scores):
                rows.append(i)
                cols.append(j)
                data.append(score)
        csr = csr_matrix((data, (rows, cols)), shape=(n, len(self.all_kb_ids)))
        return csr, hits
    
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