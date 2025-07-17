import numpy as np

from scipy.sparse import csr_matrix

from sklearn.preprocessing import normalize

from xmr4el.featurization.transformers import Transformer



class SkeletonInference:
    
    def __init__(self, htree, all_kb_ids):
        self.htree = htree
        self.all_kb_ids = all_kb_ids

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
        Route a single mention embedding through the tree, then rerank.
        Args:
            input_emb: np.ndarray shape (d,)
            k: final top-k reranked
            beam_size: number of children to explore per level
        Returns:
            list of (kb_index, score)
        """
        # Initialize beams: list of tuples (node, score)
        beams = [(self.htree, 1.0)]
        # Beam search down the tree
        while True:
            next_beams = []
            for node, score in beams:
                children = list(node.children.items()) # list of (child_label, child_node)
                if not children:
                    next_beams.append((node, score))
                    continue
                # predict child probabilities
                probs = self._predict_proba_classifier(node.tree_classifier, input_emb.reshape(1, -1))[0]
                # select top beam_size children
                top_idxs = np.argsort(probs)[-beam_size:]
                for idx in top_idxs:
                    child_label, child_node = children[idx]
                    next_beams.append((child_node, score * probs[idx]))
            # check if all beams are leaves
            if all(not node.children for node, _ in next_beams):
                break
            beams = next_beams
        
        # Collect candidate KB indices from leaf beams
        candidate_eids = set()
        for node, path_score in beams:
            # use clustering to map leaf cluster to kb_indices
            mask = (node.clustering_model.labels() == node.kb_indices) # Makes no sense
            candidate_eids.update(node.kb_indices)
            
        # Final reranking
        reranker = self.htree.reranker # assume global reranker trained across all
        entity_centroids = self.htree.entity_centroids
        # prepare pairs 
        X_pairs = []
        eids = list(candidate_eids)
        for eid in eids:
            X_pairs.append(np.hstack((input_emb, entity_centroids[eid])))
        X_pairs = np.vstack(X_pairs)
        scores = self._predict_proba_classifier(reranker, X_pairs)[:, 1]
        # top-k
        top_idxs = np.argsort(scores)[-k:][::-1]
        return [(eids[i], float(scores[i])) for i in top_idxs]
    
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
                
    def transform_input_text(self, input_text):
        transformer_model = self._predict_transformer(
            input_text, 
            self.htree.transformer_config, 
        )
        transformer_emb = normalize(transformer_model.model.embeddings, norm="l2", axis=1)

        return transformer_emb