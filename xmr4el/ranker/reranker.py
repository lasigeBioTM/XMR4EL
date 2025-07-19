import os
import pickle
import random
import json

import numpy as np

from sklearn.metrics.pairwise import cosine_similarity

from xmr4el.models.classifier_wrapper.classifier_model import ClassifierModel


class Reranker():
    
    def __init__(self, config, num_negatives, model=None):
        self.config = config
        self.num_negatives = num_negatives
        self.model = model
    
    def save(self, save_dir):
        """
        Save reranker config and trained model to disk.
        """
        os.makedirs(save_dir, exist_ok=True)
        # Save config
        cfg_path = os.path.join(save_dir, "reranker_config.json")
        with open(cfg_path, "w", encoding="utf-8") as fout:
            json.dump(self.config, fout)
        # Save model
        model_path = os.path.join(save_dir, "reranker_model.pkl")
        with open(model_path, "wb") as fout:
            pickle.dump(self.model, fout)

    @classmethod
    def load(cls, load_dir):
        """
        Load reranker config and model from disk.
        """
        # Load config
        cfg_path = os.path.join(load_dir, "reranker_config.json")
        assert os.path.exists(cfg_path), f"Config not found in {load_dir}"
        with open(cfg_path, "r", encoding="utf-8") as fin:
            config = json.load(fin)
        # Load model
        model_path = os.path.join(load_dir, "reranker_model.pkl")
        assert os.path.exists(model_path), f"Model not found in {load_dir}"
        with open(model_path, "rb") as fin:
            model = pickle.load(fin)
        
        return cls(config=config, num_negatives=config.get("num_negatives", 5), model=model)
    
    @staticmethod
    def _train_classifier(X_corpus, y_corpus, config, dtype=np.float32):
        """Trains the classifier model with the training data

        Args:
            trn_corpus (np.array): Trainign data as a Dense Array
            config (dict): Configurations of the clustering model
            dtype (np.float): Type of the data inside the array

        Return:
            RankingModel (ClassifierModel): Trained Classifier Model
        """
        # Delegate training to ClassifierModel class
        return ClassifierModel.train(X_corpus, y_corpus, config, dtype)
    
    def train(self, mention_embeddings, centroid_embeddings, mention_indices):
        """
        Train a reranker (LogisticRegression) on positive and hard-negative
        mention–entity pairs, with memory‑efficient preallocation.
        
        Args:
            mention_embeddings: List[List[np.ndarray]] of shape (num_mentions, num_synonyms, d)
            centroid_embeddings: List[np.ndarray] of shape (num_entities, d)
            mention_indices: List[int] of true-entity indices per mention
        """
        num_neg = self.num_negatives
        D = centroid_embeddings[0].shape[0]
        
        # Precompute centroid similarity matrix once
        centroid_matrix = np.vstack(centroid_embeddings)  # shape (E, d)
        
        # Compute total number of pairs we will generate
        total_pairs = 0
        for syn_list in mention_embeddings:
            total_pairs += len(syn_list) * (1 + num_neg)
        
        # Preallocate feature and label arrays
        X = np.empty((total_pairs, 2 * D), dtype=np.float32)
        y = np.empty((total_pairs,), dtype=np.int8)
        
        ptr = 0
        # For each mention
        for syn_list, true_idx in zip(mention_embeddings, mention_indices):
            true_cent = centroid_embeddings[true_idx]
            
            # Positive pairs
            for syn_emb in syn_list:
                X[ptr, :D] = syn_emb.astype(np.float32)
                X[ptr, D:] = true_cent.astype(np.float32)
                y[ptr] = 1
                ptr += 1
            
            # Hard negatives: find candidates in desired sim band
            sims = cosine_similarity(true_cent.reshape(1, -1), centroid_matrix)[0]
            band = [i for i, s in enumerate(sims) if i != true_idx and 0.2 <= s <= 0.5]
            hard_negs = sorted(band, key=lambda i: sims[i], reverse=True)[:num_neg]
            if len(hard_negs) < num_neg:
                # backfill if needed
                backup = [i for i in np.argsort(-sims) if i != true_idx and i not in hard_negs]
                hard_negs += backup[:(num_neg - len(hard_negs))]
            
            # Negative pairs
            for neg_idx in hard_negs:
                neg_cent = centroid_embeddings[neg_idx]
                for syn_emb in syn_list:
                    X[ptr, :D] = syn_emb.astype(np.float32)
                    X[ptr, D:] = neg_cent.astype(np.float32)
                    y[ptr] = 0
                    ptr += 1
        
        # Sanity check
        assert ptr == total_pairs, f"Expected {total_pairs} pairs, built {ptr}"
        
        # Fit a logistic regression model
        # You can tune solver/penalty as needed
        model = self._train_classifier(X, y, self.config)
        self.model = model
        return model


    def predict(self, mention_embedding, candidate_indices, entity_embs_dict, top_k=5):
        """
        Rank candidate entities for a single mention.

        Args:
            mention_embedding (np.ndarray): shape (d,)
            candidate_indices (List[int]): KB indices to score.
            entity_embs_dict (Dict[int, np.ndarray]): mapping index -> centroid.
            top_k (int): number of top candidates to return.

        Returns:
            List of (kb_index, score) tuples sorted desc.
        """
        pairs = np.vstack([
            np.hstack((mention_embedding, entity_embs_dict[eid]))
            for eid in candidate_indices
        ])
        # delegate prediction to underlying model
        scores = self.model.predict_proba(pairs)[:, 1]
        top_idxs = np.argsort(scores)[-top_k:][::-1]
        return [(candidate_indices[i], float(scores[i])) for i in top_idxs]