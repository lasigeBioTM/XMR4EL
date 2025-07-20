import os
import pickle
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
        """Save reranker config and trained model to disk."""
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
        """Load reranker config and model from disk."""
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
        """Train the classifier model with training data."""
        return ClassifierModel.train(X_corpus, y_corpus, config, dtype)
    
    def train(self, mention_embeddings, centroid_embeddings, mention_indices):
        """
        Train a reranker on positive and hard-negative mention-entity pairs.
        
        Optimizations:
        - Precompute all necessary similarity scores in one batch
        - Use more efficient array operations
        - Reduce memory usage through better preallocation
        """
        num_neg = self.num_negatives
        D = centroid_embeddings[0].shape[0]
        
        # Precompute all centroid similarities in one batch
        centroid_matrix = np.vstack(centroid_embeddings)  # shape (E, d)
        
        # Calculate total pairs more efficiently
        total_synonyms = sum(len(syn_list) for syn_list in mention_embeddings)
        total_pairs = total_synonyms * (1 + num_neg)
        
        # Preallocate arrays more efficiently
        X = np.empty((total_pairs, 2 * D), dtype=np.float32)
        y = np.empty(total_pairs, dtype=np.int8)
        
        ptr = 0
        # Precompute all similarity scores upfront
        all_true_centroids = np.array([centroid_embeddings[i] for i in mention_indices])
        all_sims = cosine_similarity(all_true_centroids, centroid_matrix)
        
        for idx, (syn_list, true_idx) in enumerate(zip(mention_embeddings, mention_indices)):
            true_cent = centroid_embeddings[true_idx]
            
            # Process positive pairs
            syn_array = np.array(syn_list)  # Convert to array once
            num_syn = len(syn_list)
            
            # Fill positive pairs in one operation
            X[ptr:ptr+num_syn, :D] = syn_array
            X[ptr:ptr+num_syn, D:] = true_cent
            y[ptr:ptr+num_syn] = 1
            ptr += num_syn
            
            # Find hard negatives using precomputed similarities
            sims = all_sims[idx]
            mask = (sims >= 0.2) & (sims <= 0.5)
            mask[true_idx] = False  # Exclude true index
            
            # Get top candidates
            candidates = np.where(mask)[0]
            if len(candidates) >= num_neg:
                hard_negs = candidates[np.argsort(-sims[candidates])[:num_neg]]
            else:
                # Fallback to most similar negatives
                backup = np.argsort(-sims)
                backup = backup[(backup != true_idx) & ~mask[backup]]
                hard_negs = np.concatenate([
                    candidates,
                    backup[:num_neg - len(candidates)]
                ])
            
            # Process negative pairs in batches
            for neg_idx in hard_negs:
                neg_cent = centroid_embeddings[neg_idx]
                X[ptr:ptr+num_syn, :D] = syn_array
                X[ptr:ptr+num_syn, D:] = neg_cent
                y[ptr:ptr+num_syn] = 0
                ptr += num_syn
        
        # Fit model
        self.model = self._train_classifier(X, y, self.config)
        return self.model

    def predict(self, mention_embedding, candidate_indices, entity_embs_dict, top_k=5):
        """Rank candidate entities for a single mention."""
        # Prepare all candidate pairs at once
        candidates = np.array([entity_embs_dict[eid] for eid in candidate_indices])
        pairs = np.hstack([
            np.tile(mention_embedding, (len(candidates), 1)),
            candidates
        ])
        
        # Predict scores in one batch
        scores = self.model.predict_proba(pairs)[:, 1]
        
        # Get top-k results
        top_idxs = np.argpartition(-scores, top_k)[:top_k]
        top_idxs = top_idxs[np.argsort(-scores[top_idxs])]
        
        return [(candidate_indices[i], float(scores[i])) for i in top_idxs]