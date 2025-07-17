import os
import pickle
import random
import json

import numpy as np

from xmr4el.models.classifier_wrapper.classifier_model import ClassifierModel


class Reranker():
    
    def __init__(self, config, num_negatives):
        self.config = config
        self.num_negatives = num_negatives
        self.model = None
    
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
    
    def train(self, mention_embeddings, true_indices, entity_embs_dict):
        """
        Train the reranker on positive and negative mention-entity pairs.

        Args:
            mention_embeddings (np.ndarray): shape (N, d), mention vectors.
            true_indices (List[int]): KB indices of true entities for each mention.
            entity_embs_dict (Dict[int, np.ndarray]): mapping KB index to centroid vector.
        """
        X_pairs = []
        y = []
        # Build positive and negative pairs
        all_eids = list(entity_embs_dict.keys())
        for i, m_emb in enumerate(mention_embeddings):
            true_eid = true_indices[i]
            # positive
            pos = np.hstack((m_emb, entity_embs_dict[true_eid]))
            X_pairs.append(pos); y.append(1)
            # negatives
            neg_candidates = [e for e in all_eids if e != true_eid]
            sampled = random.sample(neg_candidates, min(self.num_negatives, len(neg_candidates)))
            for neg in sampled:
                neg_pair = np.hstack((m_emb, entity_embs_dict[neg]))
                X_pairs.append(neg_pair); y.append(0)
        X = np.vstack(X_pairs)
        y = np.array(y, dtype=np.int8)

        # train binary classifier
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