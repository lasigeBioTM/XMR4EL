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
        Train the reranker on positive and negative mention-entity pairs.

        Args:
            mention_embeddings (np.ndarray): shape (N, d), mention vectors.
            centroid_embeddings (List[np.array]): centroids embeddings
            entity_embs_dict (Dict[int, np.ndarray]): mapping KB index to centroid vector.
        """
        X_pairs = []
        y = []
        # Build positive and negative pairs
        
        # Convert Centroids into matrix, 
        centroid_matrix = np.vstack(centroid_embeddings)
        
        # print(mention_indices)
        
        for _, (syn_emb_list, true_idx) in enumerate(zip(mention_embeddings, mention_indices)):
            # print(true_idx)
            # print(type(centroid_embeddings), len(centroid_embeddings))
            true_centroid = centroid_embeddings[true_idx]
            # print(true_centroid, type(true_centroid))
            for syn_emb in syn_emb_list:
                pos = np.hstack((syn_emb, true_centroid))
                X_pairs.append(pos)
                y.append(1)
                
            # -------- Hard Negatives --------
            # Compute cosine similarity between true_centroid and all other centroids
            sims = cosine_similarity(true_centroid.reshape(1, -1), centroid_matrix)[0]
            
            # 1) Collect indices in the desired similarity band [0.4, 0.6], excluding the true index
            band_idxs = [i for i, s in enumerate(sims)
                        if i != true_idx and 0.4 <= s <= 0.7]
            
            # 2) If we have enough, take the top-N by descending similarity
            hard_neg_idxs = sorted(band_idxs, key=lambda i: sims[i], reverse=True)[:self.num_negatives]
            # print(hard_neg_idxs)
            
            # 3) If we’re short, backfill from the remaining negatives by highest similarity
            if len(hard_neg_idxs) < self.num_negatives:
                # All other candidates, sorted by descending sim, excluding true_idx and already chosen
                remaining = [i for i in np.argsort(-sims)
                            if i != true_idx and i not in hard_neg_idxs]
                need = self.num_negatives - len(hard_neg_idxs)
                hard_neg_idxs += remaining[:need]

            for neg_idx in hard_neg_idxs:
                neg_centroid = centroid_embeddings[neg_idx]
                for syn_emb in syn_emb_list:
                    neg_pair = np.hstack((syn_emb, neg_centroid))
                    X_pairs.append(neg_pair)
                    y.append(0)
                    
            
        X = np.vstack(X_pairs)
        y = np.array(y, dtype=np.int8)

        # train binary classifier
        self.config["kwargs"]["onevsrest"] = False
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