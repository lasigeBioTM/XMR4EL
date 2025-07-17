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
        Route a single mention embedding through the tree using the classifiers.
        Then rerank entities at the leaf nodes using their respective rerankers.

        Args:
            input_emb (np.ndarray): shape (d,) - Embedding for a mention
            k (int): Top-k results to return
            beam_size (int): Number of children to keep at each level (beam search)

        Returns:
            List[Tuple[str, float]]: List of (kb_id, score) pairs for top-k candidates
        """
        beams = [(self.htree, 1.0)]  # Start at the root with path score = 1.0

        # ---- Beam search routing ----
        while True:
            next_beams = []

            for node, path_score in beams:
                if not node.children:
                    # Reached a leaf node, keep as is
                    next_beams.append((node, path_score))
                    continue

                # Predict class probabilities for children
                probs = self._predict_proba_classifier(node.tree_classifier, input_emb.reshape(1, -1))[0]

                class_labels = node.tree_classifier.model.model.classes_

                # 2b) Build (label, child_node, prob) only for valid children
                candidates = []
                for label, child_node in node.children.items():
                    # find the probability for this label (0.0 if absent)
                    prob = float(probs[list(class_labels).index(label)])
                    candidates.append((label, child_node, prob))

                # 2c) Select top beam_size children by their prob
                candidates.sort(key=lambda x: x[2], reverse=True)
                for label, child_node, prob in candidates[:beam_size]:
                    # update path score
                    next_beams.append((child_node, path_score * prob))

            # 2d) Stop once every beam is at a node that can rerank
            #    i.e. either it's a true leaf (no children) or it has a reranker
            if all(
                (not node.children) or hasattr(node, "reranker")
                for node, _ in next_beams
            ):
                beams = next_beams
                break

            beams = next_beams

        # print(beams)

        # ---- Rerank in each leaf node ----
        # Use entity_centroids stored globally in the root
        all_entity_centroids = self.htree.entity_centroids
        final_candidates = []

        for node, path_score in beams:
            print(node, path_score)
            print(node.reranker)
            if not hasattr(node, "reranker") or node.reranker is None:
                # print("Donest have an reranker, odd")
                continue  # Skip if this node has no reranker (shouldn't happen)

            reranker = node.reranker
            kb_indices = node.kb_indices  # List of indices into self.all_kb_ids

            # Get corresponding kb_ids
            kb_ids = [self.htree.labels[i] for i in kb_indices]

            # Build feature pairs for reranking: [mention_emb | entity_centroid]
            X_pairs = []
            valid_ids = []
            for eid in kb_ids:
                if eid not in all_entity_centroids:
                    continue  # Skip if no centroid (shouldn’t happen ideally)
                eid_emb = all_entity_centroids[eid]
                pair_emb = np.hstack((input_emb, eid_emb))  # Concatenate mention + entity
                X_pairs.append(pair_emb)
                valid_ids.append(eid)

            if not X_pairs:
                continue

            X_pairs = np.vstack(X_pairs)
            scores = self._predict_proba_classifier(reranker.model, X_pairs)[:, 1]  # Binary classification score (prob. of match)

            # Scale score by path probability
            reranked = [(eid, float(score * path_score)) for eid, score in zip(valid_ids, scores)]
            final_candidates.extend(reranked)

        # ---- Final Top-K ----
        final_candidates.sort(key=lambda x: x[1], reverse=True)
        return final_candidates[:k]

    
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