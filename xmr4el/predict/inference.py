import numpy as np

from collections import defaultdict

from scipy.sparse import csr_matrix

from sklearn.preprocessing import normalize

from xmr4el.featurization.transformers import Transformer



class SkeletonInference:
    
    def __init__(self, htree, p=1):
        self.htree = htree
        self.p = p

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
    
    def lp_hinge_fusion(self, g_score, h_score):
        g_term = np.exp(-max(1 - g_score, 0) ** self.p)
        h_term = np.exp(-max(1 - h_score, 0) ** self.p)
        return g_term * h_term
    
    def beam_search(self, htree, input_embs, beam_size=5):
        """
        Run hierarchical beam search on batch of mention embeddings.

        Args:
            htree (Skeleton): Root node of hierarchy.
            input_embs (np.ndarray): (N, d) mention embeddings.
            beam_size (int): Number of top clusters to keep at each level.

        Returns:
            List[List[int]]: For each mention, a list of candidate label indices.
        """
        N = input_embs.shape[0]
        beams = [[(htree, 0.0)] for _ in range(N)]
        final_labels = [[] for _ in range(N)]
        
        while True:
            next_beams = [[] for _ in range(N)]
            
            for i, input_emb in enumerate(input_embs):
                current_beam = beams[i]
                
                candidates = []
                for node, score in current_beam:
                    if not node.children:
                        # Leaf node - gather candidates
                        label_indices = np.arrange(len(node.labels)) # kb_indices ? 
                        global_indices = node.kb_indices[label_indices]
                        final_labels[i].extend(global_indices) # inst this just kb_indices ?
                        continue
                    
                    probs = self._predict_proba_classifier(node.classifier, input_emb.reshape(1, -1))[0]
                    topk = np.argsort(probs)[-beam_size:][::-1]
                    
                    for cluster_id in topk:
                        if cluster_id in node.children:
                            child = node.children[cluster_id]
                            candidates.append((child, score + probs[cluster_id]))
                    
                # keep top beam size candidates
                next_beams[i] = sorted(candidates, key=lambda x: -x[1])[-beam_size]
                
            # All beams finished
            if all(len(b) == 0 for b in next_beams):
                break
            beams = next_beams
            
        return final_labels
        
    def batch_inference(self, input_embs, gold_labels=None, beam_size=5, topk=10):
        """
        Perform batch inference using hierarchical matcher, reranker, and Lp-Hinge fusion.

        Args:
            input_embs: np.ndarray (N, d), mention embeddings.
            gold_labels: List[List[str]], gold label strings per mention.
            p: int, power for Lp-Hinge.
            beam_size: int, number of clusters kept at each level.
            topk: int, top predicted labels to keep per input.

        Returns:
            csr_matrix of shape (N, total_labels) with fused scores.
        """
        N = len(input_embs)
        rows, cols, data = [], [], []
        hits = []
        
        Z = self.htree.Z
        reranker_dict = self.htree.reranker
        
        candidate_list = self.beam_search(self.htree, input_embs, beam_size=beam_size)
        
        for i in range(N):
            x = input_embs[i]
            candidates = candidate_list[i]
            
            label_scores = []
            for label_idx in candidates:
                # matcher score g(x, c)
                g_score = 1.0 # Assume, for now, all equally
                
                # Reranker: h(x, i)
                # Didnt implement reranker doesnt exist
                reranker = reranker_dict.get(label_idx)
                label_emb = Z[label_idx]
                x_concat = np.concatenate([x, label_emb]).reshape(1, -1)
                h_score = self._predict_proba_classifier(reranker, x_concat)[0][1]
                
                fused_score = self.lp_hinge_fusion(g_score, h_score)
                
                label_scores.append([label_idx, fused_score])
                
            label_scores.append((label_idx, fused_score))
            top_labels = label_scores[:topk]
            
            found = False
            gold_set = set(gold_labels[i])
            for idx, score in top_labels:
                rows.append(i)
                cols.append(idx)
                data.append(score)
                if str(label_idx) in gold_set:
                    found = True
                    
            hits.append(1 if found else 0)
            
        csr = csr_matrix((data, (rows, cols)), shape=(N, len(self.htree.labels)))
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
        # transformer_emb = self.transform_input_text(input_text)
        
        # concat_emb = np.hstack((
        #      transformer_emb,
        #      text_emb
        # ))
        # concat_emb = text_emb
        
        concat_emb = normalize(concat_emb, norm="l2", axis=1)
        
        return concat_emb