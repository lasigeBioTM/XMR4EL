import os
import joblib
import pickle

import numpy as np

from scipy.sparse import csr_matrix
from scipy.special import expit

from datetime import datetime

from xmr4el.featurization.label_embedding_factory import LabelEmbeddingFactory
from xmr4el.featurization.preprocessor import Preprocessor
from xmr4el.featurization.text_encoder import TextEncoder
from xmr4el.xmr.base import HierarchicaMLModel


class XModel():
    
    def __init__(self, 
                 vectorizer_config=None,
                 transformer_config=None,
                 dimension_config=None,
                 clustering_config=None,
                 matcher_config=None,
                 reranker_config=None,
                 min_leaf_size=20,
                 max_leaf_size=None,
                 n_workers=8,
                 depth=1,
                 emb_flag=1,
                 ):
        
        self.vectorizer_config = vectorizer_config
        self.transformer_config = transformer_config
        self.dimension_config = dimension_config
        self.clustering_config = clustering_config
        self.matcher_config = matcher_config
        self.reranker_config = reranker_config
        
        self.min_leaf_size = min_leaf_size
        self.max_leaf_size = max_leaf_size
        
        self.n_workers = n_workers
        
        self.depth = depth
        self.emb_flag =emb_flag
        
        self._text_encoder = None
        self._hml = None
        self._original_labels = None
        self._X = None
        self._Y = None
        self._Z = None
    
    
    @property
    def text_encoder(self):
        return self._text_encoder
    
    @text_encoder.setter
    def text_encoder(self, value):
        self._text_encoder = value

    @property
    def model(self):
        return self._hml
    
    @model.setter
    def model(self, value):
        self._hml = value
        
    @property
    def initial_labels(self):
        return self._original_labels
    
    @initial_labels.setter
    def initial_labels(self, value):
        self._original_labels = value
        
    @property
    def X(self):
        return self._X
    
    @X.setter
    def X(self, value):
        self._X = value
        
    @property
    def Y(self):
        return self._Y
    
    @Y.setter
    def Y(self, value):
        self._Y = value
        
    @property 
    def Z(self):
        return self._Z
    
    @Z.setter
    def Z(self, value):
        self._Z = value
        
    def save(self, save_dir):
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        save_dir = os.path.join(save_dir, f"{self.__class__.__name__.lower()}_{timestamp}")
        os.makedirs(save_dir, exist_ok=False)
    
        state = self.__dict__.copy()
        
        model = self.model
        model_path = os.path.join(save_dir, f"hml")
        
        if model is not None:
            if hasattr(model, "save"):
                model.save(model_path)
            else:
                try:
                    joblib.dump(model, f"{model_path}.joblib")
                except ImportError:
                    with open(f"{model_path}.pkl", "wb") as f:
                        pickle.dump(model, f)  
            
            state.pop("_hml", None) # Popped _hml from class
        
        text_encoder = self.text_encoder
        text_encoder_path = os.path.join(save_dir, f"text_encoder")
        
        if text_encoder is not None:
            if hasattr(text_encoder, "save"):
                text_encoder.save(text_encoder_path)
            else:
                try:
                    joblib.dump(text_encoder, f"{text_encoder_path}.joblib")
                except ImportError:
                    with open(f"{text_encoder_path}.pkl", "wb") as f:
                        pickle.dump(text_encoder, f)  
                          
        with open(os.path.join(save_dir, "xmodel.pkl"), "wb") as fout:
            pickle.dump(state, fout)
    
    @classmethod
    def load(cls, load_dir):
        xmodel_path = os.path.join(load_dir, "xmodel.pkl")
        assert os.path.exists(xmodel_path), f"XModel path {xmodel_path} does not exist"
        
        with open(xmodel_path, "rb") as fin:
            model_data = pickle.load(fin)
            
        model = cls()
        model.__dict__.update(model_data)
        
        model_path = os.path.join(load_dir, "hml")
        hml = HierarchicaMLModel.load(model_path)
        setattr(model, "_hml", hml)
        
        text_encoder_path = os.path.join(load_dir, "text_encoder")
        text_encoder = TextEncoder.load(text_encoder_path)
        setattr(model, "_text_encoder", text_encoder)
        
        return model
    
    def _fit(self, X_text, Y_text):
        """Returns embeddings: ndarray"""
        
        self.initial_labels = Y_text
        
        X_processed, Y_label_to_indices = Preprocessor.prepare_data(X_text, Y_text)
        
        # Encode X_processed
        text_encoder = TextEncoder(
            vectorizer_config=self.vectorizer_config,
            transformer_config=self.transformer_config,
            dimension_config=self.dimension_config, 
            flag=self.emb_flag # Needs to be a variable, could have a stop to check
            )
        
        self.text_encoder = text_encoder
        
        X_emb = text_encoder.encode(X_processed)
        
        Y_label_matrix = LabelEmbeddingFactory.generate_label_matrix(Y_label_to_indices)
        
        # Process Labels
        Y_binazer, _ = LabelEmbeddingFactory.label_binarizer(Y_label_matrix)
        Z = LabelEmbeddingFactory.generate_PIFA(X_emb, Y_binazer)
        
        return X_emb, Y_binazer, Z 
    
    def train(self, X_text, Y_text):
        
        X, Y, Z = self._fit(X_text=X_text, Y_text=Y_text)
        
        self.X = X
        self.Y = Y
        self.Z = Z
        
        del X, Y, Z
        
        n_labels = self.Z.shape[0]
        local_to_global = np.arange(n_labels, dtype=int)
        
        global_to_local = {g: i for i, g in enumerate(local_to_global)}
        
        hml = HierarchicaMLModel(clustering_config=self.clustering_config, 
                                 matcher_config=self.matcher_config, 
                                 reranker_config=self.reranker_config, 
                                 min_leaf_size=self.min_leaf_size,
                                 max_leaf_size=self.max_leaf_size,
                                 n_workers=self.n_workers,
                                 layer=self.depth)
        
        hml.train(X_train=self.X, 
                  Y_train=self.Y, 
                  Z_train=self.Z, 
                  local_to_global=local_to_global,
                  global_to_local=global_to_local
                  )
        
        self.model = hml
        del hml
        print(self.model)

    def predict(self, X_text_query, gold_labels=None, topk=10, alpha=0.5, beam_size=10):
        """
        Hierarchical inference with reranking at every level and beam pruning.
        Returns:
        - csr_matrix of predicted scores (N x G, G = total global labels)
        - hits list for evaluation
        """
        # 1) Encode the query and get globals
        X_query = self.text_encoder.predict(X_text_query).toarray()  # (N, D)
        Z = self.Z                                                    # (G, D_label)
        N = X_query.shape[0]
        G = Z.shape[0]                                                # total global labels

        rows, cols, data = [], [], []
        hits = []

        for i in range(N):
            label_scores = {}  # dict: global_label -> best fused score
            x_init = X_query[i]

            # Initialize beam: start at layer 0, each root MLModel, score=0, feature vec=x_init
            beam = [(0, ml, 0.0, x_init) for ml in self.model.hmodel[0]]

            # Beam‐search down the hierarchy
            while beam:
                next_beam = []

                for layer_idx, ml, cum_score, x_aug in beam:
                    # 2) Get cluster‐membership scores for this node
                    #    matcher.proba shape: (1, n_clusters)
                    matcher_scores = ml.matcher_model.predict_proba(x_aug.reshape(1, -1))[0]
                    if matcher_scores.size == 0:
                        continue

                    # 3) Pick top‐k clusters
                    top_clusters = np.argsort(matcher_scores)[::-1][:topk]

                    for c in top_clusters:
                        cluster_score = matcher_scores[c]

                        # 4) Map cluster -> list of global label IDs
                        global_labels = ml.cluster_model.cluster_to_labels[c]

                        for g in global_labels:
                            # 5) Compute reranker score if available
                            rerank_score = cluster_score
                            reranker = ml.reranker_model
                            if reranker is not None and g in reranker.model_dict:
                                vec = np.hstack([x_aug, Z[g]]).reshape(1, -1)
                                raw = reranker.model_dict[g].decision_function(vec)[0]
                                rerank_score = expit(raw)

                            # 6) Fuse matcher+reranker with Lp‐hinge
                            fused = (cluster_score ** (1 - alpha)) * (rerank_score ** alpha)

                            # 7) Record best score for this global label
                            prev = label_scores.get(g, -np.inf)
                            label_scores[g] = max(prev, fused)

                            # 8) Prepare the feature extension for the next layer
                            #    Here we stack [fused, fused, fused]; you can compute
                            #    feat_sum/max over all clusters if desired.
                            feat_node = np.array([fused, fused, fused])
                            x_next = np.hstack([x_aug, feat_node])

                            # 9) Route to next‐layer models that handle label g
                            if layer_idx + 1 < len(self.model.hmodel):
                                for child_ml in self.model.hmodel[layer_idx + 1]:
                                    # child_ml.local_to_global_idx is array of globals it knows
                                    if g in child_ml.local_to_global_idx:
                                        next_beam.append((layer_idx + 1, child_ml,
                                                        cum_score + fused, x_next))

                # 10) Prune the beam to the top `beam_size` by cumulative score
                next_beam.sort(key=lambda tup: -tup[2])
                beam = next_beam[:beam_size]

            # --- finalize this query’s top‐k global labels ---
            sorted_labels = sorted(label_scores.items(), key=lambda kv: -kv[1])
            top_labels = sorted_labels[:topk]

            for g, score in top_labels:
                rows.append(i)
                cols.append(g)
                data.append(score)

            # --- compute hit for evaluation ---
            all_labels_id = [idx for idx, _ in sorted_labels]
            all_labels = [self.initial_labels[idx] for idx in all_labels_id]
            gold_set = set(gold_labels[i]) if gold_labels else set()
            found = False
            label_idx_found = -1

            for idx, label in enumerate(all_labels):
                if label in gold_set:
                    found = True
                    label_idx_found = idx
                    break

            hits.append((1, label_idx_found, gold_set) if found else (0, -1, gold_set))

        # Build final CSR: shape (N, G)
        csr = csr_matrix((data, (rows, cols)), shape=(N, G))
        return csr, hits