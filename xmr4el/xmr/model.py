from collections import defaultdict
import os

os.makedirs("/app/joblib_tmp", exist_ok=True)
os.environ["JOBLIB_TEMP_FOLDER"] = "/app/temp"

import joblib
import pickle
import pandas as pd

import numpy as np

from scipy.sparse import csr_matrix, hstack
from scipy.special import expit
from joblib import Parallel, delayed

from sklearn.preprocessing import normalize

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
                 cut_half_cluster=False,
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
        self.cut_half_cluster = cut_half_cluster
        
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
        self.X, self.Y, self.Z = self._fit(X_text=X_text, Y_text=Y_text)

        n_labels = self.Z.shape[0]
        local_to_global = np.arange(n_labels, dtype=int)
        global_to_local = {g: i for i, g in enumerate(local_to_global)}

        hml = HierarchicaMLModel(
            clustering_config=self.clustering_config,
            matcher_config=self.matcher_config,
            reranker_config=self.reranker_config,
            min_leaf_size=self.min_leaf_size,
            max_leaf_size=self.max_leaf_size,
            n_workers=self.n_workers,
            cut_half_cluster=self.cut_half_cluster,
            layer=self.depth
        )

        hml.train(
            X_train=self.X,
            Y_train=self.Y,
            Z_train=self.Z,
            local_to_global=local_to_global,
            global_to_local=global_to_local
        )

        self.model = hml

    def predict(self, X_text, gold_labels=None, topk=10, beam_size=5,
            debug=False, n_debug=5, n_jobs=-1, backend="threading", TARGET_LABEL=None):
        """
        PECOS-style inference with per-layer Z augmentation.
        TARGET_IDX: global label index we are debugging/tracking.
        """
        TOPK_DBG = 20

        def _prepare_layer_infer(X_node, Z_node, mention_indices, C, fused_scores, top_clusters_idx):
            inputs = []
            K = C.shape[1]
            fused_dense = fused_scores.toarray() if hasattr(fused_scores, "toarray") else np.asarray(fused_scores)

            for c in range(K):
                mask = np.any(top_clusters_idx == c, axis=1)
                if not np.any(mask):
                    continue
                X_sub = X_node[mask]
                mentions_sub = mention_indices[mask]

                F = fused_dense[mask, :]
                feat_c  = F[:,c].reshape(-1,1)
                feat_sum= F.sum(axis=1,keepdims=True)
                feat_max= F.max(axis=1,keepdims=True)
                X_aug   = hstack([X_sub, csr_matrix(np.hstack([feat_c,feat_sum,feat_max]))],format="csr")
                X_aug   = normalize(X_aug, norm="l2", axis=1)

                local_idxs = C[:,c].nonzero()[0]
                Z_base = Z_node[local_idxs, :]
                X_dense = X_sub.toarray() if hasattr(X_sub,"toarray") else np.asarray(X_sub)
                scores  = X_dense.dot(Z_base.T)
                label_feats = np.vstack([scores.mean(axis=0), scores.sum(axis=0), scores.max(axis=0)]).T
                Z_aug = normalize(np.hstack([Z_base,label_feats]), norm="l2", axis=1)
                inputs.append((X_aug, Z_aug, mentions_sub))
            return inputs

        X = self.text_encoder.predict(X_text)
        Z = self.Z
        N = X.shape[0]
        G = max([ml.local_to_global_idx.max() for layer in self.model.hmodel for ml in layer]) + 1

        label_scores = [defaultdict(lambda:0.0) for _ in range(N)]
        # matched_cluster_found = [False]*N        
        has_initial = hasattr(self,"initial_labels") and (self.initial_labels is not None)
        
        current_inputs = [(X, Z, np.arange(N))]

        for layer_idx, mlmodels in enumerate(self.model.hmodel):
            next_inputs = []
            print("Layer Idx", layer_idx)

            for ml, (X_node, Z_node, mention_indices) in zip(mlmodels, current_inputs):
                alpha = ml.alpha
                Z_local = ml.cluster_model.z_node
                
                if debug:
                    print(f"  ML alpha (layer {layer_idx}): {alpha}")

                if X_node.shape[0] == 0:
                    if debug:
                        print("  Skipping node with 0 samples")
                    continue

                matcher_scores = ml.matcher_model.predict_proba(X_node)
                C = ml.cluster_model.c_node
                beam_size_eff = min(beam_size, matcher_scores.shape[1])
                top_clusters_idx = np.argsort(-matcher_scores, axis=1)[:, :beam_size_eff]

                def _process_single(i_row):
                    m_idx = mention_indices[i_row]
                    x_row = X_node[i_row]
                    x_emb = x_row.toarray().ravel() if hasattr(x_row,"toarray") else np.asarray(x_row).ravel()
                    
                    gold_label = gold_labels[i_row]
                    gold_label_idx = [idx for idx, init_label in enumerate(self.initial_labels) for gold_l in gold_label if init_label == gold_l]
                    
                    fused_pairs = []
                    debug_this = debug and TARGET_LABEL in gold_label

                    if debug_this:
                        print(f"\n=== DEBUG mention {m_idx} (beam={beam_size}, layer={layer_idx}) ===")

                    for c_id in top_clusters_idx[i_row]:
                        matcher_score = float(matcher_scores[i_row, c_id])
                        labels_in_c = ml.cluster_model.cluster_to_labels[c_id]
                        
                        if debug_this: 
                            print(f" Cluster {c_id} matcher_score={matcher_score}")

                        for g in labels_in_c:                      
                            loc = ml.global_to_local_idx.get(g, None)
                            z_emb = Z_local[loc]
                            feat = np.hstack([x_emb, z_emb])

                            # --- reranker ---
                            rer = 1.0
                            if ml.reranker_model and g in ml.reranker_model.model_dict:
                                mdl = ml.reranker_model.model_dict[g]
                                try:
                                    proba = mdl.predict_proba(feat.reshape(1,-1))
                                    rer = np.clip(proba[0,1], 1e-6, 1.0)
                                except Exception:
                                    rer = np.clip(expit(mdl.decision_function(feat.reshape(1,-1))[0]), 1e-6, 1.0)

                            # --- fuse ---
                            fused_val = (matcher_score**(1-alpha))*(rer**alpha)
                            fused_pairs.append((g, fused_val))

                            if debug_this:
                                gold_flag = " <<-- GOLD" if g in gold_label_idx else ""
                                print(f"     (label={g})  m={matcher_score:.4f} r={rer:.4f} fused={fused_val:.4f}{gold_flag}")

                    if debug_this:
                        fused_pairs_sorted = sorted(fused_pairs, key=lambda kv:-kv[1])
                        print(f"  --> Top-{TOPK_DBG} fused labels:")
                        for g, s in fused_pairs_sorted[:TOPK_DBG]:
                            gold_flag = " <<-- GOLD" if g in gold_label_idx else ""
                            print(f"        {g:6d}  {s:.4f}{gold_flag}")

                    return m_idx, fused_pairs, []

                par_results = Parallel(n_jobs=n_jobs, backend=backend)(
                    delayed(_process_single)(i_row) for i_row in range(len(mention_indices))
                )

                for m_idx, fused_pairs, _ in par_results:
                    for g, fv in fused_pairs:
                        prev = label_scores[m_idx].get(g, 0.0)
                        if fv > prev:
                            label_scores[m_idx][g] = fv

                next_inputs.extend(_prepare_layer_infer(
                    X_node, ml.cluster_model.z_node, mention_indices, C, matcher_scores, top_clusters_idx
                ))

            current_inputs = next_inputs

        # final aggregation
        rows, cols, data = [], [], []
        hits = []
        for i in range(N):
            d = label_scores[i]
            gold_list = gold_labels[i]
            top_labels = sorted(d.items(), key=lambda kv:-kv[1])[:topk] if d else []

            for g, s in top_labels:
                rows.append(i); cols.append(g); data.append(s)

            label_idx_found = -1
            for rank, (g, _) in enumerate(top_labels):
                if has_initial and self.initial_labels[g] in gold_list:
                    label_idx_found = rank; break

            hit = int(label_idx_found != -1)
            hits.append((hit, label_idx_found, gold_list))

        score_mat = csr_matrix((data, (rows, cols)), shape=(N, G))
        return score_mat, hits