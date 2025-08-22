from collections import defaultdict
import os

os.makedirs("/app/joblib_tmp", exist_ok=True)
os.environ["JOBLIB_TEMP_FOLDER"] = "/app/temp"

import joblib
import pickle
import pandas as pd

import numpy as np

from scipy.sparse import csr_matrix, hstack, issparse
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
    
        self.EPS = 1e-12
        self.TOPK_DBG = 50
    
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

    def _safe_log_fuse(self, m, r, a):
        m = max(float(m), self.EPS)
        r = max(float(r), self.EPS)
        return (1.0 - a) * m + a * r

    def _topk_indices_rowwise(self, mat, k):
        if k <= 0:
            return np.zeros((mat.shape[0], 0), dtype=int)
        if k >= mat.shape[1]:
            return np.argsort(-mat, axis=1)
        part = np.argpartition(-mat, kth=k - 1, axis=1)[:, :k]
        idx = np.arange(mat.shape[0])[:, None]
        ordered = np.argsort(-mat[idx, part], axis=1)
        return part[idx, ordered]

    def _prepare_layer_infer(self, X_node, Z_node, mention_indices, C, matcher_scores, top_clusters_idx, prop_score):
        """
        Prepare augmented features and Z for the next layer.
        Returns list of tuples: (X_aug, Z_aug, mentions_sub, prop_score_sub)
        """
        inputs = []
        if matcher_scores is None or matcher_scores.size == 0:
            return inputs

        unique_clusters = np.unique(top_clusters_idx)
        if unique_clusters.size == 0:
            return inputs

        col_map = {c: i for i, c in enumerate(unique_clusters)}
        sliced = matcher_scores[:, unique_clusters]

        for c in unique_clusters:
            mask = np.any(top_clusters_idx == c, axis=1)
            if not np.any(mask):
                continue

            mentions_sub = mention_indices[mask]
            X_sub = X_node[mask]
            F = sliced[mask]

            feat_c = F[:, col_map[c]].reshape(-1, 1)
            feat_sum = F.sum(axis=1).reshape(-1, 1)
            feat_max = F.max(axis=1).reshape(-1, 1)
            extra = csr_matrix(np.hstack([feat_c, feat_sum, feat_max]))
            X_aug = hstack([X_sub, extra], format="csr")
            X_aug = normalize(X_aug, norm="l2", axis=1)

            local_idxs = C[:, c].nonzero()[0]
            if local_idxs.size == 0:
                continue

            Z_base = Z_node[local_idxs, :]
            X_sub_dense = X_sub.toarray() if issparse(X_sub) else np.asarray(X_sub)
            scores = X_sub_dense.dot(Z_base.T)
            label_feats = np.vstack([scores.mean(axis=0), scores.sum(axis=0), scores.max(axis=0)]).T
            Z_aug = normalize(np.hstack([Z_base, label_feats]), norm="l2", axis=1)

            # Propagate prop_score for this subset
            prop_score_sub = prop_score[mask]

            inputs.append((X_aug, Z_aug, mentions_sub, prop_score_sub))

        return inputs
    
    def predict(self, X_text, gold_labels=None, topk=10, beam_size=5,
                debug=False, n_jobs=-1, backend="threading", TARGET_LABEL=None):

        X = self.text_encoder.predict(X_text)
        Z = self.Z
        N = X.shape[0]

        all_global_max = [int(ml.local_to_global_idx.max()) for layer in self.model.hmodel for ml in layer]
        G = (max(all_global_max) + 1) if all_global_max else 0

        label_scores = [dict() for _ in range(N)]

        # Initial inputs: X, Z, mention idxs, prop_score=1
        current_inputs = [(X, Z, np.arange(N), np.ones(N, dtype=float))]

        for layer_idx, mlmodels in enumerate(self.model.hmodel):
            next_inputs = []
            if debug:
                print(f"Layer {layer_idx}")

            for ml, (X_node, Z_node, mention_indices, prop_score) in zip(mlmodels, current_inputs):
                if X_node.shape[0] == 0:
                    continue

                X_node_dense = X_node.toarray() if issparse(X_node) else np.asarray(X_node)
                matcher_scores = ml.matcher_model.predict_proba(X_node)
                matcher_scores = np.nan_to_num(matcher_scores, nan=0.0)

                # Map clusters to labels
                cluster_to_labels = ml.cluster_model.cluster_to_labels
                label_to_clusters = {}
                for c_id, labs in cluster_to_labels.items():
                    for lab in labs:
                        label_to_clusters.setdefault(int(lab), []).append(int(c_id))

                # Top clusters for each mention
                beam_size_eff = min(beam_size, matcher_scores.shape[1])
                top_clusters_idx = self._topk_indices_rowwise(matcher_scores, beam_size_eff)

                def _process_mention(i_row):
                    m_idx = int(mention_indices[i_row])
                    x_emb = X_node_dense[i_row].ravel()
                    gold_list = gold_labels[m_idx] if gold_labels is not None else []
                    gold_set = set(gold_list)
                    fused_pairs = []
                    debug_this = debug and (TARGET_LABEL is not None and TARGET_LABEL in gold_set)

                    clusters = top_clusters_idx[i_row]
                    local_labels = []
                    local_idxs = []

                    for c in clusters:
                        for g in cluster_to_labels[c]:
                            loc = ml.global_to_local_idx.get(g, None)
                            if loc is None:
                                continue
                            local_labels.append(g)
                            local_idxs.append(loc)

                    if not local_labels:
                        return m_idx, fused_pairs

                    Z_batch = Z_node[local_idxs, :]
                    X_rep = np.tile(x_emb.reshape(1, -1), (Z_batch.shape[0], 1))
                    feat_mat = np.hstack([X_rep, Z_batch])

                    for j, g in enumerate(local_labels):
                        rer = 1.0
                        if ml.reranker_model and g in ml.reranker_model.model_dict:
                            mdl = ml.reranker_model.model_dict[g]
                            xfeat = feat_mat[j:j+1, :]
                            try:
                                proba = mdl.predict_proba(xfeat)
                                rer = float(np.clip(proba[0, 1], self.EPS, 1.0))
                            except AttributeError:
                                try:
                                    df = mdl.decision_function(xfeat)[0]
                                    rer = float(np.clip(expit(df), self.EPS, 1.0))
                                except Exception:
                                    rer = 1.0
                            except Exception:
                                rer = 1.0

                        cluster_ids = label_to_clusters.get(int(g), [])
                        matcher_score = 1.0
                        if cluster_ids:
                            cs = [matcher_scores[i_row, c] for c in cluster_ids if 0 <= c < matcher_scores.shape[1]]
                            if cs:
                                matcher_score = float(max(cs))

                        fused_val = self._safe_log_fuse(matcher_score * prop_score[i_row], rer, ml.alpha)
                        fused_pairs.append((g, fused_val))

                        if debug_this:
                            gold_flag = " <<-- GOLD" if self.initial_labels[g] in gold_set else ""
                            print(f"    (label={g}) m={matcher_score:.6f} r={rer:.6f} fused={fused_val:.6f}{gold_flag}")

                    return m_idx, fused_pairs

                par_results = Parallel(n_jobs=n_jobs, backend=backend)(
                    delayed(_process_mention)(i_row) for i_row in range(len(mention_indices))
                )

                for m_idx, fused_pairs in par_results:
                    d = label_scores[m_idx]
                    for g, fv in fused_pairs:
                        if fv > d.get(g, 0.0):
                            d[g] = fv

                # Propagate top clusters for next layer
                surv_mask = np.array([len(top_clusters_idx[i]) > 0 for i in range(len(mention_indices))], dtype=bool)
                if surv_mask.any():
                    surv_mentions = mention_indices[surv_mask]
                    X_surv = X_node[surv_mask]
                    ms_surv = matcher_scores[surv_mask]
                    new_prop_score = np.array([max(label_scores[m].values()) for m in surv_mentions], dtype=float)

                    C = ml.cluster_model.c_node
                    top_clusters_idx_surv = self._topk_indices_rowwise(ms_surv * new_prop_score[:, None], beam_size_eff)

                    next_inputs.extend(self._prepare_layer_infer(
                        X_surv, Z_node, surv_mentions, C, ms_surv, top_clusters_idx_surv, new_prop_score
                    ))

            current_inputs = next_inputs

        # Build final sparse matrix and hits
        rows, cols, data = [], [], []
        hits = []
        for i in range(N):
            d = label_scores[i]
            top_labels = sorted(d.items(), key=lambda kv: -kv[1])[:topk] if d else []
            for g, s in top_labels:
                rows.append(i)
                cols.append(g)
                data.append(s)
            label_idx_found = -1
            gold_list = gold_labels[i] if gold_labels is not None else []
            for rank, (g, _) in enumerate(top_labels):
                if self.initial_labels[g] in gold_list:
                    label_idx_found = rank
                    break
            hits.append((int(label_idx_found != -1), label_idx_found, gold_list))

        score_mat = csr_matrix((data, (rows, cols)), shape=(N, G))
        return score_mat, hits