from collections import defaultdict
import os

os.makedirs("/app/joblib_tmp", exist_ok=True)
os.environ["JOBLIB_TEMP_FOLDER"] = "/app/temp"

import warnings
warnings.filterwarnings("ignore", message=".*does not have valid feature names.*")

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

    def _fuse(self, m, r, a): 
        # (matcher_flat**(1-self.alpha))*(reranker_score**self.alpha)
        return m ** (1-a) * (r ** a)

    def _topk_indices_rowwise(self, mat, k):
        if k <= 0:
            return np.zeros((mat.shape[0], 0), dtype=int)
        if k >= mat.shape[1]:
            return np.argsort(-mat, axis=1)
        part = np.argpartition(-mat, kth=k - 1, axis=1)[:, :k]
        idx = np.arange(mat.shape[0])[:, None]
        ordered = np.argsort(-mat[idx, part], axis=1)
        return part[idx, ordered]
    
    def _compute_entity_fused_scores(self, ml, X_node, Z_node, alpha=0.5, batch_size=32768):
        """
        Return CSR matrix of shape (n_mentions_node, L_local) with fused scores per (mention, local_label).
        Fusion matches training: fused = matcher^(1-alpha) * reranker^alpha
        """

        # ---- matcher scores: (n_mentions_node x L_local) ----
        matcher_scores = ml.matcher_model.model.predict_proba(X_node)
        # ensure csr
        matcher_scores = csr_matrix(matcher_scores) if not hasattr(matcher_scores, "tocsr") else matcher_scores.tocsr()

        # ---- flatten pairs (row i, local col j) only where matcher>0 ----
        rows_list, cols_list, mvals = [], [], []
        for i in range(X_node.shape[0]):
            row = matcher_scores.getrow(i)
            if row.nnz == 0:
                continue
            rows_list.extend([i] * row.indices.size)
            cols_list.extend(row.indices.tolist())
            mvals.extend(row.data.tolist())

        if not rows_list:
            # nothing scored -> return all-zeros
            return matcher_scores  # just zeros

        rows = np.asarray(rows_list, dtype=np.int32)
        cols = np.asarray(cols_list, dtype=np.int32)
        mvals = np.asarray(mvals, dtype=np.float32)

        # ---- reranker on these pairs (batched) ----
        rvals = np.ones_like(mvals, dtype=np.float32)
        if ml.reranker_model and getattr(ml.reranker_model, "model_dict", None):
            X_dense = X_node.toarray() if hasattr(X_node, "toarray") else np.asarray(X_node)
            local_to_global = np.asarray(ml.local_to_global_idx, dtype=int)
            global_cols = local_to_global[cols]  # map local label -> global id

            # detect hinge vs proba from any one model
            first_model = next(iter(ml.reranker_model.model_dict.values()))
            is_hinge = first_model.config.get("type") == "sklearnsgdclassifier" and \
                    first_model.config.get("kwargs", {}).get("loss") == "hinge"

            for start in range(0, rows.size, batch_size):
                end = min(start + batch_size, rows.size)
                b_rows = rows[start:end]
                b_cols = cols[start:end]
                b_gids = global_cols[start:end]

                X_part = X_dense[b_rows]
                Z_part = Z_node[b_cols]  # local rows
                batch = np.hstack([X_part, Z_part])

                scores_batch = np.ones(end - start, dtype=np.float32)
                # group by global id within the batch
                for g in np.unique(b_gids):
                    mask = (b_gids == g)
                    mdl = ml.reranker_model.model_dict.get(int(g))
                    if mdl is None:
                        continue
                    sub = batch[mask]
                    try:
                        if is_hinge:
                            s = np.clip(expit(mdl.decision_function(sub)), 1e-6, 1.0)
                        else:
                            s = np.clip(mdl.predict_proba(sub)[:, 1], 1e-6, 1.0)
                        scores_batch[mask] = s
                    except Exception:
                        # keep ones on error
                        pass
                rvals[start:end] = scores_batch

        # ---- multiplicative fusion like training ----
        a = ml.alpha if ml.alpha is not None else alpha
        fused_flat = (mvals ** (1.0 - a)) * (rvals ** a)

        # ---- assemble CSR in local row-space ----
        n_rows = X_node.shape[0]
        L_local = Z_node.shape[0]
        entity_fused = csr_matrix((fused_flat, (rows, cols)), shape=(n_rows, L_local))
        return entity_fused
    
    def _prepare_layer_infer(self, ml, X_node, Z_node, mention_indices, 
                             prop_score, beam_size, debug=False):
        """
        Inference-time prepare for one node:
        - Align Z to this node's local row-space
        - Compute fused label scores (matcher+reranker, multiplicative)
        - Project to clusters to build X's +3 features from fused cluster scores
        - Build Z's +3 features from X·Zᵀ stats (mean/sum/max) for labels inside each cluster
        - Produce child inputs for the next layer with (X_aug, Z_aug, mentions_sub, prop_score_sub,
            local_to_global_next, global_to_local_next)
        - Return (child_inputs, per-mention fused label updates for global accumulation)
        """
        # ---------- Align to node-local space ----------
        C = ml.cluster_model.c_node                      # (L_local x K_clusters)
        local_to_global = np.asarray(ml.local_to_global_idx, dtype=int)
        print(f"Shape of local to global: {local_to_global.shape}")
        assert C.shape[0] == local_to_global.shape[0], "C rows must equal #local labels"
        
        print(f"C shape: {C.shape}, Z shape: {Z_node.shape}")

        # ---------- Fused scores at label-level (local row-space) ----------
        entity_fused = self._compute_entity_fused_scores(ml, X_node, Z_node, alpha=ml.alpha or 0.5)
        # print(f"Entity Fused: {entity_fused.shape}")
        # project to clusters to get fused cluster scores (for X features & routing)
        cluster_fused = entity_fused.dot(C)  # (n_mentions_node x K_clusters)
        # print(f"Cluster Fused: {cluster_fused.shape}")

        # dense view for top-k & features
        cluster_fused_dense = cluster_fused.toarray() if hasattr(cluster_fused, "toarray") else np.asarray(cluster_fused)

        # ---------- Beam: pick top clusters per mention using fused cluster scores ----------
        K = C.shape[1]
        beam = min(beam_size, K) if K > 0 else 0
        if beam <= 0 or K == 0:
            return [], []  # no children / nothing to update

        top_clusters_idx = self._topk_indices_rowwise(cluster_fused_dense, beam)

        # ---------- Prepare child inputs (augment X, augment Z) ----------
        n = X_node.shape[0]
        inputs = []
        # For accumulating per-mention global label scores (restricted to routed clusters)
        fused_updates = []  # list of (m_global_idx, [(global_label, fused_value), ...])

        # build a quick reverse map for the next step
        global_to_local_next_cache = {}

        # precompute union of local labels to consider per mention (labels in routed clusters)
        labels_per_mention = []
        for i in range(n):
            clusters = top_clusters_idx[i]
            local_idxs_union = np.unique(np.concatenate([C[:, c].nonzero()[0] for c in clusters])) if clusters.size else np.array([], dtype=int)
            labels_per_mention.append(local_idxs_union)

        # ---- Update fused label scores (map local->global, apply propagation) ----
        # We only update labels that lie in routed clusters for each mention (keeps it sparse).
        entity_fused_dense = entity_fused.toarray() if hasattr(entity_fused, "toarray") else np.asarray(entity_fused)
        for i_row in range(n):
            m_idx = int(mention_indices[i_row])
            lab_local = labels_per_mention[i_row]
            if lab_local.size == 0:
                fused_updates.append((m_idx, []))
                continue
            vals = entity_fused_dense[i_row, lab_local]
            if prop_score is not None:
                vals = vals * float(prop_score[i_row])
            pairs = list(zip(local_to_global[lab_local].tolist(), vals.tolist()))
            fused_updates.append((m_idx, pairs))

        # ---------- Build child inputs by cluster (order by cluster id to match training) ----------
        # We use the same per-cluster loop as training's prepare_layer, and the same X/Z feature recipes.
        for c in range(C.shape[1]):
            mask = np.any(top_clusters_idx == c, axis=1)  # mentions routed to cluster c
            # print(mask)
            if not np.any(mask):
                continue

            mentions_sub = mention_indices[mask]
            X_sub = X_node[mask]

            # ----- X +3 features from fused cluster scores -----
            F = cluster_fused_dense[mask, :]        # fused over all clusters for these mentions
            feat_c = F[:, c].reshape(-1, 1)
            feat_sum = F.sum(axis=1, keepdims=True)
            feat_max = F.max(axis=1, keepdims=True)
            X_aug = hstack([X_sub, csr_matrix(np.hstack([feat_c, feat_sum, feat_max]))], format="csr")
            X_aug = normalize(X_aug, norm="l2", axis=1)

            # ----- Z +3 features from X_sub · Z_baseᵀ stats (mean/sum/max) -----
            local_idxs = C[:, c].nonzero()[0]
            if local_idxs.size == 0:
                continue
            
            print(f"{c} -> Local Indices: {local_idxs}, {local_idxs.shape}")
            # Map node-local to parent-local first
            parent_local_idxs = [g2l_parent[int(l2g_parent[i])] for i in ml.local_to_global_idx[local_idxs]]
            Z_base = Z_node[parent_local_idxs, :]
            
            try:
                X_sub_dense = X_sub.toarray()
            except Exception:
                X_sub_dense = np.asarray(X_sub)

            if X_sub_dense.size == 0 or Z_base.size == 0:
                label_feats = np.zeros((Z_base.shape[0], 3), dtype=Z_base.dtype)
            else:
                scores = X_sub_dense.dot(Z_base.T)
                label_feats = np.vstack([scores.mean(axis=0), scores.sum(axis=0), scores.max(axis=0)]).T

            Z_aug = normalize(np.hstack([Z_base, label_feats]), norm="l2", axis=1)

            # ----- mappings for the child node -----
            local_to_global_next = l2g_parent[ml.local_to_global_idx[local_idxs]]
            global_to_local_next = global_to_local_next_cache.get(id(local_to_global_next))
            if global_to_local_next is None:
                global_to_local_next = {int(g): int(i) for i, g in enumerate(local_to_global_next)}
                global_to_local_next_cache[id(local_to_global_next)] = global_to_local_next

            # ----- propagate prop score -----
            prop_score_sub = prop_score[mask] if prop_score is not None else np.ones(mentions_sub.shape[0], dtype=float)

            print(f"Augmented Z: {Z_aug.shape}")

            # collect child tuple in the SAME semantic format training used
            inputs.append((X_aug, Z_aug, mentions_sub, prop_score_sub, local_to_global_next, global_to_local_next))

            if debug:
                print(f"[infer] cluster {c}: X_sub={X_sub.shape} -> X_aug={X_aug.shape}, "
                    f"Z_base={Z_base.shape} -> Z_aug={Z_aug.shape}, routed_mentions={mentions_sub.size}")

        return inputs, fused_updates

    def predict(self, X_text, gold_labels=None, topk=10, beam_size=5,
            debug=False, n_jobs=-1, backend="threading", TARGET_LABEL=None):

        X0 = self.text_encoder.predict(X_text)          # (N x d)
        Z0 = self.Z                                      # global base/augmented label matrix
        N = X0.shape[0]

        # global label-space size
        G = int(max([int(ml.local_to_global_idx.max()) for layer in self.model.hmodel for ml in layer]) + 1)

        label_scores = [dict() for _ in range(N)]

        # Initial inputs: (X, Z, mention idxs, prop_score, l2g, g2l)
        current_inputs = [(X0, Z0, np.arange(N), np.ones(N, dtype=float))]

        for layer_idx, mlmodels in enumerate(self.model.hmodel):
            if debug:
                print(f"\n=== Layer {layer_idx} ===\n")
            next_inputs = []

            # one ml per input (same contract as training)
            for ml, (X_node, Z_parent, mention_indices, prop_score) in zip(mlmodels, current_inputs):
                # print(ml.cluster_model.z_node.shape, Z_parent.shape)
                if X_node.shape[0] == 0:
                    continue

                print(f"\nSTART -> Prepare Layer Infer, layer {layer_idx}, ml {ml.__hash__}")
                # prepare child inputs + local fused updates
                child_inputs, fused_updates = self._prepare_layer_infer(
                    ml=ml,
                    X_node=X_node,
                    Z_node=Z_parent,
                    mention_indices=mention_indices,
                    prop_score=prop_score,
                    beam_size=beam_size,
                    debug=debug
                )
                print(f"END -> Prepare Layer Infer, layer {layer_idx}, ml {ml.__hash__}\n")
                

                # accumulate fused label scores in GLOBAL id-space
                for m_idx, pairs in fused_updates:
                    d = label_scores[m_idx]
                    for g_global, val in pairs:
                        if val > d.get(int(g_global), 0.0):
                            d[int(g_global)] = float(val)

                # append children to next layer input queue
                next_inputs.extend(child_inputs)

            current_inputs = next_inputs

        # ----- finalize sparse score matrix and hits -----
        rows, cols, data = [], [], []
        hits = []
        for i in range(N):
            d = label_scores[i]
            top_labels = sorted(d.items(), key=lambda kv: -kv[1])[:topk] if d else []
            for g, s in top_labels:
                rows.append(i)
                cols.append(int(g))
                data.append(float(s))

            label_idx_found = -1
            gold_list = gold_labels[i] if gold_labels is not None else []
            for rank, (g, _) in enumerate(top_labels):
                if self.initial_labels[g] in gold_list:
                    label_idx_found = rank
                    break
            hits.append((int(label_idx_found != -1), label_idx_found, gold_list))

        score_mat = csr_matrix((data, (rows, cols)), shape=(N, G))
        return score_mat, hits