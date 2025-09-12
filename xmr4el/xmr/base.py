import os
import re
import gc
import pickle
import tempfile
import joblib
import heapq
import shutil

import numpy as np

from scipy.sparse import csr_matrix, hstack, vstack, eye as sp_eye
from scipy.special import expit

from sklearn.preprocessing import normalize

from collections import defaultdict
from typing import Any, Counter, Dict, List, Optional, Tuple
from uuid import uuid4

from joblib import Parallel, delayed, cpu_count, parallel_backend
from pathlib import Path

from xmr4el.clustering.model import Clustering
from xmr4el.matcher.model import Matcher
from xmr4el.ranker.model import Ranker

model_dir = Path(tempfile.mkdtemp(prefix="ml_model_dir"))


class MLModel():

    def __init__(self, 
                 clustering_config=None, 
                 matcher_config=None, 
                 ranker_config=None,
                 min_leaf_size=20,
                 max_leaf_size=None,
                 ranker_every_layer=False,
                 is_last_layer=False,
                 layer=None,
                 n_workers=8,
                 ):
        
        self.clustering_config = clustering_config
        self.matcher_config = matcher_config
        self.ranker_config = ranker_config
        self.min_leaf_size = min_leaf_size
        self.max_leaf_size = max_leaf_size
        self.ranker_every_layer = ranker_every_layer
        self.is_last_layer = is_last_layer
        self.layer = layer
        self.n_workers = n_workers
        
        self._local_to_global_idx = None
        self._global_to_local_idx = None
        
        self._cluster_model = None
        self._matcher_model = None
        self._ranker_model = None
        self._fused_scores = None
        self._alpha = None
        self._label_embeddings = None
    
    @property
    def local_to_global_idx(self):
        return self._local_to_global_idx
    
    @local_to_global_idx.setter
    def local_to_global_idx(self, arr: np.ndarray):
        """
        arr[i] should be the global KB label ID for local label index i.
        """
        self._local_to_global_idx = arr
        # build inverse map
        self._global_to_local_idx = {g: i for i, g in enumerate(arr)}
    
    @property
    def global_to_local_idx(self):
        return self._global_to_local_idx
    
    @global_to_local_idx.setter
    def global_to_local_idx(self, value):
        self._global_to_local_idx = value
    
    @property
    def cluster_model(self):
        return self._cluster_model
    
    @cluster_model.setter
    def cluster_model(self, value):
        self._cluster_model = value

    @property
    def matcher_model(self):
        return self._matcher_model
    
    @matcher_model.setter
    def matcher_model(self, value):
        self._matcher_model = value

    @property
    def ranker_model(self):
        return self._ranker_model
    
    @ranker_model.setter
    def ranker_model(self, value):
        self._ranker_model = value
        
    @property
    def fused_scores(self):
        return self._fused_scores
    
    @fused_scores.setter
    def fused_scores(self, value):
        self._fused_scores = value
        
    @property
    def alpha(self):
        return self._alpha
    
    @alpha.setter
    def alpha(self, value):
        self._alpha = value
        
    @property
    def label_embeddings(self):
        return self._label_embeddings
    
    @label_embeddings.setter
    def label_embeddings(self, value):
        self._label_embeddings = value
    
    @property
    def is_empty(self):
        return True if self.cluster_model is None else False
    
    @staticmethod
    def save_model_temp(model , label: int) -> str:
        """Persist a temporary model for the given label."""
        sub_dir = model_dir / str(label)
        sub_dir.mkdir(parents=True, exist_ok=True)
        model.save(str(sub_dir))
        return str(sub_dir)

    @staticmethod
    def delete_model_temp() -> None:
        """Remove all temporary ranker models from disk."""
        shutil.rmtree(model_dir)
    
    def save(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)  # Ensure directory exists

        state = self.__dict__.copy()

        # Mapping attribute names to their internal keys
        model_attrs = {
            "cluster_model": "_cluster_model",
            "matcher_model": "_matcher_model",
            "ranker_model": "_ranker_model"
        }

        for model_name, attr_key in model_attrs.items():
            model = getattr(self, model_name)
            if model is not None:
                model_path = os.path.join(save_dir, model_name)

                if hasattr(model, 'save') and callable(model.save):
                    model.save(model_path)
                else:
                    joblib.dump(model, f"{model_path}.joblib")

                # Remove model from state before pickling
                state.pop(attr_key, None)

        # Save fused scores separately
        fused_scores = self.fused_scores
        if fused_scores is not None:
            np.save(os.path.join(save_dir, "fused_scores.npy"), fused_scores)
            state.pop("_fused_scores", None)
        else:
            raise ValueError("fused_scores is None. Cannot save.")

        # Save label embeddings separately
        label_embeddings = self.label_embeddings
        if label_embeddings is not None:
            np.save(os.path.join(save_dir, "label_embeddings.npy"), label_embeddings)
            state.pop("_label_embeddings", None)

        # Save remaining state
        with open(os.path.join(save_dir, "mlmodel.pkl"), "wb") as fout:
            pickle.dump(state, fout)
    
    @classmethod
    def load(cls, load_dir):
        model_path = os.path.join(load_dir, "mlmodel.pkl")
        assert os.path.exists(model_path), f"MLModel path {model_path} does not exist"

        with open(model_path, "rb") as fin:
            model_data = pickle.load(fin)

        model = cls()
        model.__dict__.update(model_data)
        
        # Load models
        model_files = {
            "cluster_model": Clustering if hasattr(Clustering, 'load') else None,
            "matcher_model": Matcher if hasattr(Matcher, 'load') else None,
            "ranker_model": Ranker if hasattr(Ranker, 'load') else None,
        }
        
        for model_name, model_class in model_files.items():
            model_path = os.path.join(load_dir, model_name)
            # First check for model-specific save format
            if os.path.exists(model_path) and model_class is not None:
                setattr(model, model_name, model_class.load(model_path))
            else:
                print(f"Model {model_name} is not being loaded")
            
        emb_path = os.path.join(load_dir, f"fused_scores.npy")
        assert os.path.exists(emb_path), f"Expecting fused_scores to be in the path {emb_path}, but path doesnt exist"
        setattr(model, "fused_scores", np.load(emb_path, allow_pickle=True))

        label_emb_path = os.path.join(load_dir, "label_embeddings.npy")
        if os.path.exists(label_emb_path):
            setattr(model, "label_embeddings", np.load(label_emb_path, allow_pickle=True))
        else:
            setattr(model, "label_embeddings", None)

        return model
        
    def __str__(self):
        _str = f"Cluster Model: {"✔" if self.cluster_model is not None else "✖"}\n" \
                f"Matcher Model: {"✔" if self.matcher_model is not None else "✖"}\n" \
                f"Ranker Model: {"✔" if self.ranker_model is not None else "✖"}\n" 
        return _str
    
    def fused_predict(self, X, Z, C, alpha=0.5, batch_size=32768,
                      fusion: str = "lp_hinge", p: int = 3):
        """Batched matcher/ranker fusion.

        Parameters
        ----------
        X : array-like or sparse matrix
            Query feature matrix.
        Z : array-like or sparse matrix
            Label embedding matrix in the fused space.
        C : csr_matrix
            Cluster assignment matrix for projecting label scores.
        alpha : float, optional
            Interpolation coefficient between matcher and ranker scores.
        batch_size : int, optional
            Number of (query, label) pairs processed per ranker batch.
        fusion : {"geometric", "lp_hinge"}, optional
            Fusion strategy. "geometric" uses the geometric mean of matcher and
            ranker scores. "lp_hinge" performs an L-p interpolation with a
            hinge at zero.
        p : int, optional
            The ``p`` parameter used when ``fusion="lp_hinge"``.

        Returns
        -------
        csr_matrix
            Fused cluster score matrix of shape ``(n_queries, n_clusters)``.
        """
        N = X.shape[0]
        L_local = Z.shape[0]

        # --- matcher (local label-level) scores ---
        matcher_scores = csr_matrix(self.matcher_model.model.predict_proba(X), dtype=np.float32)

        # --- flatten mention-local label pairs ---
        rows_list, cols_list, matcher_flat = [], [], []
        for i in range(N):
            row = matcher_scores.getrow(i)
            local_idxs, scores = row.indices, row.data
            rows_list.extend([i] * len(local_idxs))
            cols_list.extend(local_idxs)
            matcher_flat.extend(scores)

        rows_list = np.array(rows_list, dtype=np.int32)
        cols_list = np.array(cols_list, dtype=np.int32)
        matcher_flat = np.array(matcher_flat, dtype=np.float32)

        ranker_score = np.ones(len(rows_list), dtype=np.float32)

        if self.ranker_model:
            # Transfer original mention embeddings to dense once
            X_dense = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
            global_idxs = np.array([self.local_to_global_idx[i] for i in cols_list])

            # print(self.ranker_model.model_dict)

            # detect loss type
            first_model = next(iter(self.ranker_model.model_dict.values()))
            is_hinge = first_model.config.get("type") == "sklearnsgdclassifier" and \
                    first_model.config.get("kwargs", {}).get("loss") == "hinge"

            num_pairs = len(rows_list)
            for start in range(0, num_pairs, batch_size):
                end = min(start + batch_size, num_pairs)

                b_rows = rows_list[start:end]
                b_cols = cols_list[start:end]
                b_global = global_idxs[start:end]

                # build concatenated embeddings only for this small batch
                X_part = X_dense[b_rows]
                Z_part = Z[b_cols]
                batch_inp = np.hstack([X_part, Z_part])

                scores_batch = np.ones(end - start, dtype=np.float32)
                # group by global label in this batch
                for g in np.unique(b_global):
                    mask = (b_global == g)
                    if g not in self.ranker_model.model_dict:
                        continue
                    mdl = self.ranker_model.model_dict[g]
                    sub_inp = batch_inp[mask]

                    try:
                        if is_hinge:
                            score = np.clip(expit(mdl.decision_function(sub_inp)), 1e-6, 1.0)
                        else:
                            score = np.clip(mdl.predict_proba(sub_inp)[:, 1], 1e-6, 1.0)
                        scores_batch[mask] = score
                        
                    except Exception:
                        scores_batch[mask] = 1.0

                ranker_score[start:end] = scores_batch
                
            else:
                alpha = 0
                
        self.alpha = alpha

        if fusion == "lp_hinge":
            fused = ((1 - self.alpha) * (matcher_flat ** p) +
                     self.alpha * (ranker_score ** p)) ** (1.0 / p)
            fused = np.maximum(fused, 0.0)
        else:
            fused = (matcher_flat ** (1 - self.alpha)) * (ranker_score ** self.alpha)

        # --- build local-label fused matrix ---
        entity_fused = csr_matrix((fused, (rows_list, cols_list)), shape=(N, L_local), dtype=np.float32)

        # --- project to clusters ---
        cluster_fused = entity_fused.dot(C)
        return csr_matrix(cluster_fused)
    
    def train(self, X_train, Y_train, Z_train, local_to_global, global_to_local):
        """
            X_train: X_processed
            Y_train, Y_binazier
            Z, Pifa embeddings
        """
        
        # --- Ensure Z is in fused space ---
        Z_train = normalize(Z_train, norm="l2", axis=1) 
        self.label_embeddings = Z_train
        del Z_train
        
        self.global_to_local_idx = global_to_local
        self.local_to_global_idx = np.array(local_to_global, dtype=int)
        
        del global_to_local
        
        print("Clustering")
        # Make the Clustering
        cluster_model = Clustering(self.clustering_config)
        cluster_model.train(Z=self.label_embeddings, 
                            local_to_global_idx=self.local_to_global_idx,
                            min_leaf_size=self.min_leaf_size,
                            max_leaf_size=self.max_leaf_size,
                            ) # Hardcoded
        
        if cluster_model.is_empty:
            return
        
        self.cluster_model = cluster_model
        del cluster_model
        gc.collect()
        
        # Retrieve C
        C = self.cluster_model.c_node
        cluster_labels = np.asarray(C.argmax(axis=1)).flatten()
        print(cluster_labels, type(cluster_labels))
        print(Counter(cluster_labels))
    
        print("Matcher")
        # Make the Matcher
        matcher_model = Matcher(self.matcher_config)  
        matcher_model.train(X_train, 
                            Y_train, 
                            local_to_global_idx=self.local_to_global_idx, 
                            global_to_local_idx=self.global_to_local_idx, 
                            C=C
                            )     
         
        self.matcher_model = matcher_model 
        del matcher_model
        gc.collect()
        
        train_ranker = self.ranker_every_layer or self.is_last_layer
        
        def _topb_sparse(P: np.ndarray, b: int) -> csr_matrix:
            # P: (n x K_or_L) dense proba; returns (n x K_or_L) CSR 0/1 mask of top-b per row
            n, K = P.shape
            b = max(1, min(b, K))
            idx_part = np.argpartition(P, K - b, axis=1)[:, -b:]
            rows = np.repeat(np.arange(n, dtype=np.int32), b)
            cols = idx_part.ravel()
            data = np.ones(n * b, dtype=np.int8)
            return csr_matrix((data, (rows, cols)), shape=(n, K))
        
        if train_ranker:
        
            M_TFN = self.matcher_model.m_node
            M_MAN = None
        
            if self.is_last_layer:
                P = self.matcher_model.predict_proba(X_train)
                M_MAN = _topb_sparse(P, b=5)
            
            print("Ranker")
            ranker_model = Ranker(self.ranker_config)
            ranker_model.train(X_train, 
                                Y_train, 
                                self.label_embeddings, 
                                M_TFN, 
                                M_MAN, 
                                cluster_labels,
                                local_to_global_idx=self.local_to_global_idx,
                                layer=self.layer,
                                n_label_workers=self.n_workers
                                )
            
            self.ranker_model = ranker_model
            del ranker_model
        else:
            self.ranker_model = None
            
        gc.collect()
        
        print("Fusing Scores")
        if not self.is_last_layer:
            cluster_scores = self.matcher_model.predict_proba(X_train)
            self.fused_scores = csr_matrix(np.maximum(cluster_scores, 0.0))
        else:
            I_L = sp_eye(self.label_embeddings.shape[0], format="csr", dtype=np.float32)
            self.fused_scores = self.fused_predict(X_train, self.label_embeddings, I_L, alpha=0.5, fusion="lp_hinge", p=3)
        
    def predict(self, X_query, beam_size: int = 5, topk: int | None = None, return_matrix: bool = False, 
            fusion: str = "lp_fusion", eps: float = 1e-6, alpha: float = 0.5, p: int = 3):
        # --- 1) Top-k clusters from matcher (cluster-level proba) ---
        cluster_scores = self.matcher_model.predict_proba(X_query)  # shape (n_queries, K)
        n, K = cluster_scores.shape

        C = self.cluster_model.c_node                                # shape (L, K) label->cluster (CSR)
        L = C.shape[0]
        assert K == C.shape[1], f"Matcher outputs K={K} clusters, but C has {C.shape[1]}."
        assert L == len(self.local_to_global_idx), "C rows must equal #local labels"

        k = min(beam_size, K)
        if k == 0:
            M = csr_matrix((n, 0))
            M.global_labels = np.array([], dtype=object)
            return M if return_matrix else ([np.array([])] * n, [np.array([], dtype=object)] * n)

        # pick top-k clusters per query
        idx_part = np.argpartition(cluster_scores, K - k, axis=1)[:, -k:]     # (n, k), unordered
        part = np.take_along_axis(cluster_scores, idx_part, axis=1)
        order = np.argsort(-part, axis=1)
        topk_clusters = np.take_along_axis(idx_part, order, axis=1)           # (n, k)

        # --- 2) Precompute mappings ---
        # cluster -> GLOBAL label ids (fast via CSC column indices)
        C_csc = C.tocsc()
        cluster_to_global = [self.local_to_global_idx[C_csc[:, c].indices] for c in range(K)]

        # label (local) -> cluster (single membership expected)
        label_cluster = np.asarray(C.argmax(axis=1)).ravel()

        g2l = self.global_to_local_idx
        Z = self.label_embeddings

        # --- 3) Build queries_per_label by expanding top-k clusters to all their labels ---
        queries_per_label = defaultdict(list)  # gid -> [query_idx,...]
        for qi in range(n):
            cand_global = np.unique(np.concatenate([cluster_to_global[c] for c in topk_clusters[qi]])) \
                        if k > 0 else np.array([], dtype=self.local_to_global_idx.dtype)
            for gid in cand_global:
                queries_per_label[int(gid)].append(qi)

        # --- 4) Batch ranker per label and fuse with the label's cluster score ---
        X_dense = X_query.toarray() if hasattr(X_query, "toarray") else np.asarray(X_query)
        model_dict = self.ranker_model.model_dict

        # detect hinge-style rankers (like in your earlier code)
        is_hinge = False
        if self.ranker_model and self.ranker_model.model_dict:
            first_model = next(iter(self.ranker_model.model_dict.values()))
            cfg = getattr(first_model, "config", {})
            is_hinge = (cfg.get("type") == "sklearnsgdclassifier" and
                        cfg.get("kwargs", {}).get("loss") == "hinge")

        results = [[] for _ in range(n)]  # per-query list of (gid, fused_score)

        for gid, q_indices in queries_per_label.items():
            li = g2l.get(gid)
            mdl = model_dict.get(gid)
            if li is None or not q_indices:
                continue

            # matcher score for (query, label) = cluster score of the label's cluster
            c = int(label_cluster[li])
            q_idx = np.asarray(q_indices, dtype=int)
            m = cluster_scores[q_idx, c]  # vector of length len(q_idx)

            # ranker score for (query, label)
            zvec = Z[li]
            Z_tiled = np.tile(zvec, (len(q_idx), 1))
            batch_inp = np.hstack([X_dense[q_idx], Z_tiled])

            if mdl is None:
                r = np.ones(len(q_idx), dtype=float)
            else:
                try:
                    if hasattr(mdl, "predict_proba") and not is_hinge:
                        r = mdl.predict_proba(batch_inp)[:, 1]
                    elif hasattr(mdl, "decision_function"):
                        r = expit(mdl.decision_function(batch_inp))
                    else:
                        r = np.ones(len(q_idx), dtype=float)
                except Exception:
                    r = np.ones(len(q_idx), dtype=float)

            # fuse (no reduction; label uses its single cluster)
            m = np.clip(m, eps, 1.0)
            r = np.clip(r, eps, 1.0)
            if fusion == "lp_fusion":
                fused = ((1 - alpha) * (m ** p) + alpha * (r ** p)) ** (1.0 / p)
                fused = np.maximum(fused, 0.0)
            else:  # "geometric"
                fused = (m ** (1 - alpha)) * (r ** alpha)

            for qi, sc in zip(q_idx, fused):
                results[qi].append((gid, float(sc)))

        def _sorted_truncate(pairs, top_k):
            # sort desc by score; tie-break by gid for determinism
            pairs.sort(key=lambda t: (-t[1], t[0]))
            if top_k:
                return pairs[:top_k]
            return pairs

        # --- 5) Pack results ---
        if return_matrix:
            
            if topk == 0:
                M = csr_matrix((n, 0))
                M.global_labels = np.array([], dtype=object)
                return M
                        
            truncated = [ _sorted_truncate(pairs=pairs, top_k=topk) for pairs in results ]
                    
            # 2) collect the (possibly reduced) set of global ids
            all_gids = sorted({gid for pairs in truncated for gid, _ in pairs})
            gid_to_col = {g: i for i, g in enumerate(all_gids)}

            # 3) build CSR triplets
            rows, cols, data = [], [], []
            for qi, pairs in enumerate(truncated):
                for gid, sc in pairs:
                    rows.append(qi)
                    cols.append(gid_to_col[gid])
                    data.append(sc)

            M = csr_matrix((data, (rows, cols)), shape=(n, len(all_gids)))
            M.global_labels = np.array(all_gids, dtype=object)
            return M
        
        scores_per_query, labels_per_query = [], []
        
        for pairs in results:
            pairs = _sorted_truncate(pairs=pairs, top_k=topk)                
            if not pairs:
                scores_per_query.append(np.array([], dtype=float))
                labels_per_query.append(np.array([], dtype=object))
            else:
                labels_per_query.append(np.array([g for g, _ in pairs], dtype=object))
                scores_per_query.append(np.array([s for _, s in pairs], dtype=float))
                
        return scores_per_query, labels_per_query
        
            
class HierarchicaMLModel():
    """Loops MLModel"""
    def __init__(self, 
                 clustering_config=None, 
                 matcher_config=None, 
                 ranker_config=None, 
                 min_leaf_size=20,
                 max_leaf_size=None,
                 cut_half_cluster=False,
                 ranker_every_layer=False,
                 n_workers=8,
                 layer=1):
        
        self.clustering_config = clustering_config
        self.matcher_config = matcher_config
        self.ranker_config = ranker_config
        self.min_leaf_size = min_leaf_size
        self.max_leaf_size = max_leaf_size
        self.cut_half_cluster = cut_half_cluster
        self.ranker_every_layer = ranker_every_layer
        self.n_workers = n_workers
        
        self._hmodel = []
        self._layer = layer
        self._child_index_map = None
        
        self.ml_dir = Path(tempfile.mkdtemp(prefix="ml_store_"))
        
    @property
    def hmodel(self):
        return self._hmodel
    
    @hmodel.setter
    def hmodel(self, value):
        self._hmodel = value

    @property
    def layers(self):
        return self._layer
    
    @layers.setter
    def layers(self, value):
        self._layer = value
        
    @property
    def child_index_map(self):
        return self._child_index_map
    
    @child_index_map.setter
    def child_index_map(self, value):
        self._child_index_map = value
        
    def save(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)
    
        state = self.__dict__.copy()

        for layer_idx in range(len(self.hmodel)):
            model_list = self.hmodel[layer_idx]
            layer_path = os.path.join(save_dir, f"layer_{layer_idx}")
            os.makedirs(layer_path, exist_ok=True)

            for model_idx, model in enumerate(model_list):
                if model is not None:
                    sub_model_path = os.path.join(layer_path, f"ml_{model_idx}")
                    os.makedirs(sub_model_path, exist_ok=True)

                    if hasattr(model, "save") and callable(model.save):
                        model.save(sub_model_path)
                    else:
                        try:
                            joblib.dump(model, f"{sub_model_path}.joblib")
                        except ImportError:
                            with open(f"{sub_model_path}.pkl", "wb") as f:
                                pickle.dump(model, f)

        state.pop("_hmodel", None)  # Correct key

        with open(os.path.join(save_dir, "hml.pkl"), "wb") as fout:
            pickle.dump(state, fout)
    
    @classmethod
    def load(cls, load_dir):
        xmodel_path = os.path.join(load_dir, "hml.pkl")
        assert os.path.exists(xmodel_path), f"Hierarchical ML Model path {xmodel_path} does not exist"
            
        with open(xmodel_path, "rb") as fin:
            model_data = pickle.load(fin)
            
        model = cls()
        model.__dict__.update(model_data)
            
        # Load layer directories
        layer_folders = []
        pattern = re.compile(r'^layer_\d+$')
        for entry in os.listdir(load_dir):
            full_path = os.path.join(load_dir, entry)
            if os.path.isdir(full_path) and pattern.match(entry):
                layer_folders.append(full_path)
            
        assert len(layer_folders) > 0, "No layer folders found"
            
        # Sort by numeric layer index
        layer_folders.sort(key=lambda x: int(re.search(r'layer_(\d+)', x).group(1)))
            
        # Load models from each layer
        hmodel = []
        for layer_path in layer_folders:
            layer_models = []
            for subentry in os.listdir(layer_path):
                sub_path = os.path.join(layer_path, subentry)
                if os.path.isdir(sub_path):  # If model is saved via model.save()
                    try:
                        model_obj = MLModel.load(sub_path)
                    except Exception as e:
                        raise RuntimeError(f"Failed to load MLModel from {sub_path}: {e}")
                elif subentry.endswith(".joblib"):
                    model_obj = joblib.load(sub_path)
                elif subentry.endswith(".pkl"):
                    with open(sub_path, "rb") as f:
                        model_obj = pickle.load(f)
                else:
                    continue  # Skip unexpected files

                layer_models.append(model_obj)

            assert len(layer_models) > 0, f"No models found in layer folder {layer_path}"
            hmodel.append(layer_models)

        setattr(model, "_hmodel", hmodel)

        return model
    
    def save_ml_temp(self, model, name):
        sub_dir =  self.ml_dir / str(name)
        sub_dir.mkdir(parents=True, exist_ok=True)
        model.save(str(sub_dir))
        return str(sub_dir)
            
    def prepare_layer(self, X, Y, Z, C, fused_scores, local_to_global_idx):
        """
        Returns a list of tuples, one per (non-empty) cluster c:
        (X_aug, Y_node, Z_node_aug, local_to_global_next, global_to_local_next, c)
        where `c` is the *cluster id* in the parent.
        """
        K_next = C.shape[1]
        inputs = []
        fused_dense = fused_scores.toarray() if hasattr(fused_scores, "toarray") else np.asarray(fused_scores)

        for c in range(K_next):
            local_idxs = C[:, c].nonzero()[0]
            if len(local_idxs) == 0:
                continue

            local_to_global_next = local_to_global_idx[local_idxs]
            global_to_local_next = {g: i for i, g in enumerate(local_to_global_next)}

            Y_sub = Y[:, local_idxs]
            mention_mask = (Y_sub.sum(axis=1).A1 > 0)

            X_node = X[mention_mask]
            Y_node = Y_sub[mention_mask, :]
            if X_node.shape[0] == 0:
                continue

            Z_node_base = Z[local_idxs, :]

            fused_c = fused_dense[mention_mask, :]
            feat_c = fused_c[:, c].ravel()
            feat_sum = fused_c.sum(axis=1).ravel()
            feat_max = fused_c.max(axis=1).ravel()
            feat_node = np.vstack([feat_c, feat_sum, feat_max]).T
            sparse_feats = csr_matrix(feat_node)

            X_aug = hstack([X_node, sparse_feats], format="csr")
            X_aug = normalize(X_aug, norm="l2", axis=1)

            try:
                X_node_dense = X_node.toarray()
            except Exception:
                X_node_dense = np.asarray(X_node)

            if X_node_dense.size == 0 or Z_node_base.size == 0:
                label_feats = np.zeros((Z_node_base.shape[0], 3), dtype=Z_node_base.dtype)
            else:
                scores_mat = X_node_dense.dot(Z_node_base.T)
                mean_per_label = scores_mat.mean(axis=0)
                sum_per_label = scores_mat.sum(axis=0)
                max_per_label = scores_mat.max(axis=0)
                label_feats = np.vstack([mean_per_label, sum_per_label, max_per_label]).T

            Z_node_aug = np.hstack([Z_node_base, label_feats])
            Z_node_aug = normalize(Z_node_aug, norm="l2", axis=1)

            inputs.append((X_aug, Y_node, Z_node_aug, local_to_global_next, global_to_local_next, c))

        return inputs
            
    def train(self, X_train, Y_train, Z_train, local_to_global, global_to_local):
        """
        Train multiple layers of MLModel; intermediate models are saved in a
        temporary folder which is automatically deleted at the end of training.
        """
        inputs = [(X_train, Y_train, Z_train, local_to_global, global_to_local)]

        # Use TemporaryDirectory to ensure cleanup
        with tempfile.TemporaryDirectory(prefix="ml_store_") as temp_dir:
            self.ml_dir = Path(temp_dir)
            self.hmodel = []
            child_index_map = []

            if self.ranker_every_layer:
                ranker_flag = True
            else:
                ranker_flag = False

            for layer in range(self.layers):
                next_inputs = []
                ml_list = []
                layer_failed = False
                is_last_layer = False
                cluster_map_for_layer = []

                # Optional: adjust cluster size
                if self.cut_half_cluster and layer > 0:
                    self.clustering_config["kwargs"]["n_clusters"] = max(
                        2, self.clustering_config["kwargs"]["n_clusters"] // 2
                    )
                
                for parent_idx, (X_node, Y_node, Z_node, local_to_label_node, global_to_local_node) in enumerate(inputs):
                    
                    if layer == self.layers - 1:
                        ranker_flag = True
                        is_last_layer = True
                    
                    ml = MLModel(
                        clustering_config=self.clustering_config,
                        matcher_config=self.matcher_config,
                        ranker_config=self.ranker_config,
                        min_leaf_size=self.min_leaf_size,
                        max_leaf_size=self.max_leaf_size,
                        ranker_every_layer= ranker_flag,
                        is_last_layer=is_last_layer,
                        layer=layer,
                        n_workers=self.n_workers,
                    )

                    ml.train(
                        X_train=X_node,
                        Y_train=Y_node,
                        Z_train=Z_node,
                        local_to_global=local_to_label_node,
                        global_to_local=global_to_local_node
                    )

                    if ml.is_empty:
                        layer_failed = True
                        break

                    C = ml.cluster_model.c_node
                    fused_scores = ml.fused_scores

                    # Prepare inputs for next layer
                    raw_children = self.prepare_layer(
                        X=X_node,
                        Y=Y_node,
                        Z=Z_node,
                        C=C,
                        fused_scores=fused_scores,
                        local_to_global_idx=local_to_label_node
                    )
                    
                    cluster_to_child = {}
                    for child in raw_children:
                        *payload, c_id = child
                        child_abs_idx = len(next_inputs)
                        next_inputs.append(tuple(payload))
                        cluster_to_child[int(c_id)] = child_abs_idx
                    
                    # Save model in temporary folder with unique name
                    ml_path = self.save_ml_temp(ml, f"{layer}_{uuid4()}")
                    del ml
                    ml_list.append(ml_path)
                    cluster_map_for_layer.append(cluster_to_child)

                if layer_failed:
                    break

                self.hmodel.append(ml_list)
                child_index_map.append(cluster_map_for_layer)

                inputs = next_inputs
                del ml_list
                gc.collect()

            # Reload all models for final hmodel
            new_hmodel_list = []
            for model_list in self.hmodel:
                loaded = [MLModel.load(p) for p in model_list]
                new_hmodel_list.append(loaded)
            self.hmodel = new_hmodel_list

            self._child_index_map = child_index_map
            
            return self.hmodel
        
    
    def predict(self, X_query, 
                    topk: int = 5, 
                    beam_size: int = 5, 
                    fusion: str = "lp_fusion", 
                    eps: float = 1e-9, 
                    alpha: float = 0.5,
                    topk_mode: str = "per_leaf",   # "per_leaf" | "global" | "none"
                    include_global_path: bool = True,
                    n_jobs=None): 
        """
        Same beam-search routing as before, but leaf rankers are called in batches:
        we collect all (leaf_idx, x_row, trail, qi) pairs first, then run one
        batched predict per leaf_idx and map results back per query/path.
        """
            
        def _select_topk_indices(scores: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
            """Return (idx_top_sorted, vals_sorted) of top-k by score (descending)."""
            if k <= 0:
                return np.array([], dtype=int), np.array([], dtype=float)
            k = min(k, scores.size)
            idx = np.argpartition(scores, scores.size - k)[-k:]
            vals = scores[idx]
            order = np.argsort(-vals)
            return idx[order], vals[order]
        
        def _norm_topk(k):
            return None if (k is None or k <= 0) else int(k)

        topk_norm = _norm_topk(topk)
        
        if self.hmodel is None:
            out = [{"query_index": i, "paths": [], "new_path": None} for i in range(X_query.shape[0])]
            n_labels = int(getattr(getattr(self, "label_embeddings", np.empty((0,))), "shape", [0])[0]) if getattr(self, "label_embeddings", None) is not None else 0
            return out, csr_matrix((X_query.shape[0], n_labels), dtype=float)

        n_layers = len(self.hmodel)
        n_queries = X_query.shape[0]
        
        paths_per_query = [[] for _ in range(n_queries)]
        
        per_query_scores = [defaultdict(float) for _ in range(n_queries)]
        
        # Collect all leaf work across all queries first
        # leaf_idx -> list of (qi, x_row, trail)
        pending_by_leaf: dict[int, list[tuple[int, csr_matrix, list]]] = defaultdict(list)

        print(f"Shape of the query: {X_query.shape[0]}")

        for qi in range(X_query.shape[0]):    
            # print(f"Query n: {qi}")
        
            x0 = X_query[qi:qi+1]
            beam = [(mi, x0, 0.0, []) for mi in range(len(self.hmodel[0]))]

            # Traverse all non-final layers
            for layer in range(n_layers - 1):
                ml_list = self.hmodel[layer]
                candidates = []

                for parent_model_idx, x_row, logscore, trail in beam:
                    ml = ml_list[parent_model_idx]
                    
                    cs = np.asarray(ml.matcher_model.predict_proba(x_row)).ravel()
                    if cs.size == 0 or np.all(cs <= 0):
                        continue

                    idx_top, vals_top = _select_topk_indices(cs, min(beam_size, cs.size))

                    parent_to_child = self.child_index_map[layer][parent_model_idx]

                    sum_cs = float(cs.sum())
                    max_cs = float(cs.max())
                        
                    for c, p_child in zip(idx_top, vals_top):
                        child_idx = parent_to_child.get(int(c))
                        if child_idx is None:
                            continue
                            
                        extra = csr_matrix([[float(p_child), sum_cs, max_cs]])
                        x_next = hstack([x_row, extra], format="csr")
                        x_next = normalize(x_next, norm="l2", axis=1)

                        logscore_next = logscore + float(np.log(max(p_child, eps)))
                            
                        trail_next = trail + [{
                            "layer": layer,
                            "parent_model_idx": int(parent_model_idx),
                            "chosen_cluster": int(c),
                            "matcher_prob": float(p_child),
                            "child_model_idx": int(child_idx),
                            "path_logscore": float(logscore_next)
                        }]
                        
                        candidates.append((int(child_idx), x_next, logscore_next, trail_next))

                if not candidates:
                    beam = []
                    break

                candidates.sort(key=lambda t: -t[2])
                beam = candidates[:beam_size]

            if beam:
                for child_idx, x_row, logscore, trail in beam:
                    pending_by_leaf[int(child_idx)].append((qi, x_row, trail))
                        
        # --- Batched leaf predictions per leaf model ---
        leaf_layer_models = self.hmodel[-1]
        for leaf_idx, items in pending_by_leaf.items():
            # Keep input order stable to map outputs back
            q_indices, xs, trails = zip(*items)  # lists aligned
            if len(xs) == 1:
                X_batch = xs[0]
            else:
                X_batch = vstack(xs, format="csr")

            leaf_ml = leaf_layer_models[leaf_idx]
            # This returns per-row lists; we don't change semantics, just batch
            scores_list, labels_list = leaf_ml.predict(
                X_batch,
                beam_size=100,
                fusion=fusion,
                alpha=alpha
            )

            for (qi, trail), labels, scores in zip(zip(q_indices, trails), labels_list, scores_list):
                
                # Apply per-leaf topk only if requested
                if topk_mode == "per_leaf" and topk_norm is not None:
                    labels = labels[:topk_norm]
                    scores = scores[:topk_norm]
                
                paths_per_query[qi].append({
                    "trail": trail,
                    "leaf_model_idx": int(leaf_idx),
                    "leaf_global_labels": labels.astype(np.int32, copy=False),
                    "scores": scores.astype(float, copy=False)
                })
                
                acc = per_query_scores[qi]
                for lid, sc in zip(labels, scores):
                    lid = int(lid)
                    sc = float(sc)
                    if sc > acc.get(lid, 0.0):
                        acc[lid] = sc


        # Build CSR (n_queries x n_labels). If we can't infer n_labels, fall back to max label + 1.
        try:
            n_labels_total = int(self.label_embeddings.shape[0])
        except Exception:
            max_lid = -1
            for acc in per_query_scores:
                if acc:
                    m = max(acc.keys())
                    if m > max_lid: max_lid = m
            n_labels_total = max_lid + 1 if max_lid >= 0 else 0

        # Build CSR, optionally applying global topk per query
        indptr = [0]
        indices = []
        data = []

        for qi in range(n_queries):
            acc = per_query_scores[qi]
            if not acc:
                indptr.append(len(indices))
                continue

            items = acc.items()
            if topk_mode == "global" and topk_norm is not None:
                items = heapq.nlargest(topk_norm, items, key=lambda kv: kv[1])

            lids, scs = zip(*items)
            lids = np.asarray(lids, dtype=np.int32)
            scs = np.asarray(scs, dtype=float)

            # sort descending for readability (optional)
            order = np.argsort(-scs)
            indices.extend(lids[order].tolist())
            data.extend(scs[order].tolist())
            indptr.append(len(indices))

        scores_csr = csr_matrix(
            (np.asarray(data, dtype=float),
            np.asarray(indices, dtype=np.int32),
            np.asarray(indptr, dtype=np.int32)),
            shape=(n_queries, n_labels_total),
            dtype=float
        )

        # Optional fused/global "new_path" built from the final CSR row (reflects chosen mode)
        new_paths = [None] * n_queries
        if include_global_path:
            for qi in range(n_queries):
                row = scores_csr.getrow(qi)
                if row.nnz == 0:
                    new_paths[qi] = {
                        "trail": [],
                        "leaf_model_idx": -1,
                        "source": f"global_from_scores_csr[{topk_mode}]",
                        "leaf_global_labels": np.array([], dtype=np.int32),
                        "scores": np.array([], dtype=float),
                    }
                    continue
                lbls = row.indices
                scs = row.data
                order = np.argsort(-scs)
                lbls = lbls[order].astype(np.int32, copy=False)
                scs = scs[order].astype(float, copy=False)
                new_paths[qi] = {
                    "trail": [],
                    "leaf_model_idx": -1,
                    "source": f"global_from_scores_csr[{topk_mode}]",
                    "leaf_global_labels": lbls,
                    "scores": scs,
                }

        out = [
            {
                "query_index": int(qi),
                "paths": paths_per_query[qi],
                "final_path": new_paths[qi] if include_global_path else None,
            }
            for qi in range(n_queries)
        ]
        return out, scores_csr