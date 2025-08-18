import os
import re
import gc
import pickle
import joblib

import numpy as np

from scipy.sparse import csr_matrix, lil_matrix, hstack
from scipy.special import expit

from sklearn.preprocessing import normalize

from typing import Counter

from joblib import Parallel, delayed

from xmr4el.clustering.model import Clustering
from xmr4el.matcher.model import Matcher
from xmr4el.ranker.model import ReRanker



class MLModel():

    def __init__(self, 
                 clustering_config=None, 
                 matcher_config=None, 
                 reranker_config=None,
                 min_leaf_size=20,
                 max_leaf_size=None,
                 layer=None,
                 n_workers=8,
                 ):
        
        self.clustering_config = clustering_config
        self.matcher_config = matcher_config
        self.reranker_config = reranker_config
        self.min_leaf_size = min_leaf_size
        self.max_leaf_size = max_leaf_size
        self.layer = layer
        self.n_workers = n_workers
        
        self._local_to_global_idx = None
        self._global_to_local_idx = None
        
        self._cluster_model = None
        self._matcher_model = None
        self._reranker_model = None
        self._fused_scores = None
    
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
    def reranker_model(self):
        return self._reranker_model
    
    @reranker_model.setter
    def reranker_model(self, value):
        self._reranker_model = value
        
    @property
    def fused_scores(self):
        return self._fused_scores
    
    @fused_scores.setter
    def fused_scores(self, value):
        self._fused_scores = value
        
    @property
    def is_empty(self):
        return True if self.cluster_model is None else False
    
    def save(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)  # Ensure directory exists

        state = self.__dict__.copy()

        # Mapping attribute names to their internal keys
        model_attrs = {
            "cluster_model": "_cluster_model",
            "matcher_model": "_matcher_model",
            "reranker_model": "_reranker_model"
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
            "reranker_model": ReRanker if hasattr(ReRanker, 'load') else None,
        }
        
        for model_name, model_class in model_files.items():
            model_path = os.path.join(load_dir, model_name)
            # First check for model-specific save format
            if os.path.exists(model_path) and model_class is not None:
                setattr(model, model_name, model_class.load(model_path))
            else:
                raise ValueError("Some Model is does not have an load() method")
            
        emb_path = os.path.join(load_dir, f"fused_scores.npy")
        assert os.path.exists(emb_path), f"Expecting fused_scores to be in the path {emb_path}, but path doesnt exist"
        setattr(model, "fused_scores", np.load(emb_path, allow_pickle=True))

        return model
        
    def __str__(self):
        _str = f"Cluster Model: {"✔" if self.cluster_model is not None else "✖"}\n" \
                f"Matcher Model: {"✔" if self.matcher_model is not None else "✖"}\n" \
                f"Reranker Model: {"✔" if self.reranker_model is not None else "✖"}\n" 
        return _str
    
    def fused_predict(self, X, Z, C, alpha=0.5):
        """
        Fully vectorized fused_predict: matcher + reranker fusion, avoiding per-mention loops.
        """
        N = X.shape[0]
        L_local = Z.shape[0]
        G = int(self.local_to_global_idx.max()) + 1

        # --- matcher scores ---
        matcher_scores = csr_matrix(self.matcher_model.model.predict_proba(X))

        # --- flatten mention-local label pairs ---
        rows_list, cols_list, matcher_flat = [], [], []
        for i in range(N):
            row = matcher_scores.getrow(i)
            local_idxs, scores = row.indices, row.data
            rows_list.extend([i] * len(local_idxs))
            cols_list.extend(local_idxs)
            matcher_flat.extend(scores)
        rows_list = np.array(rows_list)
        cols_list = np.array(cols_list)
        matcher_flat = np.array(matcher_flat)

        # --- reranker ---
        reranker_score = np.ones(len(rows_list), dtype=np.float32)
        if self.reranker_model:
            X_dense = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
            reranker_input = np.hstack([X_dense[rows_list], Z[cols_list]])

            # reranker type detection
            first_model = next(iter(self.reranker_model.model_dict.values()))
            is_hinge = first_model.config.get("type") == "sklearnsgdclassifier" and \
                    first_model.config.get("kwargs", {}).get("loss") == "hinge"

            # vectorized per-label batch scoring
            unique_labels = np.unique(cols_list)
            for local_idx in unique_labels:
                idx_mask = (cols_list == local_idx)
                global_idx = int(self.local_to_global_idx[local_idx])
                if global_idx not in self.reranker_model.model_dict:
                    continue
                mdl = self.reranker_model.model_dict[global_idx]
                batch_inp = reranker_input[idx_mask]

                try:
                    if is_hinge:
                        batch_scores = np.clip(expit(mdl.decision_function(batch_inp)), 1e-6, 1.0)
                    else:
                        batch_scores = np.clip(mdl.predict_proba(batch_inp)[:, 1], 1e-6, 1.0)
                    reranker_score[idx_mask] = batch_scores
                except Exception:
                    reranker_score[idx_mask] = 1.0

        # --- fuse matcher + reranker ---
        fused = (matcher_flat ** (1 - alpha)) * (reranker_score ** alpha)

        # --- build entity fused matrix with local label alignment ---
        entity_fused = csr_matrix((fused, (rows_list, cols_list)), shape=(N, L_local))

        # --- project to clusters ---
        cluster_fused = entity_fused.dot(C)  # shape (N, K), matches number of clusters
        return csr_matrix(cluster_fused)
        
    def train(self, X_train, Y_train, Z_train, local_to_global, global_to_local):
        """
            X_train: X_processed
            Y_train, Y_binazier
            Z, Pifa embeddings
        """
        
        self.global_to_local_idx = global_to_local
        self.local_to_global_idx = np.array(local_to_global, dtype=int)
        # label_indices = np.array(x for x in range(label_indices.shape[0]))
        
        print("Clustering")
        # Make the Clustering
        cluster_model = Clustering(self.clustering_config)
        cluster_model.train(Z=Z_train, 
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
        
        M_TFN = self.matcher_model.m_node
        M_MAN = self.matcher_model.model.predict(X_train)
        
        print("Reranker")
        reranker_model = ReRanker(self.reranker_config)
        reranker_model.train(X_train, 
                             Y_train, 
                             Z_train, 
                             M_TFN, 
                             M_MAN, 
                             cluster_labels,
                             local_to_global_idx=self.local_to_global_idx,
                             layer=self.layer,
                             n_label_workers=self.n_workers
                             )
        
        self.reranker_model = reranker_model
        del reranker_model
        gc.collect()
        
        # Dont predict fused scores if no layer ?
        
        print("Fusing Scores")
        fused_scores = self.fused_predict(X_train, Z_train, C)
        # print(fused_scores)
        self.fused_scores = fused_scores
        del fused_scores
        gc.collect()
        
        
class HierarchicaMLModel():
    """Loops MLModel"""
    def __init__(self, 
                 clustering_config=None, 
                 matcher_config=None, 
                 reranker_config=None, 
                 min_leaf_size=20,
                 max_leaf_size=None,
                 cut_half_cluster=False,
                 n_workers=8,
                 layer=1):
        
        self.clustering_config = clustering_config
        self.matcher_config = matcher_config
        self.reranker_config = reranker_config
        self.min_leaf_size = min_leaf_size
        self.max_leaf_size = max_leaf_size
        self.cut_half_cluster=cut_half_cluster
        self.n_workers = n_workers
        
        self._hmodel = []
        self._layer = layer
        
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
            
    def prepare_layer(self, X, Y, Z, C, fused_scores, local_to_global_idx):
        """
        Simplified prepare_layer that:
        - selects labels in each cluster
        - selects mentions relevant to that cluster (by Y)
        - builds X_aug = [X_node | feat_c, feat_sum, feat_max] (L2-normalized)
        - builds Z_node_aug = [Z_node_base | label_mean, label_sum, label_max]
        Returns list of (X_aug, Y_node, Z_node_aug, local_to_global_next, global_to_local_next)
        """
        K_next = C.shape[1]
        inputs = []

        # Make fused_scores dense once for indexing (works whether fused_scores is sparse or dense)
        fused_dense = fused_scores.toarray() if hasattr(fused_scores, "toarray") else np.asarray(fused_scores)

        for c in range(K_next):
            # 1. which labels are in cluster c (local indices relative to Z)
            local_idxs = C[:, c].nonzero()[0]
            if len(local_idxs) == 0:
                continue

            # 2. map them to *global* label IDs and build inverse map
            local_to_global_next = local_to_global_idx[local_idxs]
            global_to_local_next = {g: i for i, g in enumerate(local_to_global_next)}

            # 3. restrict Y to those labels, then find which mentions have any of them
            Y_sub = Y[:, local_idxs]
            mention_mask = (Y_sub.sum(axis=1).A1 > 0)

            # 4. slice X and Y
            X_node = X[mention_mask]
            Y_node = Y_sub[mention_mask, :]

            # if no mentions for this cluster, skip it
            if X_node.shape[0] == 0:
                continue

            # base label embeddings for this cluster (n_labels_in_cluster, emb_dim)
            Z_node_base = Z[local_idxs, :]

            # 5. Compute fused (per-mention) features and append to X_node
            fused_c = fused_dense[mention_mask, :]  # shape (n_mentions_node, K_next)
            feat_c = fused_c[:, c].ravel()
            feat_sum = fused_c.sum(axis=1).ravel()
            feat_max = fused_c.max(axis=1).ravel()
            feat_node = np.vstack([feat_c, feat_sum, feat_max]).T
            sparse_feats = csr_matrix(feat_node)

            X_aug = hstack([X_node, sparse_feats], format="csr")
            X_aug = normalize(X_aug, norm="l2", axis=1)

            # 6. Compute per-label aggregates (vectorized) and append to Z_node_base
            #    Scores matrix: (n_mentions_node, n_labels_in_cluster) = X_node_dense @ Z_node_base.T
            try:
                X_node_dense = X_node.toarray()
            except Exception:
                X_node_dense = np.asarray(X_node)

            if X_node_dense.size == 0 or Z_node_base.size == 0:
                label_feats = np.zeros((Z_node_base.shape[0], 3), dtype=Z_node_base.dtype)
            else:
                # vectorized dot-product
                scores_mat = X_node_dense.dot(Z_node_base.T)            # (n_mentions, n_labels)
                mean_per_label = scores_mat.mean(axis=0)               # (n_labels,)
                sum_per_label = scores_mat.sum(axis=0)
                max_per_label = scores_mat.max(axis=0)
                label_feats = np.vstack([mean_per_label, sum_per_label, max_per_label]).T  # (n_labels,3)

            Z_node_aug = np.hstack([Z_node_base, label_feats])  # (n_labels, emb_dim + 3)
            Z_node_aug = normalize(Z_node_aug, norm="l2", axis=1)
            # 7. append the tuple for this cluster
            inputs.append((X_aug, Y_node, Z_node_aug, local_to_global_next, global_to_local_next))

        return inputs
            
    def train(self, X_train, Y_train, Z_train, local_to_global, global_to_local):
        """
        Train multiple layers of MLModel; after each layer, prepare data for the next.
        """
        
        inputs = [(X_train, Y_train, Z_train, local_to_global, global_to_local)]
        
        for layer in range(self.layers):
            
            next_inputs = []
            ml_list = []
            layer_failed = False
            
            if self.cut_half_cluster and layer > 0:
                self.clustering_config["kwargs"]["n_clusters"] = int(max(2, self.clustering_config["kwargs"]["n_clusters"] / 2))
            
            for X_node, Y_node, Z_node, local_to_label_node, global_to_local_node in inputs:
                
                ml = MLModel(clustering_config=self.clustering_config, 
                             matcher_config=self.matcher_config,
                             reranker_config=self.reranker_config,
                             min_leaf_size=self.min_leaf_size,
                             max_leaf_size=self.max_leaf_size,
                             layer=layer,
                             n_workers=self.n_workers,
                            )
                
                ml._local_to_global_idx = local_to_global
            
                ml.train(X_train=X_node, 
                         Y_train=Y_node, 
                         Z_train=Z_node, 
                         local_to_global=local_to_label_node,
                         global_to_local=global_to_local_node
                         )
                
                if ml.is_empty:
                    layer_failed = True
                    break
                
                
                ml_list.append(ml)
                
                C = ml.cluster_model.c_node # Cluster mentions
                fused_scores = ml.fused_scores
            
                child_inputs = self.prepare_layer(X=X_node, 
                                                  Y=Y_node, 
                                                  Z=Z_node, 
                                                  C=C, 
                                                  fused_scores=fused_scores,
                                                  local_to_global_idx=local_to_label_node
                                                 )
                
                next_inputs.extend(child_inputs)
            
            if layer_failed:
                break
            
            self.hmodel.append(ml_list)
            
            del ml_list
            gc.collect()
            
            # reformat inputs for next iteration: drop cluster id and label_idx
            inputs = next_inputs
            
        return self.hmodel