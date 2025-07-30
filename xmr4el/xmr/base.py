import os
import re
import pickle
import joblib

import numpy as np

from scipy.sparse import csr_matrix, lil_matrix, hstack
from scipy.special import expit

from typing import Counter

from xmr4el.clustering.model import Clustering
from xmr4el.matcher.model import Matcher
from xmr4el.ranker.model import ReRanker


class MLModel():

    def __init__(self, 
                 clustering_config=None, 
                 matcher_config=None, 
                 reranker_config=None,
                 ):
        
        self.clustering_config = clustering_config
        self.matcher_config = matcher_config
        self.reranker_config = reranker_config
        
        self._global_label_ids = None
        self._cluster_model = None
        self._matcher_model = None
        self._reranker_model = None
        self._fused_scores = None
    
    @property
    def global_labels_idx(self):
        return self._global_label_ids
    
    @global_labels_idx.setter
    def global_labels_idx(self, value):
        self._global_label_ids = value
    
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
    
    def fused_predict(self, X, Z, alpha=0.5):
        
        # Get weak matcher scores for all clusters
        matcher_scores = self.matcher_model.model.predict_proba(X)
        matcher_scores = csr_matrix(matcher_scores)
        # print(matcher_scores)
        N, L = matcher_scores.shape
        # print(N, L)
        fused_scores = lil_matrix((N, L))
        
        for i in range(N):
            row = matcher_scores.getrow(i)
            top_label_indices = row.indices
            top_label_scores = row.data
            
            mention_emb = X[i].toarray().ravel()
           
            for j, label_idx in enumerate(top_label_indices):
                matcher_score = top_label_scores[j]
               
                if label_idx in self.reranker_model.model_dict:
                    # Create reranker input: concat(mention_emb, label_emb)
                    label_emb = Z[label_idx]
                    input_vec = np.hstack([mention_emb, label_emb]).reshape(1, -1)
                   
                    model = self.reranker_model.model_dict[label_idx]
                    score = model.decision_function(input_vec)
                    prob = expit(score)[0]
                    reranker_score = np.clip(prob, 1e-6, 1.0)
                else:
                    reranker_score = 0
                
                # Lp-Hinge
                fused = (matcher_score ** (1 - alpha)) * (reranker_score ** alpha)
                fused_scores[i, label_idx] = fused
                
        return fused_scores.tocsr()
        
    def train(self, X_train, Y_train, Z_train, local_label_indices, global_to_local_indices):
        """
            X_train: X_processed
            Y_train, Y_binazier
            Z, Pifa embeddings
        """
        
        self.global_labels_idx = global_to_local_indices
        # label_indices = np.array(x for x in range(label_indices.shape[0]))
        
        print("Clustering")
        # Make the Clustering
        cluster_model = Clustering(self.clustering_config)
        cluster_model.train(Z_train, 
                            min_leaf_size=20
                            ) # Hardcoded
        
        self.cluster_model = cluster_model
        del cluster_model
        
        # Retrieve C
        C = self.cluster_model.c_node
        cluster_labels = self.cluster_model.model.labels()
        print(Counter(cluster_labels))
    
        print("Matcher")
        # Make the Matcher
        matcher_model = Matcher(self.matcher_config)  
        matcher_model.train(X_train, 
                            Y_train, 
                            local_label_indices=local_label_indices, 
                            global_to_local_indices=global_to_local_indices, 
                            C=C
                            )     
         
        self.matcher_model = matcher_model 
        del matcher_model
        
        M_TFN = self.matcher_model.m_node
        M_MAN = self.matcher_model.model.predict(X_train)
        
        print("Reranker")
        reranker_model = ReRanker(self.reranker_config)
        reranker_model.train(X_train, 
                             Y_train, 
                             Z_train, 
                             M_TFN, 
                             M_MAN, 
                             cluster_labels
                             )
        
        self.reranker_model = reranker_model
        del reranker_model
        
        print("Fusing Scores")
        fused_scores = self.fused_predict(X_train, Z_train)
        # print(fused_scores)
        self.fused_scores = fused_scores
        del fused_scores
        
        
class HierarchicaMLModel():
    """Loops MLModel"""
    def __init__(self, 
                 clustering_config=None, 
                 matcher_config=None, 
                 reranker_config=None, 
                 layer=1):
        
        self.clustering_config = clustering_config
        self.matcher_config = matcher_config
        self.reranker_config = reranker_config
        
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

        for layer_idx in range(self.layers):
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
            
    def prepare_layer(self, X, Y, Z, C, fused_scores):
        """
        Given current-layer data, split into per-cluster inputs for next layer:
          - X_node: mention embeddings under this cluster
          - Y_node: gold mention x label matrix under this cluster
          - Z_node: label embeddings under this cluster
          - fused_scores_node: fused mention-label scores under this cluster
        """
        K_next = C.shape[1]
        inputs = []
        
        for c in range(K_next):
            # 1. which labels are in cluster c
            label_idx = np.where(C[:, c] > 0)[0]
        
            # 2. restrict Y to those labels, then find which mentions have any of them
            Y_sub = Y[:, label_idx]
            mention_mask = (Y_sub.sum(axis=1).A1 > 0)            
            
            # 3. slice X and Y
            X_node = X[mention_mask]
            Y_node = Y_sub[mention_mask, :]
            Z_node = Z[label_idx, :]
            
            # 4. Compute fused scores features
            fused_c = fused_scores[mention_mask, :].toarray()
            
            feat_c = fused_c[:, c].ravel()    
            feat_sum = fused_c.sum(axis=1).ravel()
            feat_max = fused_c.max(axis=1).ravel()

            
            feat_node = np.vstack([feat_c, feat_sum, feat_max]).T
            sparse_feats = csr_matrix(feat_node)
            
            X_aug = hstack([X_node, sparse_feats], format="csr") # Combine the two 
            
            global_to_local = {global_idx: local_idx for local_idx, global_idx in enumerate(label_idx)}
            inputs.append((c, X_aug, Y_node, Z_node, label_idx, global_to_local))
            
        return inputs
        
    def train(self, X_train, Y_train, Z_train, local_label_indices, global_to_local_indices):
        """
        Train multiple layers of MLModel; after each layer, prepare data for the next.
        """
        
        inputs = [(X_train, Y_train, Z_train, local_label_indices, global_to_local_indices)]
        
        for _ in range(self.layers):
            
            next_inputs = []
            ml_list = []
            
            for X_node, Y_node, Z_node, local_label_indices, global_to_local_indices in inputs:
                
                ml = MLModel(clustering_config=self.clustering_config, 
                             matcher_config=self.matcher_config,
                             reranker_config=self.reranker_config
                            )
            
                ml.train(X_train=X_node, 
                         Y_train=Y_node, 
                         Z_train=Z_node, 
                         local_label_indices=local_label_indices,
                         global_to_local_indices=global_to_local_indices
                         )
                
                ml_list.append(ml)
                
                C = ml.cluster_model.c_node # Cluster mentions
                fused_scores = ml.fused_scores
            
                child_inputs = self.prepare_layer(X=X_node, 
                                                  Y=Y_node, 
                                                  Z=Z_node, 
                                                  C=C, 
                                                  fused_scores=fused_scores
                                                 )
                
                next_inputs.extend(child_inputs)
            
            self.hmodel.append(ml_list)
            # reformat inputs for next iteration: drop cluster id and label_idx
            inputs = [(X_node, Y_node, Z_node, local_label_indices, global_to_local_indices) for _, X_node, Y_node, Z_node, local_label_indices, global_to_local_indices in next_inputs] 
        
        return self.hmodel