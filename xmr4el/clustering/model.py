from collections import defaultdict
import os
import pickle
import joblib

import numpy as np

from xmr4el.clustering.train import ClusteringTrainer
from xmr4el.models.cluster_wrapper.clustering_model import ClusteringModel


class Clustering():
    """Clustering pipeline"""
    
    def __init__(self, 
                 clustering_config=None,
                 dtype=np.float32):
        
        self.clustering_config = clustering_config
        self.dtype = dtype
        
        self._Z_node = None
        self._C_node = None
        self._model = None
        self._cluster_to_labels = None

    @property
    def z_node(self):
        return self._Z_node
    
    @z_node.setter
    def z_node(self, value):
        self._Z_node = value
        
    @property
    def c_node(self):
        return self._C_node
    
    @c_node.setter
    def c_node(self, value):
        self._C_node = value
        
    @property
    def model(self):
        return self._model
    
    @model.setter
    def model(self, value):
        self._model = value
        
    @property
    def cluster_to_labels(self):
        return self._cluster_to_labels
    
    @cluster_to_labels.setter
    def cluster_to_labels(self, value):
        self._cluster_to_labels = value
        
    @property
    def is_empty(self):
        return True if self.c_node is None else False
        
    def save(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)  # Ensure directory exists

        state = self.__dict__.copy()
        model = self.model

        if model is not None:
            model_path = os.path.join(save_dir, "clustering")

            # Save using model's own method or fallback to joblib
            if hasattr(model, 'save') and callable(model.save):
                model.save(model_path)
            else:
                joblib.dump(model, f"{model_path}.joblib")

            # Remove the _model from state to avoid pickling issues
            state.pop("_model", None)

        # Save remaining state
        with open(os.path.join(save_dir, "clustering.pkl"), "wb") as fout:
            pickle.dump(state, fout)
    
    @classmethod
    def load(cls, load_dir):
        cluster_path = os.path.join(load_dir, "clustering.pkl")
        assert os.path.exists(cluster_path), f"Clustering path {cluster_path} does not exist"

        with open(cluster_path, "rb") as fin:
            model_data = pickle.load(fin)

        model = cls()
        model.__dict__.update(model_data)
        
        model_path = os.path.join(load_dir, "clustering")
        matcher = ClusteringModel.load(model_path)
        setattr(model, "_model", matcher)
        
        return model
    
    def train(self, Z, local_to_global_idx, min_leaf_size, max_leaf_size):
        C_node, model = ClusteringTrainer.train(Z=Z, 
                                              config=self.clustering_config, 
                                              min_leaf_size=min_leaf_size, 
                                              max_leaf_size=max_leaf_size,
                                              dtype=self.dtype)
        
        if C_node is None or model is None:
            self.is_empty = True
            return
        
        self.z_node = Z
        self.c_node = C_node
        self.model = model
        
        # Build cluster_to_labels aligned with Z_node rows
        self.cluster_to_labels = defaultdict(list)
        for local_idx in range(C_node.shape[0]):  # iterate over actual rows in C_node
            cluster_vector = C_node[local_idx].toarray().ravel() if hasattr(C_node[local_idx], "toarray") else np.asarray(C_node[local_idx]).ravel()
            cid = np.argmax(cluster_vector)
            if local_idx >= len(local_to_global_idx):
                continue  # skip rows that do not exist in Z_node
            gidx = local_to_global_idx[local_idx]
            self.cluster_to_labels[cid].append(int(gidx))
        