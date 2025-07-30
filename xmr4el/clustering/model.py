import os
import pickle
import joblib

import numpy as np

from xmr4el.clustering.train import ClusteringTrainer
from xmr4el.models.cluster_wrapper.clustering_model import ClusteringModel


class Clustering():
    """Matcher pipeline"""
    
    def __init__(self, 
                 clustering_config=None,
                 dtype=np.float32):
        
        self.clustering_config = clustering_config
        self.dtype = dtype
        
        self._Z_node = None
        self._C_node = None
        self._model = None

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
    
    def train(self, Z, min_leaf_size):
        Z, C, model = ClusteringTrainer.train(Z=Z, 
                                              config=self.clustering_config, 
                                              min_leaf_size=min_leaf_size, 
                                              dtype=self.dtype)
        
        self.z_node = Z
        self.c_node = C
        self.model = model
        