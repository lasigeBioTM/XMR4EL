import os
import pickle
import joblib

import numpy as np

from xmr4el.matcher.train import MatcherTrainer
from xmr4el.models.classifier_wrapper.classifier_model import ClassifierModel


class Matcher():
    """Matcher pipeline"""
    
    def __init__(self, 
                 matcher_config=None,
                 dtype=np.float32):
        
        self._X_node = None
        self._Y_node = None
        self._M_node = None
        self._model = None
        
        self.matcher_config = matcher_config
        self.dtype = dtype

    @property
    def x_node(self):
        return self._X_node
    
    @x_node.setter
    def x_node(self, value):
        self._X_node = value
        
    @property
    def y_node(self):
        return self._Y_node
    
    @y_node.setter
    def y_node(self, value):
        self._Y_node = value
        
    @property
    def m_node(self):
        return self._M_node
    
    @m_node.setter
    def m_node(self, value):
        self._M_node = value
        
    @property
    def model(self):
        return self._model
    
    @model.setter
    def model(self, value):
        self._model = value
        
    def save(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)

        state = self.__dict__.copy()
        model = self.model

        if model is not None:
            model_path = os.path.join(save_dir, "matcher")

            if hasattr(model, 'save') and callable(model.save):
                model.save(model_path)
            else:
                joblib.dump(model, f"{model_path}.joblib")

            # Correct: remove the _model attribute key
            state.pop("_model", None)

        with open(os.path.join(save_dir, "matcher.pkl"), "wb") as fout:
            pickle.dump(state, fout)
            
    @classmethod
    def load(cls, load_dir):
        matcher_path = os.path.join(load_dir, "matcher.pkl")
        assert os.path.exists(matcher_path), f"Matcher path {matcher_path} does not exist"

        with open(matcher_path, "rb") as fin:
            model_data = pickle.load(fin)

        model = cls()
        model.__dict__.update(model_data)
        
        model_path = os.path.join(load_dir, "matcher")
        matcher = ClassifierModel.load(model_path)
        setattr(model, "_model", matcher)
        
        return model
    
    
    def train(self, X, Y, local_label_indices, global_to_local_indices, C):
        X_node, Y_node, M, model = MatcherTrainer.train(X=X, 
                                                        Y=Y, 
                                                        local_label_indices=local_label_indices, 
                                                        global_to_local_indices=global_to_local_indices,
                                                        C=C, 
                                                        config=self.matcher_config, 
                                                        dtype=self.dtype)
        
        self.x_node = X_node
        self.y_node = Y_node
        self.m_node = M
        self.model = model
        