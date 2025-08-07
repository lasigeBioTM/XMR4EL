import os
import gc
import psutil
import pickle

import numpy as np

from scipy.sparse import csr_matrix

from joblib import Memory

from sklearn.preprocessing import normalize

from xmr4el.models.classifier_wrapper.classifier_model import ClassifierModel
from xmr4el.ranker.train import ReRankerTrainer


class ReRanker():
    
    def __init__(self, 
                 reranker_config=None, 
                 dtype=np.float32,
                 temp_dir='./temp'):
        
        self.reranker_config = reranker_config
        self.dtype = dtype
        
        self._reranker_models = None # Reranker Model
        
        # Configure joblib memory caching
        self.memory = Memory(temp_dir, verbose=0)
        self._cached_process_label = self.memory.cache(ReRankerTrainer.process_label)
    
    @property
    def model_dict(self):
        return self._reranker_models
    
    @model_dict.setter
    def model_dict(self, value):
        self._reranker_models = value
        
    def save(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)  # Ensure directory exists
        
        state = self.__dict__.copy()
            
        model_dict = self.model_dict

        reranker_idx = model_dict.keys()
        reranker_models = model_dict.values()
        
        for idx, model in zip(reranker_idx, reranker_models):
            idx_folder = os.path.join(save_dir, f"{idx}")
            model.save(idx_folder)
        
        state.pop("_reranker_models", None)
        
        with open(os.path.join(save_dir, "reranker.pkl"), "wb") as fout:
            pickle.dump(state, fout)
        
    @classmethod
    def load(cls, load_dir):
        """
        Load reranker state and model_dict from a directory.

        Args:
            load_dir (str): Path to the saved directory.
            model_class (class): Class with a .load(path) method to reconstruct saved models.
        """
        # Load the pickled state
        with open(os.path.join(load_dir, "reranker.pkl"), "rb") as fin:
            state = pickle.load(fin)
        
        # Restore everything except model_dict (which we will re-load)
        state.pop("model_dict", None)
        
        model = cls()
        model.__dict__.update(state)

        # Load each individual model from its directory
        model_dict = {}
        for folder in os.listdir(load_dir):
            folder_path = os.path.join(load_dir, folder)
            if folder.isdigit():  # model indices are numeric (str)
                reranker_model = ClassifierModel.load(folder_path)
                model_dict[int(folder)] = reranker_model

        setattr(model, "_reranker_models", model_dict)
        return model
        
    def train(self, X, Y, Z, M_TFN, M_MAN, cluster_labels, local_to_global_idx, layer, n_label_workers=8):
        self.reranker_config["kwargs"]["n_jobs"] = 1
        
        reranker_models = ReRankerTrainer.train(X=X,
                                                Y=Y,
                                                Z=Z,
                                                M_TFN=M_TFN, 
                                                M_MAN=M_MAN, 
                                                cluster_labels=cluster_labels,
                                                local_to_global_idx=local_to_global_idx,
                                                config=self.reranker_config,
                                                layer=layer,
                                                n_label_workers=n_label_workers)
        self.model_dict = reranker_models