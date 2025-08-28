import os
import pickle

import numpy as np

from joblib import Memory

from xmr4el.models.classifier_wrapper.classifier_model import ClassifierModel
from xmr4el.ranker.train import RankerTrainer


class Ranker():
    
    def __init__(self, 
                 ranker_config=None, 
                 dtype=np.float32,
                 temp_dir='./temp'):
        
        self.ranker_config = ranker_config
        self.dtype = dtype
        
        self._ranker_models = None # Ranker Model
        
        # Configure joblib memory caching
        self.memory = Memory(temp_dir, verbose=0)
        self._cached_process_label = self.memory.cache(RankerTrainer.process_label)
    
    @property
    def model_dict(self):
        return self._ranker_models
    
    @model_dict.setter
    def model_dict(self, value):
        self._ranker_models = value
        
    def save(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)  # Ensure directory exists
        
        state = self.__dict__.copy()
            
        model_dict = self.model_dict

        ranker_idx = model_dict.keys()
        ranker_models = model_dict.values()
        
        for idx, model in zip(ranker_idx, ranker_models):
            idx_folder = os.path.join(save_dir, f"{idx}")
            model.save(idx_folder)
        
        state.pop("_ranker_models", None)
        
        with open(os.path.join(save_dir, "ranker.pkl"), "wb") as fout:
            pickle.dump(state, fout)
        
    @classmethod
    def load(cls, load_dir):
        """
        Load ranker state and model_dict from a directory.

        Args:
            load_dir (str): Path to the saved directory.
            model_class (class): Class with a .load(path) method to reconstruct saved models.
        """
        # Load the pickled state
        with open(os.path.join(load_dir, "ranker.pkl"), "rb") as fin:
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
                ranker_model = ClassifierModel.load(folder_path)
                model_dict[int(folder)] = ranker_model

        setattr(model, "_ranker_models", model_dict)
        return model
        
    def train(self, X, Y, Z, M_TFN, M_MAN, cluster_labels, local_to_global_idx, layer, n_label_workers=8):
        
        ranker_models = RankerTrainer.train(X=X,
                                                Y=Y,
                                                Z=Z,
                                                M_TFN=M_TFN, 
                                                M_MAN=M_MAN, 
                                                cluster_labels=cluster_labels,
                                                config=self.ranker_config,
                                                local_to_global_idx=local_to_global_idx,
                                                n_label_workers=n_label_workers,
                                                parallel_backend="threading")
        
        self.model_dict = ranker_models