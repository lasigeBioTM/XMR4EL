import os
import pickle

import numpy as np

from joblib import Memory

from typing import Dict, Optional

from xmr4el.models.classifier_wrapper.classifier_model import ClassifierModel
from xmr4el.ranker.train import RankerTrainer


class Ranker:
    """Wrapper around a collection of per-label ranker models."""

    def __init__(
        self,
        ranker_config: Optional[Dict] = None,
        cur_config: Optional[Dict] = None,
        dtype: np.dtype = np.float32,
        temp_dir: str = "./temp",
    ) -> None:
        
        self.ranker_config = ranker_config
        self.cur_config = cur_config
        self.dtype = dtype
        
        self._ranker_models: Optional[Dict[int, ClassifierModel]] = None
        
        # Configure joblib memory caching
        self.memory = Memory(temp_dir, verbose=0)
        self._cached_process_label = self.memory.cache(RankerTrainer.process_label_incremental)
    
    @property
    def model_dict(self) -> Optional[Dict[int, ClassifierModel]]:
        """Return the dictionary of trained ranker models."""
        return self._ranker_models
    
    @model_dict.setter
    def model_dict(self, value: Dict[int, ClassifierModel]) -> None:
        """Set the internal ranker model dictionary."""
        self._ranker_models = value
        
    def save(self, save_dir: str) -> None:
        """Persist ranker state and models to ``save_dir``."""
        os.makedirs(save_dir, exist_ok=True)
        
        state = self.__dict__.copy()
            
        model_dict = self.model_dict or {}
        
        for idx, model in model_dict.items():
            idx_folder = os.path.join(save_dir, f"{idx}")
            model.save(idx_folder)
        
        state.pop("_ranker_models", None)
        state.pop("memory", None)
        state.pop("_cached_process_label", None)
        
        with open(os.path.join(save_dir, "ranker.pkl"), "wb") as fout:
            pickle.dump(state, fout)
        
    @classmethod
        # Load the pickled state
    def load(cls, load_dir: str) -> "Ranker":
        """Load ranker state and model dictionary from ``load_dir``."""
        with open(os.path.join(load_dir, "ranker.pkl"), "rb") as fin:
            state = pickle.load(fin)
        
        state.pop("model_dict", None)
        
        model = cls()
        model.__dict__.update(state)

        # Load each individual model from its directory
        model_dict: Dict[int, ClassifierModel] = {}
        for folder in os.listdir(load_dir):
            folder_path = os.path.join(load_dir, folder)
            if folder.isdigit():
                ranker_model = ClassifierModel.load(folder_path)
                model_dict[int(folder)] = ranker_model

        setattr(model, "_ranker_models", model_dict)
        return model
        
    def train(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        Z: np.ndarray,
        M_TFN: np.ndarray,
        M_MAN: Optional[np.ndarray],
        cluster_labels: np.ndarray,
        local_to_global_idx: np.ndarray,
        layer: int,
        n_label_workers: int = 8,
    ) -> None:
        """Train ranker models for each label cluster."""
        n_epochs = (self.ranker_config or {}).get("n_epochs", 3)
        
        ranker_models = RankerTrainer.train(
            X=X,
            Y=Y,
            Z=Z,
            M_TFN=M_TFN,
            M_MAN=M_MAN,
            cluster_labels=cluster_labels,
            config=self.ranker_config,
            cur_config=self.cur_config,
            local_to_global_idx=local_to_global_idx,
            n_label_workers=n_label_workers,
            parallel_backend="threading",
            n_epochs=n_epochs
        )

        self.model_dict = ranker_models