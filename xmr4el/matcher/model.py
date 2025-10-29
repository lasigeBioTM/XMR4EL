import os
import pickle
import joblib

import numpy as np

from typing import Any, Dict, List, Optional
from xmr4el.matcher.train import MatcherTrainer
from xmr4el.models.classifier_wrapper.classifier_model import ClassifierModel


class Matcher:
    """Matcher pipeline that encapsulates model training and inference."""

    def __init__(
        self,
    ) -> None:
        """Initialize the matcher pipeline.

        Parameters
        ----------
        matcher_config:
            Optional configuration dictionary for the matcher.
        dtype:
            Desired data type for internal numpy arrays.
        """

        self._M_node: Optional[np.ndarray] = None
        self._model: Optional[ClassifierModel] = None
        
    @property
    def m_node(self) -> Optional[np.ndarray]:
        """Return the binarized matching matrix."""
        return self._M_node
    
    @m_node.setter
    def m_node(self, value: np.ndarray) -> None:
        """Set the binarized matching matrix."""
        self._M_node = value
        
    @property
    def model(self) -> Optional[ClassifierModel]:
        """Return the trained classification model."""
        return self._model
    
    @model.setter
    def model(self, value: ClassifierModel) -> None:
        """Set the trained classification model."""
        self._model = value
        
    def save(self, save_dir: str) -> None:
        """Persist the matcher object to disk.

        Parameters
        ----------
        save_dir:
            Directory where the model and state will be saved.
        """

        os.makedirs(save_dir, exist_ok=True)

        state = self.__dict__.copy()
        model = self.model

        if model is not None:
            model_path = os.path.join(save_dir, "matcher")

            if hasattr(model, "save") and callable(model.save):
                model.save(model_path)
            else:
                joblib.dump(model, f"{model_path}.joblib")

            state.pop("_model", None)

        with open(os.path.join(save_dir, "matcher.pkl"), "wb") as fout:
            pickle.dump(state, fout)
            
    @classmethod
    def load(cls, load_dir: str) -> "Matcher":
        """Load a matcher object from disk.

        Parameters
        ----------
        load_dir:
            Directory containing the saved matcher state.

        Returns
        -------
        Matcher
            Reconstructed matcher instance.
        """
        
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
    
    
    def train(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        local_to_global_idx: List[int],
        global_to_local_idx: Dict[int, int],
        C: np.ndarray,
        matcher_config: Optional[Dict[str, Any]] = None,
        dtype: Any = np.float32,
    ) -> None:
        """Train the matcher model.

        Parameters
        ----------
        X:
            Feature matrix for documents.
        Y:
            Label matrix for documents.
        local_to_global_idx:
            Mapping of local label indices to global indices.
        global_to_local_idx:
            Mapping of global label indices to local indices.
        C:
            Matrix used to construct the matching graph.
        """

        _, _, M, model = MatcherTrainer.train(
            X=X,
            Y=Y,
            local_to_global_idx=local_to_global_idx,
            global_to_local_idx=global_to_local_idx,
            C=C,
            config=matcher_config,
            dtype=dtype,
        )
        
        self.m_node = M
        self.model = model
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict labels for the given feature matrix."""
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict label probabilities for the given feature matrix."""
        return self.model.predict_proba(X)
    
    def classes(self) -> np.ndarray:
        """Return the classes predicted by the classifier."""
        return self.model.classes()