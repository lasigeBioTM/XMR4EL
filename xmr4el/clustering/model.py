from collections import defaultdict
import os
import pickle
import joblib

import numpy as np

from typing import Dict, List, Optional

from xmr4el.clustering.train import ClusteringTrainer
from xmr4el.models.cluster_wrapper.clustering_model import ClusteringModel


class Clustering:
    """Pipeline that encapsulates label clustering."""

    def __init__(
        self,
        clustering_config: Optional[Dict[str, any]] = None,
        dtype: any = np.float32,
    ) -> None:
        """Initialize the clustering pipeline."""
        
        self.clustering_config = clustering_config
        self.dtype = dtype
        
        self._Z_node: Optional[np.ndarray] = None
        self._C_node: Optional[np.ndarray] = None
        self._model: Optional[ClusteringModel] = None
        self._cluster_to_labels: Optional[Dict[int, List[int]]] = None

    @property
    def z_node(self) -> Optional[np.ndarray]:
        """Dense label embedding matrix."""
        return self._Z_node
    
    @z_node.setter
    def z_node(self, value: np.ndarray) -> None:
        """Set the label embedding matrix."""
        self._Z_node = value
        
    @property
    def c_node(self) -> Optional[np.ndarray]:
        """Sparse cluster assignment matrix."""
        return self._C_node
    
    @c_node.setter
    def c_node(self, value: np.ndarray) -> None:
        """Set the cluster assignment matrix."""
        self._C_node = value
        
    @property
    def model(self) -> Optional[ClusteringModel]:
        """Return the underlying clustering model."""
        return self._model
    
    @model.setter
    def model(self, value: ClusteringModel) -> None:
        """Set the underlying clustering model."""
        self._model = value
        
    @property
    def cluster_to_labels(self) -> Optional[Dict[int, List[int]]]:
        """Mapping from cluster id to list of label indices."""
        return self._cluster_to_labels
    
    @cluster_to_labels.setter
    def cluster_to_labels(self, value: Dict[int, List[int]]) -> None:
        """Set the cluster to label mapping."""
        self._cluster_to_labels = value
    
    @property
    def is_empty(self) -> bool:
        """Return ``True`` if clustering was not trained."""
        return self.c_node is None

    def save(self, save_dir: str) -> None:
        """Persist the clustering object to disk."""
        os.makedirs(save_dir, exist_ok=True)

        state = self.__dict__.copy()
        model = self.model

        if model is not None:
            model_path = os.path.join(save_dir, "clustering")

            if hasattr(model, "save") and callable(model.save):
                model.save(model_path)
            else:
                joblib.dump(model, f"{model_path}.joblib")

            state.pop("_model", None)

        # Save remaining state
        with open(os.path.join(save_dir, "clustering.pkl"), "wb") as fout:
            pickle.dump(state, fout)
    
    @classmethod
    def load(cls, load_dir: str) -> "Clustering":
        """Load a clustering object from ``load_dir``."""
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
    
    def train(
        self,
        Z: np.ndarray,
        local_to_global_idx: List[int],
        min_leaf_size: int,
        max_leaf_size: Optional[int],
    ) -> None:
        """Train the clustering model and populate cluster assignments."""
        
        C_node, model = ClusteringTrainer.train(
            Z=Z,
            config=self.clustering_config,
            min_leaf_size=min_leaf_size,
            max_leaf_size=max_leaf_size,
            dtype=self.dtype,
        )
        
        if C_node is None or model is None:
            return
        
        self.c_node = C_node
        self.model = model
        
        self.cluster_to_labels = defaultdict(list)
        for local_idx in range(C_node.shape[0]):
            cluster_vector = (
                C_node[local_idx].toarray().ravel()
                if hasattr(C_node[local_idx], "toarray")
                else np.asarray(C_node[local_idx]).ravel()
            )
            cid = int(np.argmax(cluster_vector))
            if local_idx >= len(local_to_global_idx):
                continue
            gidx = local_to_global_idx[local_idx]
            self.cluster_to_labels[cid].append(int(gidx))
        