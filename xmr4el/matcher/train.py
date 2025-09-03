import numpy as np

from typing import Any, Dict, List, Tuple

from xmr4el.models.classifier_wrapper.classifier_model import ClassifierModel


class MatcherTrainer():
    
    # Y = Y_binazer
    @staticmethod
    def train(
        X: np.ndarray,
        Y: np.ndarray,
        local_to_global_idx: List[int],
        global_to_local_idx: Dict[int, int],
        C: np.ndarray,
        config: Dict[str, Any],
        dtype: Any = np.float32,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, ClassifierModel]:
        """Train the matcher classifier.

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
        config:
            Configuration dictionary for the classifier.
        dtype:
            Data type for internal numpy arrays.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray, ClassifierModel]
            Filtered feature matrix, filtered label matrix, matching matrix and the trained classifier model.
        """
    
        label_indices_local = [global_to_local_idx[g] for g in local_to_global_idx]
        Y_sub = Y[:, label_indices_local]
        
        keep_mask = np.asarray(Y_sub.sum(axis=1)).flatten() > 0
        
        X_node = X[keep_mask]
        Y_node = Y_sub[keep_mask]
        
        M_raw = Y_node @ C
        M = (M_raw > 0).astype(int)
        
        model = ClassifierModel.train(X_node, M, config, dtype, onevsrest=True)
        
        return X_node, Y_node, M, model
    