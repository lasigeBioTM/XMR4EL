import numpy as np

from typing import Dict, Iterable, List, Sequence, Tuple

from sklearn.preprocessing import MultiLabelBinarizer, normalize

class LabelEmbeddingFactory():
    """Utility factory to build label embeddings."""
        
    @staticmethod
    def generate_label_matrix(label_to_indices: Dict[int, List[int]]) -> List[List[int]]:
        """Expand a mapping from labels to corpus indices into a label matrix."""
        label_to_matrix: List[List[int]] = []
        
        for key in list(label_to_indices.keys()):
            labels_ids = label_to_indices[key]
            for _ in labels_ids:
                label_to_matrix.append([key])
        
        return label_to_matrix    
        
    @staticmethod
    def label_binarizer(labels: Sequence[Sequence[int]]) -> Tuple[np.ndarray, np.ndarray]:
        """Binarize labels using ``MultiLabelBinarizer``."""
        mlb = MultiLabelBinarizer(sparse_output=True)
        Y = mlb.fit_transform(labels)
        return Y, mlb.classes_
    
    @staticmethod
    def generate_PIFA(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """Generate PIFA label embeddings from features and label matrix."""
        Z: List[np.ndarray] = []
        
        for label_idx in range(Y.shape[1]):
            row_indices = Y[:, label_idx].nonzero()[0]
            
            if len(row_indices) == 0:
                Z.append(np.zeros(X.shape[1]))
                
            else:
                positive_x = X[row_indices]
                v_ell = np.sum(positive_x, axis=0)
                v_ell = np.asarray(v_ell).ravel()
                z_ell = v_ell / (np.linalg.norm(v_ell) + 1e-10)
                Z.append(z_ell)

        Z = np.vstack(Z)
        Z = normalize(Z, norm="l2", axis=1)
        return Z
    
