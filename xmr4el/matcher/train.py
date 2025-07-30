import numpy as np
from xmr4el.models.classifier_wrapper.classifier_model import ClassifierModel


class MatcherTrainer():
    
    # Y = Y_binazer
    @staticmethod
    def train(X, Y, local_label_indices, global_to_local_indices, C, config, dtype=np.float32):
    
        label_indices_local = [global_to_local_indices[idx] for idx in local_label_indices]
        Y_sub = Y[:, label_indices_local]
        
        keep_mask = np.asarray(Y_sub.sum(axis=1)).flatten() > 0
        
        X_node = X[keep_mask]
        Y_node = Y_sub[keep_mask]
        
        M_raw = Y_node @ C
        M = (M_raw > 0).astype(int)
        
        model = ClassifierModel.train(X, M, config, dtype, onevsrest=True)
        
        return X_node, Y_node, M, model