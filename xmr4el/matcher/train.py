import numpy as np
from xmr4el.models.classifier_wrapper.classifier_model import ClassifierModel


class MatcherTrainer():
    
    # Y = Y_binazer
    @staticmethod
    def train(X, Y, local_to_global_idx, global_to_local_idx, C, config, dtype=np.float32):
    
        label_indices_local = [global_to_local_idx[g] for g in local_to_global_idx]
        Y_sub = Y[:, label_indices_local]
        
        keep_mask = np.asarray(Y_sub.sum(axis=1)).flatten() > 0
        
        X_node = X[keep_mask]
        Y_node = Y_sub[keep_mask]
        
        M_raw = Y_node @ C
        M = (M_raw > 0).astype(int)
        
        model = ClassifierModel.train(X_node, M, config, dtype, onevsrest=True)
        
        return X_node, Y_node, M, model