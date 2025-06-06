import faiss

import numpy as np

from scipy.sparse import csr_matrix


class CandidateRetrieval():
    
    def __init__(self):
        pass
    
    @staticmethod
    def convert_predictions_into_csr(predictions, num_labels=None):
        """
        Converts prediction results into sparse matrix format.
        
        Args:
            data: List of predictions in format [(label_index, score), ...]
            num_labels: Total number of possible labels
            
        Returns:
            csr_matrix: Sparse matrix of predictions
        """
        
        rows, cols, vals = [], [], []
        
        for row_idx, instance in enumerate(predictions):
            col, score = instance
            for idx, _ in enumerate(range(len(score))):
                rows.append(row_idx)
                cols.append(col[idx])
                vals.append(np.float32(score[idx]))

        if num_labels is None:
            num_labels = max(cols) + 1  # infer if not provided
        
        return csr_matrix((vals, (rows, cols)), 
                          shape=(len(predictions), num_labels), 
                          dtype=np.float32)
    
    
    def retrival(self, conc_input, conc_emb, candidates=100):
        # Using faiss
        index = faiss.IndexFlatIP(conc_emb.shape[1])
        index.add(conc_emb)
        (scores, indices) = index.search(conc_input, candidates)
        
        return (scores[0].astype(float), indices) # problem here added indices