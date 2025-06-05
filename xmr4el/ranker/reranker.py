import os
import torch
import logging

import faiss # Later remove and place at the reranker

import numpy as np
import torch.nn as nn

from scipy.sparse import csr_matrix, issparse
from torch.amp import autocast
from xmr4el.models.classifier_wrapper.classifier_model import ClassifierModel


LOGGER = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

class ReRanker():
    
    def __init__(self, config=None):
        self.model = None
        self.config = config
    
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
    
    def candidate_retrival(self, kb_indices, conc_input, conc_emb, candidates=100):
        # Using faiss
        index = faiss.IndexFlatIP(conc_emb.shape[1])
        index.add(conc_emb)
        (scores, indices) = index.search(conc_input, candidates)
        kb_indices = np.array(kb_indices)   
        
        self.indices = indices[0]
        
        return (kb_indices[indices[0]], scores[0].astype(float)) # problem here added indices
    
    def generate_train(self):
        pass
    
    def train(self, X_train, Y_train):
        """Train on (query + doc) embeddings and binary labels."""
        self.model = ClassifierModel.train(X_train, Y_train, self.config)
    
    def rank(self, query_emb, candidates_emb):
        """
        Rerank candidates using the trained model.
        
        Args:
            query_embedding: Embedding of the input query (shape [embed_dim])
            candidate_embeddings: Embeddings of top-100 candidates (shape [100, embed_dim])
            
        Returns:
            (indices, scores): Sorted candidates and their relevance scores
        """
        # Concatenate query with each candidate
        X_rerank = np.concatenate([
            np.title(query_emb, (len(candidates_emb), 1)),
            candidates_emb
        ], axis=1)
        
        scores = self.model.predict_proba(X_rerank)[:, 1] # Probability of class 1 (relevant)
        
        sorted_indices = np.argsort(scores)[::-1]
        return sorted_indices, scores[sorted_indices]
        
        
        
        
