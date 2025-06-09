import faiss

import numpy as np


class CandidateRetrieval():
    
    def __init__(self, M=32, efConstruction=40, efSearch=16):
        # M = 32 # Number of connections per node (higher = more accurate but slower)
        # efConstruction = 40  # Controls index build time/accuracy
        # efSearch = 16  # Controls search time/accuracy
        
        self.M = M
        self.efConstruction = efConstruction
        self.efSearch = efSearch
    
    def retrival(self, conc_input, conc_emb, candidates=100):
        """
        Perform similarity search on dynamic embeddings.
        
        Args:
            conc_input: Query vector(s) (shape: [1, dim] or [n_queries, dim]).
            conc_emb: Current embeddings to index (shape: [n_vectors, dim]).
            candidates: Top-k results to return.
        
        Returns:
            Tuple of (scores, indices) for the nearest neighbors.
        """
        # Could make an storage of conc_emb so it the construction doest repeat
        
        # Validate inputs
        if len(conc_emb) == 0:
            return np.array([]), np.array([])
        
        # Initialize HNSW index (rebuild every time)
        dim = conc_emb.shape[1]
        
        index = faiss.IndexHNSWFlat(dim, self.M, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efSearch = self.efSearch  # Control query-time accuracy
        index.add(conc_emb)  # Build index
        
        # Search (clamp candidates to avoid errors)
        candidates = min(candidates, len(conc_emb))
        scores, indices = index.search(conc_input, candidates)
        
        # Return first query's results (if single query)
        if conc_input.ndim == 1 or conc_input.shape[0] == 1:
            return scores[0].astype(float), indices[0]
        return scores.astype(float), indices  # Batch queries