import faiss

import numpy as np


class CandidateRetrieval():
    """
    A FAISS-based similarity search class for retrieving top-k candidates from embedding vectors.
    
    This class provides efficient nearest neighbor search capabilities using FAISS library,
    currently implementing exact search with IndexFlatIP (Inner Product) but designed to
    support approximate HNSW search in the future.
    
    HNSW Tested not better than regular IndexFlatIP
    
    Attributes:
        M (int): Number of connections per node for HNSW (higher = more accurate but slower).
                 Currently unused (reserved for future HNSW implementation).
        efConstruction (int): Controls index build time/accuracy for HNSW.
                              Currently unused (reserved for future HNSW implementation).
        efSearch (int): Controls search time/accuracy for HNSW.
                        Currently unused (reserved for future HNSW implementation).
    """
    def __init__(self, M=32, efConstruction=40, efSearch=16):
        """
        Initializes the CandidateRetrieval with HNSW parameters (currently unused).
        
        Args:
            M (int): Number of connections per node for HNSW. Defaults to 32.
            efConstruction (int): Construction time/accuracy trade-off. Defaults to 40.
            efSearch (int): Search time/accuracy trade-off. Defaults to 16.
        """
        
        self.M = M
        self.efConstruction = efConstruction
        self.efSearch = efSearch
    
    def retrival(self, conc_input, conc_emb, candidates=200):
        """
        Performs similarity search to find top-k candidates from embedding vectors.
        
        Currently implements exact search using FAISS's IndexFlatIP (Inner Product).
        The method rebuilds the index for each call, which is simple but could be
        optimized by storing the index if the embeddings don't change frequently.
        
        Args:
            conc_input (numpy.ndarray): Query vector(s) with shape:
                                      - [dim] for single query
                                      - [n_queries, dim] for multiple queries
            conc_emb (numpy.ndarray): Database embeddings to search against with shape
                                    [n_vectors, dim]
            candidates (int): Number of top results to return. Defaults to 100.
            
        Returns:
            tuple: Contains two numpy.ndarrays:
                - scores: Similarity scores of the top candidates
                - indices: Positions of the top candidates in conc_emb
                
            For single query input:
                - scores shape: [k]
                - indices shape: [k]
            For multiple queries:
                - scores shape: [n_queries, k]
                - indices shape: [n_queries, k]
                
        Note:
            - Automatically handles empty input embeddings by returning empty arrays
            - Clamps candidates to available number of vectors to prevent errors
            - Uses inner product (dot product) as similarity metric
        """
        # Could make an storage of conc_emb so it the construction doest repeat
        # Validate inputs
        if len(conc_emb) == 0:
            return np.array([]), np.array([])
        
        # Initialize HNSW index (rebuild every time)
        dim = conc_emb.shape[1]
        
        index = faiss.IndexFlatIP(dim)
        # index = faiss.IndexHNSWFlat(dim, self.M, faiss.METRIC_INNER_PRODUCT)
        # index.hnsw.efSearch = self.efSearch  # Control query-time accuracy
        index.add(conc_emb)  # Build index
        
        # Search (clamp candidates to avoid errors)
        candidates = min(candidates, len(conc_emb))
        scores, indices = index.search(conc_input, candidates)
        
        # Return first query's results (if single query)
        if conc_input.ndim == 1 or conc_input.shape[0] == 1:
            return scores[0].astype(float), indices[0]
        return scores.astype(float), indices  # Batch queries