import torch

import numpy as np

from sentence_transformers import CrossEncoder
from multiprocessing import Pool, cpu_count


class CrossEncoderMP():
    """
    A multiprocessing-enabled wrapper for Sentence Transformers CrossEncoder that efficiently scores
    query-document pairs with optional GPU acceleration and CPU multiprocessing.
    
    Model Hardcoded for now
    
    This class handles batch processing of (query, candidates) pairs, automatically managing:
    - GPU vs CPU execution
    - Multiprocessing on CPU
    - Memory-efficient batch processing
    - Top-k candidate selection
    
    Attributes:
        device (torch.device): The computation device (CUDA if available, otherwise CPU)
        model (CrossEncoder): The underlying CrossEncoder model
        use_multiprocessing (bool): Whether to use multiprocessing (enabled only on CPU)
        num_workers (int): Number of worker processes to use (CPU cores/2 when multiprocessing)
    """
    def __init__(self, model_name='cross-encoder/ms-marco-TinyBERT-L-2-v2'):
        """
        Initializes the CrossEncoderMP with the specified model.
        
        Args:
            model_name (str): Name/path of the CrossEncoder model to load. Defaults to
                            'cross-encoder/ms-marco-TinyBERT-L-2-v2', a small but efficient model.
        """
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CrossEncoder(model_name, device=self.device)
        
        # Disable multiprocessing if on GPU
        self.use_multiprocessing = (self.device.type == "cpu")
        self.num_workers = cpu_count() // 2 if self.use_multiprocessing else 1

    def _score_pairs(self, pairs_chunk):
        """
        Internal method to score a chunk of text pairs.
        
        Args:
            pairs_chunk (list): List of (query, candidate) tuples to score
            
        Returns:
            numpy.ndarray: Array of similarity scores for the input pairs
        """
        return self.model.predict(pairs_chunk)

    def predict_batch(self, text_pairs, k=10):
        """
        Predicts top-k candidates for a batch of queries with their candidates.
        
        The method processes the input in three phases:
        1. Flattens the input structure for efficient batch processing
        2. Computes scores using either multiprocessing (CPU) or batched GPU processing
        3. Selects and returns the top-k candidates for each query
        
        Args:
            text_pairs (list): List of (query, candidates) tuples where:
                            - query (str): The search query
                            - candidates (list[str]): List of candidate documents/passages
            k (int): Number of top candidates to return for each query. Defaults to 10.
            
        Returns:
            list: List of (top_k_idx, top_k_scores) tuples for each input query where:
                - top_k_idx (numpy.ndarray): Indices of top candidates in original list
                - top_k_scores (numpy.ndarray): Corresponding scores of top candidates
        """
        # --- Phase 1: Flatten Input ---
        # Each text_pair = (query, list of candidates)
        flat_pairs = []
        pair_ranges = []  # Tracks start/end indices for each query's candidates
        ptr = 0

        for query, candidates in text_pairs:
            flat_pairs.extend((query, cand) for cand in candidates)
            pair_ranges.append((ptr, ptr + len(candidates)))
            ptr += len(candidates)

        k = min(k, max(end - start for start, end in pair_ranges))

        # --- Phase 2: Predict Scores ---
        if self.use_multiprocessing:
            chunk_size = max(1, len(flat_pairs) // (self.num_workers * 4))
            with Pool(self.num_workers) as pool:
                scores = np.concatenate([
                    pool.map(self._score_pairs, [
                        flat_pairs[i:i + chunk_size]
                        for i in range(0, len(flat_pairs), chunk_size)
                    ])
                ])
        else:
            scores = []
            batch_size = 2048
            for i in range(0, len(flat_pairs), batch_size):
                scores.extend(self.model.predict(flat_pairs[i:i + batch_size]))
            scores = np.array(scores)

        # --- Phase 3: Top-k Selection ---
        results = []
        for (start, end) in pair_ranges:
            query_scores = scores[start:end]

            top_k_idx = np.argpartition(query_scores, -k)[-k:]
            top_k_idx = top_k_idx[np.argsort(query_scores[top_k_idx])[::-1]]
            top_k_scores = query_scores[top_k_idx]

            results.append((top_k_idx, top_k_scores))

        return results

    def predict(self, text_pairs, k=10, batch_size=2048):
        """
        Memory-optimized batch prediction handler for large numbers of queries.
        
        Processes input in batches to prevent memory issues, especially on GPU.
        Automatically clears CUDA cache between batches when running on GPU.
        
        Args:
            text_pairs (list): List of (query, candidates) tuples where:
                            - query (str): The search query
                            - candidates (list[str]): List of candidate documents/passages
            k (int): Number of top candidates to return for each query. Defaults to 10.
            batch_size (int): Number of queries to process at once. Defaults to 2048.
            
        Returns:
            list: List of (top_k_idx, top_k_scores) tuples for each input query where:
                - top_k_idx (numpy.ndarray): Indices of top candidates in original list
                - top_k_scores (numpy.ndarray): Corresponding scores of top candidates
        """
        results = []
        
        for i in range(0, len(text_pairs), batch_size):
            print(f"Text Pair Number: {i}")
            batch_results = self.predict_batch(
                text_pairs[i:i + batch_size],
                k=k
            )
            results.extend(batch_results)
            
            if self.device.type == "cuda":
                torch.cuda.empty_cache()  # Prevent OOM
        
        return results