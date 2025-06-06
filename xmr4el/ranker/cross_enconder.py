import torch

import numpy as np

from sentence_transformers import CrossEncoder
from multiprocessing import Pool, cpu_count


class CrossEncoderMP():
    
    def __init__(self, model_name='cross-encoder/ms-marco-TinyBERT-L-2-v2'):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CrossEncoder(model_name, device=self.device)
        
        # Disable multiprocessing if on GPU
        self.use_multiprocessing = (self.device.type == "cpu")
        self.num_workers = cpu_count() // 2 if self.use_multiprocessing else 1

    def _score_pairs(self, pairs_chunk):
        return self.model.predict(pairs_chunk)

    def predict(self, query_text, candidates, k=10):
        print(f"Candidate Length -> {len(candidates)}, Variants per Candidate -> {len(candidates[0])}")
        k = min(k, len(candidates))
        
        # Generate all (query, variant) pairs
        pairs = [(query_text, variant) for candidate in candidates for variant in candidate]

        # --- Scoring Strategy ---
        if self.use_multiprocessing:
            # CPU: Use multiprocessing
            chunk_size = max(1, len(pairs) // self.num_workers)
            chunks = [pairs[i:i + chunk_size] for i in range(0, len(pairs), chunk_size)]
            with Pool(self.num_workers) as pool:
                scores_chunks = pool.map(self._score_pairs, chunks)
            all_scores = np.concatenate(scores_chunks)
        else:
            # GPU: Single batch (no multiprocessing needed)
            all_scores = self.model.predict(pairs)

        # Aggregate max scores per candidate
        ptr = 0
        candidates_score = []
        for candidate in candidates:
            num_variants = len(candidate)
            candidates_score.append(np.max(all_scores[ptr:ptr + num_variants]))
            ptr += num_variants

        # Return top-k indices and scores
        top_k_indices = np.array(candidates_score).argsort()[-k:][::-1]
        return top_k_indices, np.array(candidates_score)[top_k_indices]