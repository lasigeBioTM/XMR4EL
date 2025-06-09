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
        
        self.log_interval = 100

    def _score_pairs(self, pairs_chunk):
        return self.model.predict(pairs_chunk)

    def predict_batch(self, text_pairs, k=10):
        """Process pre-generated text_pairs in maximum throughput batches."""
        # --- Phase 1: Flatten and Score ---
        query_texts = [pair[0] for pair in text_pairs]
        candidates_list = [pair[1] for pair in text_pairs]
        
        k = min(k, len(candidates_list[0]))
        print(k)
        
        # Flatten all candidate variants
        flat_pairs = []
        pair_ranges = []  # Tracks (start_idx, end_idx) per query
        ptr = 0
        
        for query, candidates in text_pairs:
            variants = [(query, cand) for cand in candidates]
            flat_pairs.extend(variants)
            pair_ranges.append((ptr, ptr + len(variants)))
            ptr += len(variants)
        
        # --- Phase 2: Parallel Scoring ---
        if self.use_multiprocessing:
            # Dynamic chunk sizing (4 chunks per worker)
            chunk_size = max(1, len(flat_pairs) // (self.num_workers * 4))
            with Pool(self.num_workers) as pool:
                scores = np.concatenate(
                    pool.map(self._score_pairs,
                        [flat_pairs[i:i + chunk_size]
                            for i in range(0, len(flat_pairs), chunk_size)])
                )
        else:
            # GPU batch processing (larger batches for throughput)
            scores = []
            batch_size = 1024  # Modern GPUs handle this well
            for i in range(0, len(flat_pairs), batch_size):
                scores.extend(self.model.predict(flat_pairs[i:i + batch_size]))
            scores = np.array(scores)
        
        # --- Phase 3: Top-k Aggregation ---
        results = []
        for (start, end), candidates in zip(pair_ranges, candidates_list):
            query_scores = scores[start:end]
            
            # Group scores by candidate (handles multi-variant cases)
            candidate_scores = []
            score_ptr = 0
            for cand in candidates:
                num_variants = 1  # Adjust if candidates have sub-variants
                candidate_scores.append(np.max(query_scores[score_ptr:score_ptr + num_variants]))
                score_ptr += num_variants
            
            # Get top-k with argpartition (O(n) instead of O(nlogn))
            top_k_idx = np.argpartition(candidate_scores, -k)[-k:][::-1]
            results.append((top_k_idx, np.array(candidate_scores)[top_k_idx]))
        
        return results

    def predict(self, text_pairs, k=10, batch_size=512):
        """Memory-optimized outer batch handler."""
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