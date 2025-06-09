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

    def predict_batch(self, query_texts, candidates_list, k=10):
        """Process multiple queries in one batch"""
        all_results = []
        
        for i, (query_text, candidates) in enumerate(zip(query_texts, candidates_list)):
            # Log progress periodically
            if i % self.log_interval == 0:
                print(f"Processing {i}/{len(query_texts)}...")
            
            # Generate pairs
            pairs = [(query_text, variant) for candidate in candidates for variant in candidate]
            
            # Score pairs
            if self.use_multiprocessing:
                chunk_size = max(1, len(pairs) // self.num_workers)
                chunks = [pairs[i:i + chunk_size] for i in range(0, len(pairs), chunk_size)]
                with Pool(self.num_workers) as pool:
                    scores_chunks = pool.map(self._score_pairs, chunks)
                all_scores = np.concatenate(scores_chunks)
            else:
                # Use larger batches on GPU
                batch_size = 512  # Adjust based on GPU memory
                all_scores = []
                for j in range(0, len(pairs), batch_size):
                    all_scores.extend(self.model.predict(pairs[j:j+batch_size]))
                all_scores = np.array(all_scores)
            
            # Aggregate scores
            ptr = 0
            candidates_score = []
            for candidate in candidates:
                num_variants = len(candidate)
                candidates_score.append(np.max(all_scores[ptr:ptr + num_variants]))
                ptr += num_variants
            
            # Get top-k
            top_k_indices = np.array(candidates_score).argsort()[-k:][::-1]
            all_results.append((top_k_indices, np.array(candidates_score)[top_k_indices]))
        
        return all_results
    
    def predict(self, query_texts, candidates_list, k=10, batch_size=32):
        """Memory-optimized version for large batches"""
        results = []
        
        # Process in smaller chunks to avoid OOM
        for i in range(0, len(query_texts), batch_size):
            batch_queries = query_texts[i:i+batch_size]
            batch_candidates = candidates_list[i:i+batch_size]
            
            # Process this batch
            batch_results = self.predict_batch(batch_queries, batch_candidates, k)
            results.extend(batch_results)
            
            # Clear memory
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
        
        return results