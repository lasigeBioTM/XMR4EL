class CrossEncoderMP():
    
    def __init__(self, model_name='cross-encoder/ms-marco-TinyBERT-L-2-v2'):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CrossEncoder(model_name, device=self.device)
        
        # Disable multiprocessing if on GPU
        self.use_multiprocessing = (self.device.type == "cpu")
        self.num_workers = cpu_count() // 2 if self.use_multiprocessing else 1

    def _score_pairs(self, pairs_chunk):
        return self.model.predict(pairs_chunk)

    def predict_batch(self, text_pairs, k=10):
        """Predict top-k candidates for each query using CrossEncoder."""

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

            # Normalize using min-max (avoids divide-by-zero)
            min_s, max_s = query_scores.min(), query_scores.max()
            if max_s > min_s:
                query_scores = (query_scores - min_s) / (max_s - min_s)
            else:
                query_scores = np.zeros_like(query_scores)

            # Top-k selection
            top_k_idx = np.argpartition(query_scores, -k)[-k:]
            top_k_idx = top_k_idx[np.argsort(query_scores[top_k_idx])[::-1]]
            top_k_scores = query_scores[top_k_idx]

            results.append((top_k_idx, top_k_scores))

        return results

    def predict(self, text_pairs, k=10, batch_size=2048):
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