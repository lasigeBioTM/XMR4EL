import hashlib

import numpy as np

from scipy.sparse import csr_matrix


class PIFAEmbeddingFactory():
    
    def __init__(self, ngram_range=(1, 2), num_pos_buckets=10, n_features=2**20):
        """
        PIFA extractor with positional n-grams and feature hashing.
        
        Args:
            ngram_range: tuple (min_n, max_n), n-gram sizes to extract.
            num_pos_buckets: int, number of positional buckets per document.
            n_features: int, size of hashed feature space (power of 2 recommended).
        """
        self.ngram_range = ngram_range
        self.num_pos_buckets = num_pos_buckets
        self.n_features = n_features
        
    def _hash_features(self, n_gram, pos_bucket):
        """
        Hash an (ngram, position bucket) pair to an integer feature index.
        """
        combined = "_".join(n_gram) + f"_pos{pos_bucket}"
        h = int(hashlib.md5(combined.encode('utf-8')).hexdigest(), 16)
        return h % self.n_features
    
    def _tokenize(self, text):
        """
        Simple whitespace tokenizer; can be replaced by more sophisticated tech.
        """
        return text.lower().split()
    
    def transform(self, texts):
        """
        Extract PIFA features from a list of texts.
        
        Args:
            texts: list of strings, each representing concatenated entity phrases.
        
        Returns:
            csr_matrix of shape (len(texts), n_features) with PIFA features counts.
        """
        indptr = [0]
        indices = []
        data = []
        
        for text in texts:
            tokens = self._tokenize(text) # joao
            lenght = len(tokens) # 4
            bucket_size = max(1, lenght // self.num_pos_buckets)
            
            feat_counts = {}
            
            for n in range(self.ngram_range[0], self.ngram_range[1] + 1):
                for i in range(lenght - n + 1): # 4, 3, 2
                    ngram = tuple(tokens[i:i+n])
                    pos_bucket = min(i // bucket_size, self.num_pos_buckets - 1)
                    feat_idx = self._hash_features(ngram, pos_bucket)

                    feat_counts[feat_idx] = feat_counts.get(feat_idx, 0) + 1
            
            indices.extend(feat_counts.keys())
            data.extend(feat_counts.values())
            indptr.append(len(indices))
            
        X = csr_matrix((data, indices, indptr), shape=(len(texts), self.n_features), dtype=np.float32)
        return X    
    