import torch

import numpy as np
import torch.nn as nn

from torch.amp import autocast


class ReRanker(nn.Module):
    def __init__(self, embed_dim, hidden_dim=128, batch_size=256):
        """
        Batched reranker using cosine + neural scoring.

        Args:
            embed_dim (int): Dimension of embeddings
            hidden_dim (int): Hidden layer size for neural scoring
            batch_size (int): Neural reranking batch size
            alpha (float): Weight for cosine similarity vs neural score (0 = neural only, 1 = cosine only)
        """
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.embed_dim = embed_dim
        self.batch_size = batch_size
        self.dtype = torch.float32

        # Neural Score: input is concat of x, y, x*y, |x-y|
        self.neural_score = nn.Sequential(
            nn.Linear(embed_dim * 4, hidden_dim), # Must be 4 here
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1)
        )
        
        # Learnable fusion of cosine + neural score
        self.fusion_layer = nn.Sequential(
            nn.Linear(2, 1),
            nn.Sigmoid()
        )
        
        self.to(self.device)

    def _to_tensor(self, array: np.ndarray) -> torch.Tensor:
        """
        Converts a numpy array to a float tensor on the correct device.
        """
        if not isinstance(array, np.ndarray):
            raise TypeError("Expected input to be a numpy.ndarray")
        tensor = torch.from_numpy(array).float().to(self.device)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        if tensor.shape[-1] != self.embed_dim:
            raise ValueError(f"Expected last dimension to be {self.embed_dim}, got {tensor.shape[-1]}")
        return tensor

    def _fast_cosine_similarity(self, x, y):
        """
        Computes cosine similarity between x and y.
        """
        x = torch.nn.functional.normalize(x, dim=-1)
        y = torch.nn.functional.normalize(y, dim=-1)
        return torch.matmul(x, y.T)

    def _compute_features(self, x, y):
        return torch.cat([x, y, x * y, torch.abs(x - y)], dim=1)
    
    def _normalize_scores(self, scores):
        return (scores - scores.mean()) / (scores.std() + 1e+6)
    
    def forward(self, input_vec, label_vecs, top_k=10, candidates=100):
        """
        Compute top-k reranked matches.
        Returns:
            Tuple of (top_indices, top_scores) as numpy arrays
        """
        x = self._to_tensor(input_vec)
        y = self._to_tensor(label_vecs)
        
        with torch.no_grad():
            sim_scores = self._fast_cosine_similarity(x, y).squeeze(0)
            candidate_indices = torch.topk(sim_scores, max(candidates, y.size(0))).indices
            candidate_vecs = y[candidate_indices]
            x_expanded = x.expand_as(candidate_vecs)
            
        scores = []
        
        with autocast(self.device.type):
            for i in range(0, len(candidate_vecs), self.batch_size):
                x_batch = x_expanded[i:i + self.batch_size]
                y_batch = candidate_vecs[i:i + self.batch_size]
                
                # Compute features
                features = self._compute_features(x_batch, y_batch)
                
                neural_scores = self.neural_score(features).squeeze(1)
                cosine_scores = self._normalize_scores(neural_scores)
                
                # Stack them for fusion
                combined = torch.stack([cosine_scores, neural_scores], dim=1)
                fused_scores = self.fusion_layer(combined).squeeze(1)
                
                scores.append(fused_scores)
                
        final_scores = torch.cat(scores)
        top_scores, top_indices = torch.topk(final_scores, min(top_k, final_scores.size(0)))
        
        return candidate_indices[top_indices].cpu().numpy(), top_scores.to(self.dtype).cpu().detach().numpy()
