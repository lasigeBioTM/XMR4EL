import torch
import numpy as np
import torch.nn as nn

from torch.amp import autocast


class ReRanker:
    def __init__(self, embed_dim, hidden_dim=128, batch_size=256, alpha=0.0):
        """
        Batched reranker using cosine + neural scoring.

        Args:
            embed_dim (int): Dimension of embeddings
            hidden_dim (int): Hidden layer size for neural scoring
            batch_size (int): Neural reranking batch size
            alpha (float): Weight for cosine similarity vs neural score (0 = neural only, 1 = cosine only)
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.embed_dim = embed_dim
        self.batch_size = batch_size
        self.alpha = alpha

        self.neural_score = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        ).to(self.device)
        
        self.neural_score.eval()

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

    def _fast_cosine_similarity(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Computes cosine similarity between x and y.
        """
        x = torch.nn.functional.normalize(x, dim=-1)
        y = torch.nn.functional.normalize(y, dim=-1)
        return torch.matmul(x, y.T)

    def match(self, input_vec: np.ndarray, label_vecs: np.ndarray, top_k=10, candidates=100):
        """
        Single input reranking.

        Returns:
            Tuple of (top_indices, top_scores), both numpy arrays.
        """
        x = self._to_tensor(input_vec)
        y = self._to_tensor(label_vecs)

        with torch.no_grad():
            sim_scores = self._fast_cosine_similarity(x, y).squeeze(0)
            candidate_indices = torch.topk(sim_scores, min(candidates, y.size(0))).indices
            candidate_vecs = y[candidate_indices]
            x_expanded = x.expand_as(candidate_vecs)

            scores = []
            with autocast(self.device.type):
                for i in range(0, len(candidate_vecs), self.batch_size):
                    x_batch = x_expanded[i:i + self.batch_size]
                    y_batch = candidate_vecs[i:i + self.batch_size]
                    features = torch.cat([x_batch, y_batch], dim=1)
                    batch_scores = self.neural_score(features).squeeze(1)
                    scores.append(batch_scores)

            scores = torch.cat(scores)

            if self.alpha > 0:
                cosine_part = sim_scores[candidate_indices]
                final_scores = self.alpha * cosine_part + (1 - self.alpha) * scores
            else:
                final_scores = scores

            top_scores, top_indices = torch.topk(final_scores, min(top_k, final_scores.size(0)))
            return candidate_indices[top_indices].cpu().numpy(), top_scores.float().cpu().numpy()

    def match_batch(self, input_vecs: np.ndarray, label_vecs: np.ndarray, top_k=10, candidates=100):
        """
        Batch input reranking.

        Returns:
            List of (top_indices, top_scores) tuples for each input.
        """
        input_tensor = self._to_tensor(input_vecs)
        label_tensor = self._to_tensor(label_vecs)

        results = []

        with torch.no_grad():
            for x in input_tensor:
                sim_scores = self._fast_cosine_similarity(x.unsqueeze(0), label_tensor).squeeze(0)
                candidate_indices = torch.topk(sim_scores, min(candidates, label_tensor.size(0))).indices
                candidate_vecs = label_tensor[candidate_indices]
                x_expanded = x.unsqueeze(0).expand_as(candidate_vecs)

                scores = []
                with autocast(self.device.type):
                    for i in range(0, len(candidate_vecs), self.batch_size):
                        x_batch = x_expanded[i:i + self.batch_size]
                        y_batch = candidate_vecs[i:i + self.batch_size]
                        features = torch.cat([x_batch, y_batch], dim=1)
                        batch_scores = self.neural_score(features).squeeze(1)
                        scores.append(batch_scores)

                scores = torch.cat(scores)

                if self.alpha > 0:
                    cosine_part = sim_scores[candidate_indices]
                    final_scores = self.alpha * cosine_part + (1 - self.alpha) * scores
                else:
                    final_scores = scores

                top_scores, top_indices = torch.topk(final_scores, min(top_k, final_scores.size(0)))
                results.append((
                    candidate_indices[top_indices].cpu().numpy(),
                    top_scores.float().cpu().numpy()
                ))

        return results
