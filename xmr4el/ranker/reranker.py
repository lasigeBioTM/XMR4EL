import torch
import torch.nn as nn
import numpy as np

from torch.cuda.amp import autocast

from xmr4el.gpu_availability import is_cuda_available


class XMRReranker():
    
    def __init__(self, embed_dim, hidden_dim=128):
        """
        Optimized combined cosine + neural reranker.
        
        Args:
            embed_dim: Dimension of input embeddings
            hidden_dim: Hidden layer size for neural scorer
        """
        
        self.device = 'cuda' if is_cuda_available() else 'cpu'
        self.embed_dim = embed_dim
        
        # Neural scoring component
        self.neural_score = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        ).to(self.device)
        
        # Similarity metric
        self.similarity_fn = self.__fast_cosine_similarity
        
    def __fast_cosine_similarity(x, y):
        """Optimized cosine similarity using PyTorch"""
        x_norm = torch.nn.functional.normalize(x, p=2, dim=-1)
        y_norm = torch.nn.functional.normalize(y, p=2, dim=-1)
        return torch.mm(x_norm, y_norm.T)
        
    def match(self, input_vec, label_vecs, top_k=10, candidates=100):
        """
        Two-stage retrieval with automatic device handling.
        
        Args:
            input_vec: (embed_dim,) numpy array or torch tensor (any device)
            label_vecs: (num_labels, embed_dim) numpy array or torch tensor (any device)
            top_k: Final number of predictions to return
            candidates: Number of candidates for neural reranking
            
        Returns:
            Tuple of (indices, scores) for top_k matches (always CPU numpy arrays)
        """
        # Convert inputs to tensors and match device
        input_tensor = torch.as_tensor(input_vec, dtype=torch.float32, device=self.device)
        label_tensor = torch.as_tensor(label_vecs, dtype=torch.float32, device=self.device)
        
        # Stage 1: Candidate selection
        with torch.no_grad():
            sim_scores = self.similarity_fn(input_tensor.unsqueeze(0), label_tensor)
            candidate_indices = torch.topk(
                sim_scores.squeeze(0), 
                min(candidates, len(label_tensor))
            ).indices
            
        # Stage 2: Neural reranking
        candidate_vecs = label_tensor[candidate_indices]
        expanded_input = input_tensor.unsqueeze(0).expand_as(candidate_vecs)
        
        # Score candidates
        with autocast(enabled=(self.device.type == 'cuda')):  # Mixed precision
            scores = self.neural_score(
                torch.cat([expanded_input, candidate_vecs], dim=1)
            ).squeeze(1)
        
        # Get top-k predictions and return as numpy
        top_scores, top_indices = torch.topk(scores, min(top_k, len(scores)))
        return (
            candidate_indices[top_indices].cpu().numpy(),
            top_scores.cpu().detach().numpy()
        )

    def _ensure_tensor(self, x):
        """Convert input to tensor on correct device"""
        if not torch.is_tensor(x):
            x = torch.FloatTensor(x)
        return x.to(self.device)

    def _compute_similarity(self, x, y):
        """Compute similarity with automatic device handling"""
        if self.similarity_fn.__module__.startswith("sklearn"):
            # For sklearn functions (requires CPU numpy)
            x_np = x.cpu().numpy()
            y_np = y.cpu().numpy()
            return torch.from_numpy(
                self.similarity_fn(x_np, y_np)
            ).to(x.device)
        return self.similarity_fn(x, y)