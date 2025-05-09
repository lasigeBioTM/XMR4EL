import torch
import torch.nn as nn
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity

from xmr4el.gpu_availability import is_cuda_available


class XMRReranker():
    
    def __init__(self, embed_dim, hidden_dim=128):
        """
        Combined cosine + neural reranker
        
        Args:
            embed_dim: Dimension of input embeddings
            hidden_dim: Hidden layer size for neural scorer
            device: 'cpu' or 'cuda'
        """
        
        gpu_availability = is_cuda_available()
        
        if gpu_availability:
            device = 'cuda'
        else:
            device = 'cpu'
        
        self.device = device
        self.embed_dim = embed_dim
        
        # Neural scoring component
        self.neural_score = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.neural_score.to(device)
        
        # Similarity metric
        self.similarity_fn = cosine_similarity
        
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
        input_tensor = self._ensure_tensor(input_vec)
        label_tensor = self._ensure_tensor(label_vecs)
        
        # Stage 1: Candidate selection
        with torch.no_grad():
            sim_scores = self._compute_similarity(input_tensor, label_tensor)
            candidate_indices = torch.topk(
                sim_scores, 
                min(candidates, label_tensor.size(0))
            ).indices.squeeze()
            
        # Stage 2: Neural reranking
        candidate_vecs = label_tensor[candidate_indices].to(self.device)
        expanded_input = input_tensor.expand_as(candidate_vecs).to(self.device)
        
        # Score candidates
        scores = self.neural_score(
            torch.cat([expanded_input, candidate_vecs], dim=1)
        ).squeeze(1)
        
        # Get top-k predictions
        top_scores, top_indices = torch.topk(
            scores, 
            min(top_k, scores.size(0))
        )
        
        # Return as numpy arrays on CPU
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