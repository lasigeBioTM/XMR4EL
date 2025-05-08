import torch
import torch.nn as nn
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity

from xmr4el.gpu_availability import is_cuda_available


class Reranker():
    
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
        Two-stage retrieval: 
        1. Cosine similarity for candidate selection
        2. Neural reranking of top candidates
        
        Args:
            input_vec: (embed_dim,) numpy array or torch tensor
            label_vecs: (num_labels, embed_dim) numpy array or torch tensor
            top_k: Final number of predictions to return
            candidates: Number of candidates for neural reranking
            
        Returns:
            Tuple of (indices, scores) for top_k matches
        """
        # Ensure tensors
        if not torch.is_tensor(input_vec):
            input_vec = torch.FloatTensor(input_vec)
        if not torch.is_tensor(label_vecs):
            label_vecs = torch.FloatTensor(label_vecs)

        input_vec = input_vec.to(self.device)
        label_vecs = label_vecs.to(self.device)
        
        # print(input_vec.shape, label_vecs.shape)
        
        # Stage 1: Cosine similarity pre-filtering
        sim_scores = self.similarity_fn(input_vec, label_vecs)
        
        sim_scores = torch.tensor(sim_scores)
        
        candidate_indices = torch.topk(sim_scores, min(candidates, label_vecs.size(0))).indices
        candidate_indices = candidate_indices.squeeze()
        
        # Stage 2: Neural reranking
        candidate_vecs = label_vecs[candidate_indices]
        expanded_input = input_vec.expand(candidate_vecs.size(0), -1)
        
        # print(candidate_vecs.shape, expanded_input.shape)

        # print(candidate_vecs)
        # print(expanded_input, expanded_input.shape)

        # Ensure both tensors are 2D
        if expanded_input.dim() == 1:
            expanded_input = expanded_input.unsqueeze(0)  # shape: (1, embed_dim)
        if candidate_vecs.dim() == 3:
            candidate_vecs = candidate_vecs.squeeze(1)  # shape: (candidates, embed_dim)

        concat = torch.cat([expanded_input, candidate_vecs], dim=1)  # shape: (candidates, 2 * embed_dim)

        scores = self.neural_score(concat).squeeze(1)
        top_scores, top_indices = torch.topk(scores, min(top_k, scores.size(0)))

        # Returns numpy array
        return candidate_indices[top_indices].cpu().detach().numpy(), top_scores.cpu().detach().numpy()