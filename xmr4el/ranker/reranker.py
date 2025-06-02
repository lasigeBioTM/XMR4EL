import os
import torch
import logging

import numpy as np
import torch.nn as nn

from torch.amp import autocast


LOGGER = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


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

    def save(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        reranker_path = os.path.join(save_dir, "reranker.pt")
        torch.save({
            'model_state_dict': self.state_dict(),
            'embed_dim': self.embed_dim,
            'hidden_dim': self.hidden_dim,
            'batch_size': self.batch_size
        }, reranker_path)
    
    @classmethod
    def load(cls, load_dir):
        LOGGER.info(
            f"Loading MLP Reranker from {load_dir}"
        )
        reranker_path = os.path.join(load_dir, "reranker.pt")
        assert os.path.exists(
            reranker_path
        ), f"Reranker path {reranker_path} does not exist"
        
        checkpoint = torch.load(reranker_path, torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        model = cls(
            embed_dim=checkpoint['embed_dim'],
            hidden_dim=checkpoint['hidden_dim'],
            batch_size=checkpoint['batch_size']
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        return model

    def fit(self, X, Y, tree_node=None, num_epochs=10, learning_rate=1e-3):
        """
        Train the reranker considering hierarchical structure
        
        Args:
            X: Input embeddings (n_samples, embed_dim)
            Y: Cluster labels or class labels (n_samples,)
            tree_node: Current node in the hierarchy (for context)
            num_epochs: Number of training epochs
            learning_rate: Learning rate for optimizer
        """
        self.train()
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        criterion = nn.BCEWithLogitsLoss() # For binary relevance
        
        # Convert data to tensors
        X_tensor = self._to_tensor(X)
        Y_labels = torch.from_numpy(Y).to(self.device)
        
        # Create positive and negative pairs
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            
            # Create batches of positive and negative pairs
            pos_pairs, neg_pairs = self._create_global_pairs(X_tensor, Y_labels)
            
            for batch_pos, batch_neg in zip(self._batch_iterator(pos_pairs),
                                            self._batch_iterator(neg_pairs)):
                
                optimizer.zero_grad()
                
                # Process positive pairs
                pos_scores = self._score_pairs(batch_pos)
                pos_labels = torch.ones_like(pos_scores)
                
                # Process negative pairs
                neg_scores = self._score_pairs(batch_neg)
                neg_labels = torch.ones_like(neg_scores)
                
                # Combine and compute loss
                all_scores = torch.cat([pos_scores, neg_scores])
                all_labels = torch.cat([pos_labels, neg_labels])
                
                loss = criterion(all_scores, all_labels)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                
                print(epoch_loss)
                
            LOGGER.info(f"Epoch {epoch+1}, Loss: {epoch_loss/len(pos_pairs)}")
            
    def _create_global_pairs(self, X_tensor, Y_labels):
        """
        Create pairs without hierarchical information
        Returns:
            Tuple of (positive_pairs, negative_pairs)
        """
        pos_pairs = []
        neg_pairs = []
        
        Y_np = Y_labels.cpu().numpy()
        unique_labels = np.unique(Y_np)
        
        # NEW: Ensure we sample negatives from ALL classes evenly
        label_to_indices = {label: np.where(Y_np == label)[0] for label in unique_labels}
        
        for i, (x, label) in enumerate(zip(X_tensor, Y_np)):
            # Positives (same as before)
            same_class_idx = label_to_indices[label]
            for j in np.random.choice(same_class_idx, size=min(3, len(same_class_idx)-1), replace=False):
                if i != j:
                    pos_pairs.append((x, X_tensor[j]))
            
            # NEW: Better negative sampling using unique_labels
            neg_classes = [l for l in unique_labels if l != label]
            selected_negs = []
            
            # Sample 1-2 negatives per other class
            for neg_class in np.random.choice(neg_classes, size=min(2, len(neg_classes)), replace=False):
                selected_negs.extend(
                    np.random.choice(label_to_indices[neg_class], 
                                size=1, 
                                replace=False))
                
            for j in selected_negs:
                neg_pairs.append((x, X_tensor[j]))
        
        return pos_pairs, neg_pairs
            
    def _create_hierarchical_pairs(self, X, Y, tree_node):
        """
        Create training pairs considering hierarchical structure
    
        Returns:
            Tuple of (positive_pairs, negative_pairs)
        """
        pos_pairs = []
        neg_pairs = []
        
        unique_labels = np.unique(Y)
        
        for i, (x, label) in enumerate(zip(X, Y)):
            # Positive pairs: same label at current level
            same_label_idx = np.where(Y == label)[0]
            for j in same_label_idx:
                if i != j:
                    pos_pairs.append((x, X[j]))
                    
            # Negative pairs: different labels at current level
            diff_label_idx = np.where(Y != label)[0]
            for j in diff_label_idx[:5]:
                neg_pairs.append((x, X[j]))
            
            # If we have hierarchical info, add cross-level negatives, LOOK OUT
            if tree_node and tree_node.parent:
                # Get some examples from parent's other children
                sibling_examples = self._get_sibling_examples(tree_node)
                for sib_x in sibling_examples:
                    neg_pairs.append((x, sib_x))
                
        return pos_pairs, neg_pairs
    
    def _get_sibling_examples(self, tree_node):
        """
        Get examples from sibling nodes for hard negative mining
        """
        sibling_examples = []
        if tree_node.parent:
            for sibling in tree_node.parent.children.values():
                if sibling != tree_node and hasattr(sibling, 'concatenated_embeddings'):
                    sibling_examples.extend(sibling.concatenated_embeddings)
        return sibling_examples[:5] # Return up to 5 exampels
    
    def _score_pairs(self, pairs):
        """
        Score a batch of pairs
        """
        x_batch = torch.stack(p[0] for p in pairs)
        y_batch = torch.stack(p[1] for p in pairs)
        features = self._compute_features(x_batch, y_batch)
        return self.neural_score(features).squeeze()

    def _batch_iterator(self, pairs, batch_size=None):
        """
        Batch generator for pairs
        """
        batch_size = batch_size or self.batch_size
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i:i + batch_size]
            yield batch

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
            candidate_indices = torch.topk(sim_scores, min(candidates, y.size(0))).indices
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
