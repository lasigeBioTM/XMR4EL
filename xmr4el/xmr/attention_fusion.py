import torch
from torch import nn

class AttentionFusion(nn.Module):
    def __init__(self, X_dim, Y_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(X_dim + Y_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
            nn.Softmax(dim=1)
        )
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Batched
    def forward(self, X, Y):
        """X, Y are tensors"""        
        if X.device != self.device:
            X = X.to(self.device, non_blocking=True)
        if Y.device != self.device:
            Y = Y.to(self.device, non_blocking=True)
        
        combined = torch.cat([X, Y], dim=1)
        weights = self.attention(combined)
        weights = weights.unsqueeze(-1)
        
        # Apply weights
        fused = (weights[:, 0] * X) + (weights[:, 1] * Y)
        return fused