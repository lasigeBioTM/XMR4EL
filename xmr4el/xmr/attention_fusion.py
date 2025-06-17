import torch

from torch import nn


class AttentionFusion(nn.Module):
    """
    Attention-based fusion module that learns to dynamically combine two input tensors.
    
    This module computes attention weights to determine the optimal combination of
    input features X and Y. The attention mechanism consists of:
    - A two-layer neural network with ReLU activation
    - Softmax normalization to produce valid weights
    
    The fused output is computed as: w₁*X + w₂*Y where w₁+w₂=1
    
    Attributes:
        attention (nn.Sequential): The attention network that computes combination weights
        device (torch.device): The computation device (CUDA if available, else CPU)
    """
    
    def __init__(self, X_dim, Y_dim):
        """
        Initializes the AttentionFusion module.
        
        Args:
            X_dim (int): Dimensionality of the first input tensor
            Y_dim (int): Dimensionality of the second input tensor
        """
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
        """
        Computes the fused representation of input tensors X and Y.
        
        The fusion process:
        1. Ensures inputs are on the correct device
        2. Concatenates inputs along feature dimension
        3. Computes attention weights
        4. Applies weights to input tensors
        5. Returns weighted combination
        
        Args:
            X (torch.Tensor): First input tensor of shape (batch_size, X_dim)
            Y (torch.Tensor): Second input tensor of shape (batch_size, Y_dim)
            
        Returns:
            torch.Tensor: Fused output tensor of shape (batch_size, X_dim) or 
                        (batch_size, Y_dim) [whichever matches input dimensions]
                        
        Note:
            - Input tensors must have matching batch dimensions
            - Automatically handles device transfer if needed
            - Uses non-blocking transfers when moving to GPU
        """    
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