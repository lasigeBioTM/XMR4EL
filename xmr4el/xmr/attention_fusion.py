import torch
from torch import nn

class AttentionFusion(nn.Module):
    def __init__(self, tfidf_dim, pifa_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(tfidf_dim + pifa_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
            nn.Softmax(dim=1)
        )
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def forward(self, tfidf, pifa):
        # Compute attention weighs
        tfidf = torch.tensor(tfidf.toarray(), dtype=torch.float).to(self.device)
        pifa = torch.tensor(pifa.toarray(), dtype=torch.float).to(self.device)
        
        combined = torch.cat([tfidf, pifa], dim=1)
        weights = self.attention(combined)
        
        # Apply weights
        tfidf_weighted = weights[:, 0:1] * tfidf
        pifa_weighted = weights[:, 1:2] * pifa
        
        return tfidf_weighted + pifa_weighted