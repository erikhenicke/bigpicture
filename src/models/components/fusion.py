import torch
import torch.nn as nn
from abc import abstractmethod

class Fusion(nn.Module):
    def __init__(self, hr_dim, lr_dim, out_dim): 
        super().__init__()

        self.hr_dim = hr_dim
        self.lr_dim = lr_dim
        self.out_dim = out_dim

    @abstractmethod
    def forward(
        self,
        hr_features: torch.Tensor,
        lr_features: torch.Tensor,
    ) -> torch.Tensor:
        pass

class ConcatFusion(Fusion):
    def __init__(self, hr_dim, lr_dim, out_dim):
        super().__init__(hr_dim, lr_dim, out_dim)

        self.intermediate_dim = (hr_dim + lr_dim) // 2
    

        self.fusion = nn.Sequential(
            nn.Linear(hr_dim + lr_dim, self.intermediate_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.intermediate_dim, out_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

    def forward(self, lr_features, hr_features):
        concatenated = torch.cat([lr_features, hr_features], dim=1)
        return self.fusion(concatenated)