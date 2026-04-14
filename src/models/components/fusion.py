import torch
import torch.nn as nn
from abc import abstractmethod

from models.components.domain_relations import D3GRelation

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

    def forward(self, lr_features: torch.Tensor, hr_features: torch.Tensor) -> torch.Tensor:
        concatenated = torch.cat([lr_features, hr_features], dim=1)
        return self.fusion(concatenated)


class FiLM(nn.Module):
    def __init__(self, x_dim: int, z_dim: int) -> None:
        super().__init__()
        self.film_add = nn.Sequential(
                nn.Linear(z_dim, x_dim // 2),
                nn.ReLU(),
                nn.Linear(x_dim // 2, x_dim)
            )
        self.film_mul = nn.Sequential(
                nn.Linear(z_dim, x_dim // 2),
                nn.ReLU(),
                nn.Linear(x_dim // 2, x_dim)
            )

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        mul = self.film_mul(z)
        add = self.film_add(z)
        return mul * x + add


class FilmFusion(Fusion):
    def __init__(self, hr_dim, lr_dim, out_dim):
        super().__init__(hr_dim, lr_dim, out_dim)

        self.film = FiLM(hr_dim, lr_dim)

        self.fusion = nn.Sequential(
            self.film,
            nn.Linear(hr_dim, out_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

    def forward(self, lr_features: torch.Tensor, hr_features: torch.Tensor) -> torch.Tensor:
        return self.fusion(hr_features, lr_features)


class D3GFusion(Fusion):
    """TODO: Under construction"""
    def __init__(self, hr_dim, lr_dim, out_dim, beta, lr_encoder):
        super().__init__(hr_dim, lr_dim, out_dim)

        self.d3g_relation = D3GRelation(beta, 256, lr_encoder)
        self.fusion = nn.Sequential(
            nn.Linear(hr_dim + lr_dim, out_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

    def forward(self, lr_features: torch.Tensor, hr_features: torch.Tensor) -> torch.Tensor:
        d3g_weights = self.d3g_relation(hr_features, lr_features, torch.tensor(0., device=lr_features.device))
        weighted_hr = d3g_weights * hr_features.unsqueeze(1)  # batch_size x 1 x hr_dim
        concatenated = torch.cat([weighted_hr.squeeze(1), lr_features], dim=1)
        return self.fusion(concatenated)