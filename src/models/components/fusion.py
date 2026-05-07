import torch
import torch.nn as nn
import torch.nn.functional as F
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

    def forward(self, hr_features: torch.Tensor, lr_features: torch.Tensor) -> torch.Tensor:
        concatenated = torch.cat([hr_features, lr_features], dim=1)
        return self.fusion(concatenated)


class FiLM(nn.Module):
    def __init__(self, x_dim: int, z_dim: int, pre_fusion_relu: bool = False) -> None:
        super().__init__()
        self.pre_fusion_relu = pre_fusion_relu
        self.film_add = nn.Sequential(
                nn.Linear(z_dim, x_dim // 2),
                nn.ReLU(),
                nn.Linear(x_dim // 2, x_dim),
                *([nn.ReLU()] if pre_fusion_relu else []),
            )
        self.film_mul = nn.Sequential(
                nn.Linear(z_dim, x_dim // 2),
                nn.ReLU(),
                nn.Linear(x_dim // 2, x_dim),
                *([nn.ReLU()] if pre_fusion_relu else []),
            )

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        if self.pre_fusion_relu:
            x = F.relu(x)
        mul = self.film_mul(z)
        add = self.film_add(z)
        return mul * x + add


class FilmFusion(Fusion):
    def __init__(self, hr_dim, lr_dim, out_dim, pre_fusion_relu=False):
        super().__init__(hr_dim, lr_dim, out_dim)

        self.film = FiLM(hr_dim, lr_dim, pre_fusion_relu=pre_fusion_relu)

        self.projection = nn.Sequential(
            nn.Linear(hr_dim, out_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

    def forward(self, hr_features: torch.Tensor, lr_features: torch.Tensor) -> torch.Tensor:
        film_features = self.film(hr_features, lr_features)
        return self.projection(film_features)


class MultSimFusion(Fusion):
    def __init__(self, hr_dim, lr_dim, out_dim, pre_fusion_relu=True):
        super().__init__(hr_dim, lr_dim, out_dim)

        self.hr_projection = nn.Sequential(
            nn.Linear(hr_dim, out_dim),
            *([nn.ReLU()] if pre_fusion_relu else []),
            nn.Dropout(0.1),
        )

        self.lr_projection = nn.Sequential(
            nn.Linear(lr_dim, out_dim),
            *([nn.ReLU()] if pre_fusion_relu else []),
            nn.Dropout(0.1),
        )

    def forward(self, hr_features: torch.Tensor, lr_features: torch.Tensor) -> torch.Tensor:
        hr_projected = self.hr_projection(hr_features)
        lr_projected = self.lr_projection(lr_features)
        return torch.mul(hr_projected, lr_projected)


class GeoPriorFusion(Fusion):
    def __init__(self, hr_dim, lr_dim, out_dim, pre_fusion_relu=True):
        super().__init__(hr_dim, lr_dim, out_dim)

        self.hr_projection = nn.Sequential(
            nn.Linear(hr_dim, out_dim),
            *([nn.ReLU()] if pre_fusion_relu else []),
            nn.Dropout(0.1),
        )

        self.lr_projection = nn.Sequential(
            nn.Linear(lr_dim, out_dim),
            *([nn.ReLU()] if pre_fusion_relu else []),
            nn.Dropout(0.1),
        )

    def forward(self, hr_features: torch.Tensor, lr_features: torch.Tensor) -> torch.Tensor:
        hr_projected = self.hr_projection(hr_features)
        lr_projected = self.lr_projection(lr_features)
        return hr_projected + lr_projected