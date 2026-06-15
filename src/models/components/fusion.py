from pathlib import Path
from typing import Literal, Optional

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import abstractmethod

# Repo-root-relative default for the FMoW metadata CSV the class prior is read from.
# fusion.py lives at src/models/components/, so parents[3] is the repo root.
DEFAULT_PRIOR_METADATA = Path(__file__).resolve().parents[3] / "data" / "rgb_metadata_extended.csv"


def compute_class_prior(metadata_path: Path, split: str, num_classes: int) -> torch.Tensor:
    """Per-class frequency vector P(y) over ``split``, indexed by the ``y`` label.

    Counts come from the FMoW metadata CSV (not model metrics); classes absent from
    the split get a count of 0 (:class:`DecisionRule` clamps the log before use).
    """
    if not metadata_path.exists():
        raise FileNotFoundError(f"Class-prior metadata CSV not found: {metadata_path}")
    df = pd.read_csv(metadata_path, usecols=["split", "y"])
    sub = df[df["split"] == split]
    if sub.empty:
        raise ValueError(f"No rows with split == {split!r} in {metadata_path}")
    prior = torch.zeros(num_classes)
    for cls, n in sub["y"].value_counts().items():
        prior[int(cls)] = float(n)
    return prior

class Fusion(nn.Module):
    def __init__(self, hr_dim, lr_dim, out_dim):
        super().__init__()

        self.hr_dim = hr_dim
        self.lr_dim = lr_dim
        self.out_dim = out_dim

    @property
    def produces_logits(self) -> bool:
        return False

    @abstractmethod
    def forward(
        self,
        hr_features: torch.Tensor,
        lr_features: torch.Tensor,
    ) -> torch.Tensor:
        pass

    @abstractmethod
    def forward_branch_ablation(
        self,
        hr_features: torch.Tensor,
        lr_features: torch.Tensor,
        cutoff: Literal["lr", "hr"],
    ) -> torch.Tensor:
        pass

class ConcatFusion(Fusion):
    def __init__(self, hr_dim, lr_dim, out_dim):
        super().__init__(hr_dim, lr_dim, out_dim)

        self.intermediate_dim = (hr_dim + lr_dim) // 2

        self.fusion = nn.Sequential(
            nn.Linear(hr_dim + lr_dim, self.out_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

    def forward(self, hr_features: torch.Tensor, lr_features: torch.Tensor) -> torch.Tensor:
        concatenated = torch.cat([hr_features, lr_features], dim=1)
        return self.fusion(concatenated)

    def forward_branch_ablation(self, hr_features: torch.Tensor, lr_features: torch.Tensor, cutoff: Literal["lr", "hr"]) -> torch.Tensor:
        if cutoff == "lr":
            lr_features = torch.zeros_like(lr_features)
        else:
            hr_features = torch.zeros_like(hr_features)
        return self.forward(hr_features, lr_features)


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

    def forward_branch_ablation(self, x: torch.Tensor, z: torch.Tensor, cutoff: Literal["z", "x"]) -> torch.Tensor:
        if cutoff == "z":
            if self.pre_fusion_relu:
                x = F.relu(x)
            return x
        else:
            return self.film_mul(z) + self.film_add(z)


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

    def forward_branch_ablation(self, hr_features: torch.Tensor, lr_features: torch.Tensor, cutoff: Literal["lr", "hr"]) -> torch.Tensor:
        if cutoff == "lr":
            film_features = self.film.forward_branch_ablation(x=hr_features, z=lr_features, cutoff="z") 
        else:
            film_features = self.film.forward_branch_ablation(x=hr_features, z=lr_features, cutoff="x")
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

    def forward_branch_ablation(self, hr_features: torch.Tensor, lr_features: torch.Tensor, cutoff: Literal["lr", "hr"]) -> torch.Tensor:
        if cutoff == "lr":
            return self.hr_projection(hr_features)
        return self.lr_projection(lr_features)


class DecisionRule(nn.Module):
    """Parameter-free decision fusion of two classifiers' posteriors.

    Combines the per-branch posteriors ``P(y|HR)`` and ``P(y|LR)`` (softmax of each
    frozen head's logits) by a fixed rule and returns the *log* of the renormalized
    fused posterior, so the downstream softmax/argmax/ECE see a proper distribution.

    Rules follow Kittler et al. (1998) - here with R = 2:
      - ``sum``:     ``q(y) = P(y|HR) + P(y|LR) - P(y)`` 
      - ``max``:     ``q(y) = 2 * max(P(y|HR), P(y|LR)) - P(y)`` 
      - ``min``:     ``q(y) = min(P(y|HR), P(y|LR)) / P(y)``
      - ``product``: ``q(y) = P(y|HR) * P(y|LR) / P(y)`` -- GeoPrior
        rule. Each trained classifier's posterior already embeds the train-set class
        prior ``P(y)``; multiplying the two double-counts it, so the product rule
        divides ``P(y)`` back out once. ``P(y)`` is computed from the FMoW metadata in
        the constructor (``use_class_prior=False`` -> uniform, i.e. no correction).

    Set ``use_class_prior=False`` to drop the ``P(y)`` term entirely (an ablation):
    ``sum`` -> ``P(y|HR) + P(y|LR)``, ``max`` -> ``max(P(y|HR), P(y|LR))``,
    ``min`` -> ``min(P(y|HR), P(y|LR))``, ``product`` -> ``P(y|HR) * P(y|LR)``.

    Multiplicative rules (``min`` / ``product``) keep their numerics in log space;
    ``sum`` / ``max`` are computed on the posteriors because their prior term is a
    subtraction. Everything is renormalized to a proper distribution at the end.
    """

    VALID_RULES = ("sum", "max", "min", "product")

    def __init__(
        self,
        rule: str,
        num_task_labels: int,
        use_class_prior: bool = True,
        class_prior_split: str = "train",
        class_prior_metadata: Optional[str] = None,
    ):
        super().__init__()
        if rule not in self.VALID_RULES:
            raise ValueError(f"rule must be one of {self.VALID_RULES}, got {rule!r}")
        self.rule = rule
        self.use_class_prior = use_class_prior
        # log P(y); zeros == uniform prior, which leaves the product rule unchanged
        # after the final renormalization.
        self.register_buffer("log_class_prior", torch.zeros(num_task_labels))
        # Compute and set the class prior straight from the FMoW metadata. When the
        # prior is disabled it stays uniform (the registered zeros above).
        if use_class_prior:
            metadata_path = Path(class_prior_metadata) if class_prior_metadata else DEFAULT_PRIOR_METADATA
            prior = compute_class_prior(metadata_path, class_prior_split, num_task_labels)
            self._set_class_prior(prior)

    def _set_class_prior(self, prior: torch.Tensor) -> None:
        """Set the class prior P(y) (only used by the ``product`` rule).

        ``prior`` is a length-``num_task_labels`` vector of (unnormalized) class
        frequencies; it is normalized to a distribution and stored as ``log P(y)``.
        """
        prior = prior.to(self.log_class_prior)
        prior = prior / prior.sum()
        self.log_class_prior = torch.log(prior.clamp_min(1e-12))

    def forward(self, hr_logits: torch.Tensor, lr_logits: torch.Tensor) -> torch.Tensor:
        log_p_hr = F.log_softmax(hr_logits, dim=1)
        log_p_lr = F.log_softmax(lr_logits, dim=1)

        if self.rule in ("sum", "max"):
            prior = self.log_class_prior.exp() if self.use_class_prior else 0.0
            if self.rule == "sum":
                q = log_p_hr.exp() + log_p_lr.exp() - prior
            else:
                q = 2.0 * torch.maximum(log_p_hr.exp(), log_p_lr.exp()) - prior
            log_q = q.clamp_min(1e-12).log()
        elif self.rule == "min":
            log_q = torch.minimum(log_p_hr, log_p_lr)
            if self.use_class_prior:
                log_q = log_q - self.log_class_prior
        else:  # product / GeoPrior
            log_q = log_p_hr + log_p_lr
            if self.use_class_prior:
                log_q = log_q - self.log_class_prior

        # Renormalize to a proper log-distribution so softmax(log_q) == q.
        return F.log_softmax(log_q, dim=1)

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

    @property
    def produces_logits(self) -> bool:
        return True

    def forward(self, hr_features: torch.Tensor, lr_features: torch.Tensor) -> torch.Tensor:
        hr_projected = self.hr_projection(hr_features)
        lr_projected = self.lr_projection(lr_features)
        return hr_projected + lr_projected

    def forward_branch_ablation(self, hr_features: torch.Tensor, lr_features: torch.Tensor, cutoff: Literal["lr", "hr"]) -> torch.Tensor:
        if cutoff == "lr":
            return self.hr_projection(hr_features)
        return self.lr_projection(lr_features)
