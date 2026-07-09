"""Feature-fusion mechanisms combining an HR and an LR branch's feature vectors.

Defines the abstract :class:`Fusion` base class and its concrete subclasses --
:class:`ConcatFusion` (concatenate + MLP), :class:`FilmFusion` (FiLM conditioning
of HR features on LR features, via the standalone :class:`FiLM` module),
:class:`MultSimFusion` (project both branches then elementwise multiply), and
:class:`GeoPriorFusion` (project both branches then add, producing logits
directly) -- each used as the ``fusion`` module of
``models.components.fusion_models.FeatureFusionModel``. Also defines
:class:`DecisionRule`, a parameter-free *decision*-fusion module that combines two
already-trained classifiers' posteriors (as used by ``DecisionFusionModel``), and
:func:`compute_class_prior`, the helper that reads the class-prior vector
``DecisionRule`` needs for its ``product``/``min`` rules from the FMoW metadata CSV.
"""
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

    Args:
        metadata_path (Path): Path to the FMoW metadata CSV; must have ``split``
            and ``y`` columns.
        split (str): Value to filter the ``split`` column on (e.g. ``"train"``).
        num_classes (int): Number of task classes; determines the output length.

    Returns:
        torch.Tensor: Shape ``(num_classes,)``, float32. Raw (unnormalized) count
        of rows with each class label ``y`` within ``split``.

    Raises:
        FileNotFoundError: If ``metadata_path`` does not exist.
        ValueError: If no rows in the CSV match ``split``.
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
    """Abstract base class for HR/LR feature-fusion modules.

    Subclasses combine an HR feature vector of width ``hr_dim`` and an LR feature
    vector of width ``lr_dim`` into a single output of width ``out_dim`` (either a
    fused feature vector, or task logits directly if :attr:`produces_logits` is
    True). Used as the ``fusion`` argument of
    ``models.components.fusion_models.FeatureFusionModel``.
    """

    def __init__(self, hr_dim, lr_dim, out_dim):
        """Store the branch and output feature widths.

        Args:
            hr_dim (int): Feature dimension of the HR branch input.
            lr_dim (int): Feature dimension of the LR branch input.
            out_dim (int): Feature dimension of the fusion output (or number of
                task classes, if :attr:`produces_logits` is True).
        """
        super().__init__()

        self.hr_dim = hr_dim
        self.lr_dim = lr_dim
        self.out_dim = out_dim

    @property
    def produces_logits(self) -> bool:
        """bool: True if :meth:`forward` returns task logits directly rather than
        a fused feature vector that still needs a task classifier head."""
        return False

    @abstractmethod
    def forward(
        self,
        hr_features: torch.Tensor,
        lr_features: torch.Tensor,
    ) -> torch.Tensor:
        """Fuse an HR and an LR feature batch.

        Args:
            hr_features (torch.Tensor): Shape ``(batch_size, hr_dim)``, float32.
            lr_features (torch.Tensor): Shape ``(batch_size, lr_dim)``, float32.

        Returns:
            torch.Tensor: Shape ``(batch_size, out_dim)``, float32.
        """
        pass

    @abstractmethod
    def forward_branch_ablation(
        self,
        hr_features: torch.Tensor,
        lr_features: torch.Tensor,
        cutoff: Literal["lr", "hr"],
    ) -> torch.Tensor:
        """Fuse with one branch ablated, to measure that branch's contribution.

        Args:
            hr_features (torch.Tensor): Shape ``(batch_size, hr_dim)``, float32.
            lr_features (torch.Tensor): Shape ``(batch_size, lr_dim)``, float32.
            cutoff (Literal["lr", "hr"]): Which branch to ablate (zero out or
                otherwise remove the contribution of).

        Returns:
            torch.Tensor: Shape ``(batch_size, out_dim)``, float32.
        """
        pass

class ConcatFusion(Fusion):
    """Concatenate HR and LR features, then project with an MLP.

    ``forward`` concatenates the two feature vectors along the feature dimension
    and passes them through a single ``Linear -> ReLU -> Dropout`` block.
    """

    def __init__(self, hr_dim, lr_dim, out_dim):
        """Build the concatenation-then-projection block.

        Args:
            hr_dim (int): Feature dimension of the HR branch input.
            lr_dim (int): Feature dimension of the LR branch input.
            out_dim (int): Output feature dimension after projection.
        """
        super().__init__(hr_dim, lr_dim, out_dim)

        self.intermediate_dim = (hr_dim + lr_dim) // 2

        self.fusion = nn.Sequential(
            nn.Linear(hr_dim + lr_dim, self.out_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

    def forward(self, hr_features: torch.Tensor, lr_features: torch.Tensor) -> torch.Tensor:
        """Concatenate and project.

        Args:
            hr_features (torch.Tensor): Shape ``(batch_size, hr_dim)``, float32.
            lr_features (torch.Tensor): Shape ``(batch_size, lr_dim)``, float32.

        Returns:
            torch.Tensor: Shape ``(batch_size, out_dim)``, float32.
        """
        concatenated = torch.cat([hr_features, lr_features], dim=1)
        return self.fusion(concatenated)

    def forward_branch_ablation(self, hr_features: torch.Tensor, lr_features: torch.Tensor, cutoff: Literal["lr", "hr"]) -> torch.Tensor:
        """Zero out one branch's features, then run the normal forward pass.

        Args:
            hr_features (torch.Tensor): Shape ``(batch_size, hr_dim)``, float32.
            lr_features (torch.Tensor): Shape ``(batch_size, lr_dim)``, float32.
            cutoff (Literal["lr", "hr"]): Branch to zero out before fusing.

        Returns:
            torch.Tensor: Shape ``(batch_size, out_dim)``, float32.
        """
        if cutoff == "lr":
            lr_features = torch.zeros_like(lr_features)
        else:
            hr_features = torch.zeros_like(hr_features)
        return self.forward(hr_features, lr_features)


class FiLM(nn.Module):
    """Feature-wise Linear Modulation: condition ``x`` on ``z`` via an affine transform.

    Computes two small MLPs from the conditioning input ``z`` -- a multiplicative
    gate ``film_mul(z)`` and an additive offset ``film_add(z)``, both of width
    ``x_dim`` -- and applies ``mul * x + add`` elementwise. Used internally by
    :class:`FilmFusion` to condition HR features on LR features.
    """

    def __init__(self, x_dim: int, z_dim: int, pre_fusion_relu: bool = False) -> None:
        """Build the multiplicative and additive conditioning MLPs.

        Args:
            x_dim (int): Feature dimension of the tensor being modulated (``x``);
                also the output width of both conditioning MLPs.
            z_dim (int): Feature dimension of the conditioning tensor (``z``);
                input width of both conditioning MLPs.
            pre_fusion_relu (bool): If True, applies ReLU to ``x`` before
                modulating it, and appends a trailing ReLU to both conditioning
                MLPs (``film_add`` / ``film_mul``).
        """
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
        """Modulate ``x`` with an affine transform conditioned on ``z``.

        Args:
            x (torch.Tensor): Shape ``(batch_size, x_dim)``, float32 -- tensor to
                be modulated.
            z (torch.Tensor): Shape ``(batch_size, z_dim)``, float32 --
                conditioning tensor.

        Returns:
            torch.Tensor: Shape ``(batch_size, x_dim)``, float32. ``mul * x + add``
            (with ``x`` optionally ReLU'd first, per ``pre_fusion_relu``).
        """
        if self.pre_fusion_relu:
            x = F.relu(x)
        mul = self.film_mul(z)
        add = self.film_add(z)
        return mul * x + add

    def forward_branch_ablation(self, x: torch.Tensor, z: torch.Tensor, cutoff: Literal["z", "x"]) -> torch.Tensor:
        """Modulate with one input ablated, to measure that input's contribution.

        Args:
            x (torch.Tensor): Shape ``(batch_size, x_dim)``, float32.
            z (torch.Tensor): Shape ``(batch_size, z_dim)``, float32.
            cutoff (Literal["z", "x"]): If ``"z"``, drop the conditioning entirely
                and return ``x`` unmodulated (optionally ReLU'd). If ``"x"``, drop
                ``x`` (i.e. treat it as all-ones) and return the modulation
                ``mul + add`` on its own.

        Returns:
            torch.Tensor: Shape ``(batch_size, x_dim)``, float32.
        """
        if cutoff == "z":
            if self.pre_fusion_relu:
                x = F.relu(x)
            return x
        else:
            return self.film_mul(z) + self.film_add(z)


class FilmFusion(Fusion):
    """FiLM-conditions HR features on LR features, then projects to ``out_dim``.

    Uses :class:`FiLM` with HR features as ``x`` and LR features as ``z`` (so LR
    features modulate the HR features), then applies a
    ``Linear -> ReLU -> Dropout`` projection to the fused width.
    """

    def __init__(self, hr_dim, lr_dim, out_dim, pre_fusion_relu=False):
        """Build the FiLM block and output projection.

        Args:
            hr_dim (int): Feature dimension of the HR branch input (FiLM's
                modulated tensor ``x``).
            lr_dim (int): Feature dimension of the LR branch input (FiLM's
                conditioning tensor ``z``).
            out_dim (int): Output feature dimension after projection.
            pre_fusion_relu (bool): Passed through to :class:`FiLM`; see there.
        """
        super().__init__(hr_dim, lr_dim, out_dim)

        self.film = FiLM(hr_dim, lr_dim, pre_fusion_relu=pre_fusion_relu)

        self.projection = nn.Sequential(
            nn.Linear(hr_dim, out_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

    def forward(self, hr_features: torch.Tensor, lr_features: torch.Tensor) -> torch.Tensor:
        """FiLM-modulate HR features on LR features, then project.

        Args:
            hr_features (torch.Tensor): Shape ``(batch_size, hr_dim)``, float32.
            lr_features (torch.Tensor): Shape ``(batch_size, lr_dim)``, float32.

        Returns:
            torch.Tensor: Shape ``(batch_size, out_dim)``, float32.
        """
        film_features = self.film(hr_features, lr_features)
        return self.projection(film_features)

    def forward_branch_ablation(self, hr_features: torch.Tensor, lr_features: torch.Tensor, cutoff: Literal["lr", "hr"]) -> torch.Tensor:
        """Fuse with one branch ablated via :meth:`FiLM.forward_branch_ablation`.

        Args:
            hr_features (torch.Tensor): Shape ``(batch_size, hr_dim)``, float32.
            lr_features (torch.Tensor): Shape ``(batch_size, lr_dim)``, float32.
            cutoff (Literal["lr", "hr"]): ``"lr"`` drops the LR conditioning
                (FiLM's ``cutoff="z"``); ``"hr"`` drops the HR features being
                modulated (FiLM's ``cutoff="x"``).

        Returns:
            torch.Tensor: Shape ``(batch_size, out_dim)``, float32.
        """
        if cutoff == "lr":
            film_features = self.film.forward_branch_ablation(x=hr_features, z=lr_features, cutoff="z") 
        else:
            film_features = self.film.forward_branch_ablation(x=hr_features, z=lr_features, cutoff="x")
        return self.projection(film_features)


class MultSimFusion(Fusion):
    """Project HR and LR features to a shared width, then multiply elementwise.

    Each branch is passed through its own ``Linear [-> ReLU] -> Dropout``
    projection to ``out_dim``, and the two projections are combined with
    elementwise multiplication (a similarity-style interaction).
    """

    def __init__(self, hr_dim, lr_dim, out_dim, pre_fusion_relu=True):
        """Build the per-branch projections.

        Args:
            hr_dim (int): Feature dimension of the HR branch input.
            lr_dim (int): Feature dimension of the LR branch input.
            out_dim (int): Shared projected feature dimension (and output width).
            pre_fusion_relu (bool): If True, apply ReLU after each branch's linear
                projection, before dropout.
        """
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
        """Project both branches, then multiply elementwise.

        Args:
            hr_features (torch.Tensor): Shape ``(batch_size, hr_dim)``, float32.
            lr_features (torch.Tensor): Shape ``(batch_size, lr_dim)``, float32.

        Returns:
            torch.Tensor: Shape ``(batch_size, out_dim)``, float32.
        """
        hr_projected = self.hr_projection(hr_features)
        lr_projected = self.lr_projection(lr_features)
        return torch.mul(hr_projected, lr_projected)

    def forward_branch_ablation(self, hr_features: torch.Tensor, lr_features: torch.Tensor, cutoff: Literal["lr", "hr"]) -> torch.Tensor:
        """Return only the surviving branch's projection (no multiplication).

        Args:
            hr_features (torch.Tensor): Shape ``(batch_size, hr_dim)``, float32.
            lr_features (torch.Tensor): Shape ``(batch_size, lr_dim)``, float32.
            cutoff (Literal["lr", "hr"]): Branch to ablate; the other branch's
                projection is returned directly.

        Returns:
            torch.Tensor: Shape ``(batch_size, out_dim)``, float32.
        """
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
        """Select the decision rule and (optionally) compute the class prior.

        Args:
            rule (str): One of :attr:`VALID_RULES` (``"sum"``, ``"max"``,
                ``"min"``, ``"product"``).
            num_task_labels (int): Number of task classes; length of the
                registered ``log_class_prior`` buffer.
            use_class_prior (bool): If True, compute ``P(y)`` from the FMoW
                metadata CSV and use it in the rule (see class docstring). If
                False, ``log_class_prior`` stays all-zero (uniform prior), which
                is equivalent to dropping the ``P(y)`` term.
            class_prior_split (str): Metadata ``split`` value to compute the
                class prior over (passed to :func:`compute_class_prior`).
            class_prior_metadata (Optional[str]): Path to the FMoW metadata CSV;
                defaults to :data:`DEFAULT_PRIOR_METADATA` if not given.

        Raises:
            ValueError: If ``rule`` is not one of :attr:`VALID_RULES`.
        """
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

        Args:
            prior (torch.Tensor): Shape ``(num_task_labels,)`` -- unnormalized
                per-class frequency counts, as returned by
                :func:`compute_class_prior`.
        """
        prior = prior.to(self.log_class_prior)
        prior = prior / prior.sum()
        self.log_class_prior = torch.log(prior.clamp_min(1e-12))

    def forward(self, hr_logits: torch.Tensor, lr_logits: torch.Tensor) -> torch.Tensor:
        """Combine the HR and LR classifiers' logits into fused log-probabilities.

        Args:
            hr_logits (torch.Tensor): Shape ``(batch_size, num_task_labels)``,
                float32 -- HR classifier's raw logits.
            lr_logits (torch.Tensor): Shape ``(batch_size, num_task_labels)``,
                float32 -- LR classifier's raw logits.

        Returns:
            torch.Tensor: Shape ``(batch_size, num_task_labels)``, float32.
            Log of the renormalized fused posterior (i.e. ``log_softmax`` output),
            per the selected rule -- see the class docstring for the exact
            per-rule formula.
        """
        log_p_hr = F.log_softmax(hr_logits, dim=1)
        log_p_lr = F.log_softmax(lr_logits, dim=1)

        if self.rule in ("sum", "max"):
            prior = self.log_class_prior.exp() if self.use_class_prior else 0.0
            if self.rule == "sum":
                q = log_p_hr.exp() + log_p_lr.exp() - prior
            else:
                q = 2.0 * torch.maximum(log_p_hr.exp(), log_p_lr.exp()) - prior
            # Per-row shift to non-negative (only where a row has negatives), so each
            # sample's fused distribution is independent of the rest of the batch.
            q = q - q.amin(dim=1, keepdim=True).clamp(max=0.0)
            log_q = q.clamp(1e-12).log()
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
    """Project HR and LR features to logit-width and add (log-additive fusion).

    Named after the GeoPrior-style additive combination of an image classifier's
    logits with a geo-prior signal: each branch is projected to ``out_dim`` (the
    number of task classes) and the two projections are summed directly as
    logits, i.e. this module produces logits itself rather than a feature vector
    (see :attr:`produces_logits`), so no separate task classifier head is needed
    downstream.
    """

    def __init__(self, hr_dim, lr_dim, out_dim, pre_fusion_relu=True):
        """Build the per-branch logit projections.

        Args:
            hr_dim (int): Feature dimension of the HR branch input.
            lr_dim (int): Feature dimension of the LR branch input.
            out_dim (int): Number of task classes (both branches project to this
                width so their logits can be summed).
            pre_fusion_relu (bool): If True, apply ReLU after each branch's linear
                projection, before dropout.
        """
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
        """bool: Always True -- :meth:`forward` returns task logits directly."""
        return True

    def forward(self, hr_features: torch.Tensor, lr_features: torch.Tensor) -> torch.Tensor:
        """Project both branches to logit-width and add.

        Args:
            hr_features (torch.Tensor): Shape ``(batch_size, hr_dim)``, float32.
            lr_features (torch.Tensor): Shape ``(batch_size, lr_dim)``, float32.

        Returns:
            torch.Tensor: Shape ``(batch_size, out_dim)``, float32 -- summed task
            logits (``out_dim == num_task_labels`` at the call sites).
        """
        hr_projected = self.hr_projection(hr_features)
        lr_projected = self.lr_projection(lr_features)
        return hr_projected + lr_projected

    def forward_branch_ablation(self, hr_features: torch.Tensor, lr_features: torch.Tensor, cutoff: Literal["lr", "hr"]) -> torch.Tensor:
        """Return only the surviving branch's logit projection (no summation).

        Args:
            hr_features (torch.Tensor): Shape ``(batch_size, hr_dim)``, float32.
            lr_features (torch.Tensor): Shape ``(batch_size, lr_dim)``, float32.
            cutoff (Literal["lr", "hr"]): Branch to ablate; the other branch's
                logit projection is returned directly.

        Returns:
            torch.Tensor: Shape ``(batch_size, out_dim)``, float32.
        """
        if cutoff == "lr":
            return self.hr_projection(hr_features)
        return self.lr_projection(lr_features)
