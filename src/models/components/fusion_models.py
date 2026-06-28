from abc import ABC, abstractmethod
from typing import Dict, List, Optional
from pathlib import Path

import torch
import torch.nn as nn

from models.components.branches import Branch, DualBranch, SatCLIPLocationBranch
from models.components.fusion import Fusion, DecisionRule
from models.components.domain_relations import D3GRelation
from results.utils import find_best_checkpoints, find_run_dir


class MultiScaleModel(nn.Module, ABC):
    """Abstract base class every model wrapped by ``MultiScaleClassificationModule``
    derives from. It is itself an ``nn.Module``, so every model is-a ``nn.Module`` and
    is-a ``MultiScaleModel``. The wrapper is fusion-agnostic: it depends only on this
    interface, not on any concrete subclass.

    Subclasses: ``SingleBranchModel`` / ``SingleBranchLRModel`` /
    ``SingleBranchLocationModel`` (single-scale), ``EarlyFusionModel`` (early
    fusion), ``FeatureFusionModel`` (feature fusion), ``D3GModel`` (domain-gated feature
    fusion), and ``DecisionFusionModel`` (decision fusion).

    The abstract methods below must be implemented by every concrete subclass -- an
    incomplete subclass cannot be instantiated (``TypeError`` at construction). Only
    ``train_model`` has a concrete default.

    ``forward`` always returns a dict containing ``task_logits``. The capability-gated
    keys (``lr_domain_logits`` / ``hr_domain_logits`` / ``rel_logits``) and the optional
    members (``forward_branch_ablation``, ``lr_domain_loss_coeff``,
    ``consistency_loss_coeff``) are present only when the matching ``supports_*`` method
    returns True.
    """

    @abstractmethod
    def forward(
        self, x: Dict[str, torch.Tensor], region_ids: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]: ...

    def train_model(self) -> bool:
        """Whether this model is learned via ``Trainer.fit``.

        Concrete default: True for every model, inherited by all subclasses. The sole
        exception is ``DecisionFusionModel``, which overrides this to return its own
        ``train_model`` flag -- False means its frozen heads are evaluated test-only,
        True means the decision heads are trained jointly from scratch.
        """
        return True

    @abstractmethod
    def supports_d3g_objective(self) -> bool: ...
    @abstractmethod
    def supports_lr_domain_classification(self) -> bool: ...
    @abstractmethod
    def supports_hr_domain_classification(self) -> bool: ...
    @abstractmethod
    def supports_branch_ablation(self) -> bool: ...

    @abstractmethod
    def task_parameters(self) -> List[torch.nn.Parameter]: ...
    @abstractmethod
    def lr_domain_parameters(self) -> List[torch.nn.Parameter]: ...
    @abstractmethod
    def hr_domain_parameters(self) -> List[torch.nn.Parameter]: ...


class SingleBranchModel(MultiScaleModel):
    """HR-only baseline: single encoder + task classifier, no fusion."""

    def __init__(self, encoder: Branch, num_task_labels: int, num_domain_labels: int = 6):
        super().__init__()
        self.encoder = encoder
        self.task_classifier = nn.Linear(encoder.out_dim, num_task_labels)
        self.hr_domain_classifier = nn.Linear(encoder.out_dim, num_domain_labels)

    def supports_d3g_objective(self) -> bool:
        return False

    def supports_lr_domain_classification(self) -> bool:
        return False

    def supports_hr_domain_classification(self) -> bool:
        return True

    def supports_branch_ablation(self) -> bool:
        return False

    def forward(self, x: Dict[str, torch.Tensor], region_ids: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        features = self.encoder(x["rgb"])
        return {
            "task_logits": self.task_classifier(features),
            "hr_domain_logits": self.hr_domain_classifier(features.detach()),
        }

    def task_parameters(self) -> List[torch.nn.Parameter]:
        return list(self.encoder.parameters()) + list(self.task_classifier.parameters())

    def lr_domain_parameters(self) -> List[torch.nn.Parameter]:
        return []

    def hr_domain_parameters(self) -> List[torch.nn.Parameter]:
        return list(self.hr_domain_classifier.parameters())


class SingleBranchLRModel(MultiScaleModel):
    """LR-only model: single LR encoder + task classifier + LR domain head."""

    def __init__(
        self,
        encoder: Branch,
        num_task_labels: int,
        num_domain_labels: int = 6,
        lr_domain_loss_coeff: float = 0.1667,
        landsat_channels: int = 6,
    ):
        super().__init__()
        self.encoder = encoder
        self.landsat_channels = landsat_channels
        self.lr_domain_loss_coeff = lr_domain_loss_coeff
        self.task_classifier = nn.Linear(encoder.out_dim, num_task_labels)
        self.lr_domain_classifier = nn.Linear(encoder.out_dim, num_domain_labels)

    def supports_d3g_objective(self) -> bool:
        return False

    def supports_lr_domain_classification(self) -> bool:
        return True

    def supports_hr_domain_classification(self) -> bool:
        return False

    def supports_branch_ablation(self) -> bool:
        return False

    def forward(self, x: Dict[str, torch.Tensor], region_ids: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        features = self.encoder(x["landsat"][:, :self.landsat_channels, :, :])
        return {
            "task_logits": self.task_classifier(features),
            "lr_domain_logits": self.lr_domain_classifier(features),
        }

    def task_parameters(self) -> List[torch.nn.Parameter]:
        return list(self.encoder.parameters()) + list(self.task_classifier.parameters())

    def lr_domain_parameters(self) -> List[torch.nn.Parameter]:
        return list(self.lr_domain_classifier.parameters())

    def hr_domain_parameters(self) -> List[torch.nn.Parameter]:
        return []


class SingleBranchLocationModel(MultiScaleModel):
    """Location-only model: SatCLIP encoder + task classifier + LR domain head."""

    def __init__(
        self,
        encoder: SatCLIPLocationBranch,
        num_task_labels: int,
        num_domain_labels: int = 6,
        lr_domain_loss_coeff: float = 0.1667,
    ):
        super().__init__()
        self.encoder = encoder
        self.lr_domain_loss_coeff = lr_domain_loss_coeff
        self.task_classifier = nn.Linear(encoder.out_dim, num_task_labels)
        self.lr_domain_classifier = nn.Linear(encoder.out_dim, num_domain_labels)

    def supports_d3g_objective(self) -> bool:
        return False

    def supports_lr_domain_classification(self) -> bool:
        return True

    def supports_hr_domain_classification(self) -> bool:
        return False

    def supports_branch_ablation(self) -> bool:
        return False

    def forward(self, x: Dict[str, torch.Tensor], region_ids: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        features = self.encoder(x["coords"])
        return {
            "task_logits": self.task_classifier(features),
            "lr_domain_logits": self.lr_domain_classifier(features),
        }

    def task_parameters(self) -> List[torch.nn.Parameter]:
        return list(self.encoder.parameters()) + list(self.task_classifier.parameters())

    def lr_domain_parameters(self) -> List[torch.nn.Parameter]:
        return list(self.lr_domain_classifier.parameters())

    def hr_domain_parameters(self) -> List[torch.nn.Parameter]:
        return []

class EarlyFusionModel(MultiScaleModel):
    """Early fusion model: concatenates HR RGB and Landsat along channels, feeds a single encoder."""

    def __init__(
        self,
        encoder: Branch,
        num_task_labels: int,
        num_domain_labels: int = 6,
        lr_domain_loss_coeff: float = 0.1667,
        landsat_channels: int = 6,
    ):
        super().__init__()
        self.encoder = encoder
        self.landsat_channels = landsat_channels
        self.lr_domain_loss_coeff = lr_domain_loss_coeff
        self.task_classifier = nn.Linear(encoder.out_dim, num_task_labels)
        self.lr_domain_classifier = nn.Linear(encoder.out_dim, num_domain_labels)

    def supports_d3g_objective(self) -> bool:
        return False

    def supports_lr_domain_classification(self) -> bool:
        return True

    def supports_hr_domain_classification(self) -> bool:
        return False

    def supports_branch_ablation(self) -> bool:
        return False

    def forward(self, x: Dict[str, torch.Tensor], region_ids: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        stacked = torch.cat([x["rgb"], x["landsat"][:, :self.landsat_channels, :, :]], dim=1)
        features = self.encoder(stacked)
        return {
            "task_logits": self.task_classifier(features),
            "lr_domain_logits": self.lr_domain_classifier(features),
        }

    def task_parameters(self) -> List[torch.nn.Parameter]:
        return list(self.encoder.parameters()) + list(self.task_classifier.parameters())

    def lr_domain_parameters(self) -> List[torch.nn.Parameter]:
        return list(self.lr_domain_classifier.parameters())

    def hr_domain_parameters(self) -> List[torch.nn.Parameter]:
        return []


class DecisionFusionModel(MultiScaleModel):
    """Decision fusion over precomputed single-branch features.

    Holds two single-branch task heads (HR + LR) and a :class:`DecisionRule`. It
    consumes cached encoder features served by ``source="features"`` -- ``x["rgb"]``
    is the HR feature vector and ``x["landsat"]`` the LR feature vector -- applies
    each head to get per-branch logits, and combines them with the decision rule.

    Two modes, selected by ``train_model``:

    - ``train_model=False`` (default): test-only. The heads copy the trained
      single-branch weights via :meth:`load_heads` (mirroring extract_features.py's
      "build once, reload weights per seed" pattern) and are frozen, so the model
      exposes no task parameters and there is nothing to fit.
    - ``train_model=True``: the heads are reinitialized and left trainable, so they
      can be learned jointly from scratch through the decision rule via
      ``Trainer.fit``. Only the input width is taken from the checkpoint; the weights
      are not copied.

    Either way the heads supports neither the domain nor the D3G objectives: the
    domain metrics already live in each single-branch run folder and worst-group
    accuracy uses the ground-truth region, so no domain head is needed here.

    The heads are built in the constructor, which reads each branch's feature width
    straight from the saved classifier weight -- so the feature dimension is never
    configured, it is inferred from the checkpoint. The class prior P(y) is handled
    entirely by the :class:`DecisionRule` (computed from metadata in its constructor).
    """

    def __init__(
        self,
        num_task_labels: int,
        rule: DecisionRule,
        hr_run_name: str,
        lr_run_name: str,
        run_idx: int = 0,
        train_model: bool = False,
    ):
        super().__init__()
        self.num_task_labels = num_task_labels
        self.rule = rule
        # Stored under a leading underscore so it never shadows the train_model()
        # method (an instance attribute of the same name would mask the method).
        self._train_model = train_model
        self.hr_head: Optional[nn.Linear] = None
        self.lr_head: Optional[nn.Linear] = None
        self._load_heads(hr_run_name, lr_run_name, run_idx)

    def train_model(self) -> bool:
        return self._train_model

    def supports_d3g_objective(self) -> bool:
        return False

    def supports_lr_domain_classification(self) -> bool:
        return False

    def supports_hr_domain_classification(self) -> bool:
        return False

    def supports_branch_ablation(self) -> bool:
        return False

    def forward(self, x: Dict[str, torch.Tensor], region_ids: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        if self.hr_head is None or self.lr_head is None:
            raise RuntimeError("DecisionFusionModel heads were not built in the constructor.")
        hr_logits = self.hr_head(x["rgb"])
        lr_logits = self.lr_head(x["landsat"])
        return {"task_logits": self.rule(hr_logits, lr_logits)}

    def _load_heads(self, hr_run_name: str, lr_run_name: str, run_idx: int) -> None:
        """Build this seed's two task heads from the single-branch checkpoints.

        The HR/LR heads come from the same single-branch runs that produced the cached
        features, so ``hr_run_name`` / ``lr_run_name`` must be the experiment keys (e.g.
        ``densenet_baseline``, which ``find_run_dir`` resolves to ``train_<key>``). The
        head for seed ``run_idx`` is built and its feature width read straight from the
        saved ``model.task_classifier.*`` weight (the ``model.`` prefix comes from the
        ``MultiScaleClassificationModule`` wrapper), so the feature dimension is never
        configured.

        In test-only mode (``train_model=False``) the heads copy the trained weights and
        are frozen. In training mode (``train_model=True``) only the shape is taken from
        the checkpoint -- the heads are reinitialized and left trainable so they can be
        learned jointly from scratch through the decision rule.
        """
        hr_dir = find_run_dir(hr_run_name)
        lr_dir = find_run_dir(lr_run_name)
        if hr_dir is None or lr_dir is None:
            raise FileNotFoundError(
                f"Could not resolve head runs: hr_run_name={hr_run_name} -> {hr_dir}, "
                f"lr_run_name={lr_run_name} -> {lr_dir}"
            )
        hr_ckpts = find_best_checkpoints(hr_dir)
        lr_ckpts = find_best_checkpoints(lr_dir)
        if run_idx >= len(hr_ckpts) or run_idx >= len(lr_ckpts):
            raise IndexError(
                f"Seed {run_idx} out of range (HR has {len(hr_ckpts)} seeds, LR has {len(lr_ckpts)})."
            )

        def state(path: Path) -> dict:
            return torch.load(path, map_location="cpu", weights_only=False)["state_dict"]

        device = self.rule.log_class_prior.device
        copy_weights = not self._train_model
        self.hr_head = self._build_head(state(hr_ckpts[run_idx]), copy_weights=copy_weights).to(device)
        self.lr_head = self._build_head(state(lr_ckpts[run_idx]), copy_weights=copy_weights).to(device)
        if not self._train_model:
            # Frozen: loaded from trained checkpoints, never optimized.
            self.requires_grad_(False)

    def _build_head(self, state_dict: Dict[str, torch.Tensor], copy_weights: bool) -> nn.Linear:
        weight = state_dict["model.task_classifier.weight"]
        bias = state_dict["model.task_classifier.bias"]
        out_dim, in_dim = weight.shape
        if out_dim != self.num_task_labels:
            raise ValueError(
                f"Head has {out_dim} outputs but num_task_labels={self.num_task_labels}."
            )
        head = nn.Linear(in_dim, out_dim)
        if copy_weights:
            with torch.no_grad():
                head.weight.copy_(weight)
                head.bias.copy_(bias)
        return head

    def task_parameters(self) -> List[torch.nn.Parameter]:
        if not self._train_model:
            return []
        params: List[torch.nn.Parameter] = []
        if self.hr_head is not None:
            params += list(self.hr_head.parameters())
        if self.lr_head is not None:
            params += list(self.lr_head.parameters())
        return params

    def lr_domain_parameters(self) -> List[torch.nn.Parameter]:
        return []

    def hr_domain_parameters(self) -> List[torch.nn.Parameter]:
        return []


class FeatureFusionModel(MultiScaleModel):
    def __init__(
        self,
        branches: DualBranch,
        fusion: Optional[Fusion],
        num_task_labels: int,
        num_domain_labels: int,
        lr_domain_loss_coeff: float = 0.1667,
        detach_lr_for_task: bool = False,
    ):
        super().__init__()

        self.branches: DualBranch = branches
        self.fusion = fusion
        self.lr_domain_loss_coeff = lr_domain_loss_coeff
        self.detach_lr_for_task = detach_lr_for_task
        self.task_classifier: Optional[nn.Linear] = None
        if self.fusion is not None and not self.fusion.produces_logits:
            self.task_classifier = nn.Linear(self.fusion.out_dim, num_task_labels)
        elif self.fusion is not None:
            assert self.fusion.out_dim == num_task_labels, (
                f"Fusion produces logits but out_dim ({self.fusion.out_dim}) != "
                f"num_task_labels ({num_task_labels})"
            )
        self.lr_domain_classifier = nn.Linear(branches.lr_encoder.out_dim, num_domain_labels)
        self.hr_domain_classifier = nn.Linear(branches.hr_encoder.out_dim, num_domain_labels)

    def supports_d3g_objective(self) -> bool:
        return isinstance(self, D3GModel)

    def supports_lr_domain_classification(self) -> bool:
        return True

    def supports_hr_domain_classification(self) -> bool:
        return True

    def forward(
        self,
        x: Dict[str, torch.Tensor],
        region_ids: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        if self.fusion is None:
            raise RuntimeError("FeatureFusionModel requires a fusion module for forward().")

        hr_branch_out, lr_branch_out = self.branches(x)
        lr_for_domain = lr_branch_out
        if self.detach_lr_for_task:
            lr_branch_out = lr_branch_out.detach()
        fused = self.fusion(hr_branch_out, lr_branch_out)
        task_logits = self.task_classifier(fused) if self.task_classifier is not None else fused
        outputs = {
            "task_logits": task_logits,
            "lr_domain_logits": self.lr_domain_classifier(lr_for_domain),
            "hr_domain_logits": self.hr_domain_classifier(hr_branch_out.detach()),
        }
        return outputs

    def supports_branch_ablation(self) -> bool:
        return self.fusion is not None

    def forward_branch_ablation(
        self, x: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        hr_features, lr_features = self.branches(x)
        lr_ablated = self.fusion.forward_branch_ablation(hr_features, lr_features, cutoff="lr")
        hr_ablated = self.fusion.forward_branch_ablation(hr_features, lr_features, cutoff="hr")
        if self.task_classifier is not None:
            lr_ablated = self.task_classifier(lr_ablated)
            hr_ablated = self.task_classifier(hr_ablated)
        return {
            "lr_ablated_logits": lr_ablated,
            "hr_ablated_logits": hr_ablated,
        }

    def task_parameters(self) -> List[torch.nn.Parameter]:
        parameters = list(self.branches.parameters())
        if self.fusion is not None:
            parameters += list(self.fusion.parameters())
        if self.task_classifier is not None:
            parameters += list(self.task_classifier.parameters())
        return parameters

    def lr_domain_parameters(self) -> List[torch.nn.Parameter]:
        return list(self.lr_domain_classifier.parameters())

    def hr_domain_parameters(self) -> List[torch.nn.Parameter]:
        return list(self.hr_domain_classifier.parameters())


class D3GModel(FeatureFusionModel):
    def __init__(
        self,
        branches: DualBranch,
        num_task_labels: int,
        num_domain_labels: int,
        lr_domain_loss_coeff: float = 0.1176,
        consistency_loss_coeff: float = 0.2941,
        learnable_relation_coeff: float = 0.8,
        pred_domain_for_d3g: bool = True,
        detach_lr_for_consistency: bool = False,
        detach_hr_for_consistency: bool = False,
    ):
        super().__init__(
            branches=branches,
            fusion=None,
            num_task_labels=num_task_labels,
            num_domain_labels=num_domain_labels,
            lr_domain_loss_coeff=lr_domain_loss_coeff,
        )
        self.detach_lr_for_consistency = detach_lr_for_consistency
        self.detach_hr_for_consistency = detach_hr_for_consistency
        self.num_heads = num_domain_labels
        self.d3g_relation = D3GRelation(learnable_relation_coeff=learnable_relation_coeff, internal_dim=256, lr_features_dim=branches.lr_encoder.out_dim, num_domains=num_domain_labels)
        self.consistency_loss_coeff = consistency_loss_coeff
        self.pred_domain_for_d3g = pred_domain_for_d3g
        self.task_classifier = None
        self.task_classifiers = nn.ModuleList(
            [nn.Linear(branches.hr_encoder.out_dim, num_task_labels) for _ in range(self.num_heads)]
        )

    def forward(self, x: Dict[str, torch.Tensor], region_ids: Optional[torch.Tensor] = None):
        hr_features, lr_features = self.branches(x)

        if self.training:
            if region_ids is None:
                raise ValueError("D3GModel requires region_ids for training.")
        else:  # evaluation mode - here we have the option to use ground truth region ids or predicted ones
            if not self.pred_domain_for_d3g:
                if region_ids is None:
                    raise ValueError("D3GModel requires region_ids for inference when pred_domain_for_d3g is False.")
            else:
                region_ids = self.lr_domain_classifier(lr_features).argmax(dim=1)

        head_outputs = torch.stack(
            [head(hr_features) for head in self.task_classifiers],
            dim=1,
        )

        if self.training:
            task_logits = head_outputs[torch.arange(len(region_ids), device=region_ids.device), region_ids]
            hr_for_consistency = hr_features.detach() if self.detach_hr_for_consistency else hr_features
            head_outputs_detached = torch.stack(
                [head(hr_for_consistency) for head in self.task_classifiers],
                dim=1,
            )
            lr_for_consistency = lr_features.detach() if self.detach_lr_for_consistency else lr_features
            domain_weights = self.domain_weights(region_ids, lr_for_consistency)
            consistency_weights = domain_weights.clone()
            consistency_weights[
                torch.arange(len(region_ids), device=region_ids.device),
                region_ids,
            ] = 0.0
            consistency_weights_sum = torch.clamp(consistency_weights.sum(dim=1), min=1e-6)
            rel_logits = torch.sum(consistency_weights * head_outputs_detached, dim=1) / consistency_weights_sum
        else:
            domain_weights = self.domain_weights(region_ids, lr_features)
            weights_sum = torch.clamp(domain_weights.sum(dim=1), min=1e-6)
            rel_logits = torch.sum(domain_weights * head_outputs, dim=1) / weights_sum
            task_logits = rel_logits

        outputs = {
            "task_logits": task_logits,
            "rel_logits": rel_logits,
            "lr_domain_logits": self.lr_domain_classifier(lr_features),
            "hr_domain_logits": self.hr_domain_classifier(hr_features.detach()),
        }
        return outputs

    def task_parameters(self) -> List[torch.nn.Parameter]:
        return (
            list(self.branches.parameters())
            + list(self.d3g_relation.parameters())
            + list(self.task_classifiers.parameters())
        )

    def domain_weights(
        self,
        region_ids: torch.Tensor,
        lr_features: torch.Tensor,
    ) -> torch.Tensor:
        weights = torch.cat(
            [
                self.d3g_relation(
                    region_ids,
                    lr_features,
                    torch.tensor(head_idx, device=lr_features.device),
                )
                for head_idx in range(self.num_heads)
            ],
            dim=1,
        )
        return weights.unsqueeze(-1)
