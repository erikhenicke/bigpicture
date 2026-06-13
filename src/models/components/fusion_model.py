from typing import Dict, List, Optional

import torch
import torch.nn as nn

from models.components.branches import Branch, DualBranch, SatCLIPLocationBranch
from models.components.fusion import Fusion, DecisionRule
from models.components.domain_relations import D3GRelation


class SingleBranchModel(nn.Module):
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


class SingleBranchLRModel(nn.Module):
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


class SingleBranchLocationModel(nn.Module):
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

class SingleBranchStackedModel(nn.Module):
    """Stacked-input model: concatenates HR RGB and Landsat along channels, feeds a single encoder."""

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


class DecisionFusionModel(nn.Module):
    """Parameter-free decision fusion over precomputed single-branch features.

    Holds the two *frozen* single-branch task heads (HR + LR) and a
    :class:`DecisionRule`. It consumes cached encoder features served by
    ``source="features"`` -- ``x["rgb"]`` is the HR feature vector and
    ``x["landsat"]`` the LR feature vector -- applies each frozen head to get
    per-branch logits, and combines them with the decision rule.

    There is nothing to train: the heads are loaded from the single-branch
    checkpoints via :meth:`load_heads` (mirroring extract_features.py's
    "build once, reload weights per seed" pattern), so the model exposes no task /
    domain parameters and supports neither the domain nor the D3G objectives. The
    domain metrics already live in each single-branch run folder and worst-group
    accuracy uses the ground-truth region, so no domain head is needed here.

    The heads are built lazily in :meth:`load_heads`, which reads each branch's
    feature width straight from the saved classifier weight -- so the feature
    dimension is never configured, it is inferred from the checkpoint.
    """

    def __init__(self, num_task_labels: int, rule: DecisionRule):
        super().__init__()
        self.num_task_labels = num_task_labels
        self.rule = rule
        # Built (and frozen) lazily from the checkpoints in load_heads().
        self.hr_head: Optional[nn.Linear] = None
        self.lr_head: Optional[nn.Linear] = None

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
            raise RuntimeError("DecisionFusionModel.load_heads() must be called before forward().")
        hr_logits = self.hr_head(x["rgb"])
        lr_logits = self.lr_head(x["landsat"])
        return {"task_logits": self.rule(hr_logits, lr_logits)}

    def load_heads(self, hr_state_dict: Dict[str, torch.Tensor], lr_state_dict: Dict[str, torch.Tensor]) -> None:
        """Build the two frozen task heads from single-branch checkpoint state dicts.

        Both ``SingleBranchModel`` (HR) and ``SingleBranchLRModel`` (LR) store their
        head under ``model.task_classifier.*`` (the ``model.`` prefix comes from the
        ``LateFusionModule`` wrapper). Each head's input width is read from the saved
        weight, so the feature dimension never has to be configured.
        """
        device = self.rule.log_class_prior.device
        self.hr_head = self._build_head(hr_state_dict).to(device)
        self.lr_head = self._build_head(lr_state_dict).to(device)
        # Frozen: loaded from trained checkpoints, never optimized.
        self.requires_grad_(False)

    def _build_head(self, state_dict: Dict[str, torch.Tensor]) -> nn.Linear:
        weight = state_dict["model.task_classifier.weight"]
        bias = state_dict["model.task_classifier.bias"]
        out_dim, in_dim = weight.shape
        if out_dim != self.num_task_labels:
            raise ValueError(
                f"Head has {out_dim} outputs but num_task_labels={self.num_task_labels}."
            )
        head = nn.Linear(in_dim, out_dim)
        with torch.no_grad():
            head.weight.copy_(weight)
            head.bias.copy_(bias)
        return head

    def task_parameters(self) -> List[torch.nn.Parameter]:
        return []

    def lr_domain_parameters(self) -> List[torch.nn.Parameter]:
        return []

    def hr_domain_parameters(self) -> List[torch.nn.Parameter]:
        return []


class LateFusionModel(nn.Module):
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
            raise RuntimeError("LateFusionModel requires a fusion module for forward().")

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


class D3GModel(LateFusionModel):
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
        self.d3g_relation = D3GRelation(learnable_relation_coeff=learnable_relation_coeff, internal_dim=256, lr_features_dim=branches.lr_encoder.out_dim)
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
