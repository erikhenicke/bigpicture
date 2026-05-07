from typing import Dict, List, Optional

import torch
import torch.nn as nn

from models.components.branches import Branch, DualBranch
from models.components.fusion import Fusion
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
        if self.fusion is not None:
            self.task_classifier = nn.Linear(self.fusion.out_dim, num_task_labels)
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
        if self.fusion is None or self.task_classifier is None:
            raise RuntimeError("LateFusionModel requires a fusion module and task classifier for forward().")

        hr_branch_out, lr_branch_out = self.branches(x)
        lr_for_domain = lr_branch_out  # save before potential detach
        if self.detach_lr_for_task:
            lr_branch_out = lr_branch_out.detach()
        fused = self.fusion(hr_branch_out, lr_branch_out)
        outputs = {
            "task_logits": self.task_classifier(fused),
            "lr_domain_logits": self.lr_domain_classifier(lr_for_domain),
            "hr_domain_logits": self.hr_domain_classifier(hr_branch_out.detach()),
        }
        return outputs

    def supports_branch_ablation(self) -> bool:
        return self.fusion is not None and self.task_classifier is not None

    def forward_branch_ablation(
        self, x: Dict[str, torch.Tensor], constant_value: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        hr_features, lr_features = self.branches(x)

        lr_const = torch.full_like(lr_features, constant_value)
        lr_const_logits = self.task_classifier(self.fusion(hr_features, lr_const))

        hr_const = torch.full_like(hr_features, constant_value)
        hr_const_logits = self.task_classifier(self.fusion(hr_const, lr_features))

        return {
            "lr_constant_logits": lr_const_logits,
            "hr_constant_logits": hr_const_logits,
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