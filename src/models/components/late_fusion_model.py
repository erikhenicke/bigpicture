from typing import Dict, List, Optional

import torch
import torch.nn as nn

from models.components.branches import DualBranch
from models.components.fusion import Fusion
from models.components.domain_relations import D3GRelation


class LateFusionModel(nn.Module):
    def __init__(
        self,
        branches: DualBranch,
        fusion: Optional[Fusion],
        num_task_labels: int,
        num_domain_labels: int,
        enable_domain_head: bool = True,
        domain_loss_coeff: float = 0.5,
    ):
        super().__init__()

        self.branches: DualBranch = branches
        self.fusion = fusion
        self.domain_loss_coeff = domain_loss_coeff
        self.task_classifier: Optional[nn.Linear] = None
        if self.fusion is not None:
            self.task_classifier = nn.Linear(self.fusion.out_dim, num_task_labels)
        self.enable_domain_head = enable_domain_head
        self.domain_classifier: Optional[nn.Linear] = None
        if self.enable_domain_head:
            self.domain_classifier = nn.Linear(branches.lr_encoder.out_dim, num_domain_labels)

    def supports_domain_objective(self) -> bool:
        return self.enable_domain_head and self.domain_classifier is not None

    def supports_d3g_objective(self) -> bool:
        return isinstance(self, D3GModel)

    def forward(
        self,
        x: Dict[str, torch.Tensor],
        region_ids: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        if self.fusion is None or self.task_classifier is None:
            raise RuntimeError("LateFusionModel requires a fusion module and task classifier for forward().")

        hr_branch_out, lr_branch_out = self.branches(x)
        fused = self.fusion(hr_branch_out, lr_branch_out)
        outputs = {"task_logits": self.task_classifier(fused)}
        if self.supports_domain_objective():
            outputs["domain_logits"] = self.domain_classifier(lr_branch_out)
            outputs["domain_logits_detached"] = self.domain_classifier(lr_branch_out.detach())
        return outputs

    def task_parameters(self) -> List[torch.nn.Parameter]:
        parameters = list(self.branches.parameters())
        if self.fusion is not None:
            parameters += list(self.fusion.parameters())
        if self.task_classifier is not None:
            parameters += list(self.task_classifier.parameters())
        return parameters

    def domain_parameters(self) -> List[torch.nn.Parameter]:
        if self.domain_classifier is None:
            return []
        return list(self.domain_classifier.parameters())


class D3GModel(LateFusionModel):
    def __init__(
        self,
        branches: DualBranch,
        num_task_labels: int,
        num_domain_labels: int,
        enable_domain_head: bool = True,
        domain_loss_coeff: float = 0.5,
        learnable_relation_coeff: float = 0.8,
        consistency_loss_coeff: float = 0.5,
        pred_domain_for_d3g: bool = True,
    ):
        super().__init__(
            branches=branches,
            fusion=None,
            num_task_labels=num_task_labels,
            num_domain_labels=num_domain_labels,
            enable_domain_head=enable_domain_head,
            domain_loss_coeff=domain_loss_coeff,
        )
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
                if not self.supports_domain_objective():
                    raise ValueError("D3GModel requires a domain head when pred_domain_for_d3g is True.")
                region_ids = self.domain_classifier(lr_features).argmax(dim=1)
                

        head_outputs = torch.stack(
            [head(hr_features) for head in self.task_classifiers],
            dim=1,
        )

        domain_weights = self.domain_weights(region_ids, lr_features)  # batch_size x num_domain_labels

        if self.training:
            task_logits = head_outputs[torch.arange(len(region_ids), device=region_ids.device), region_ids]
            consistency_weights = domain_weights.clone()
            consistency_weights[
                torch.arange(len(region_ids), device=region_ids.device),
                region_ids,
            ] = 0.0
            consistency_weights_sum = torch.clamp(consistency_weights.sum(dim=1), min=1e-6)
            rel_logits = torch.sum(consistency_weights * head_outputs, dim=1) / consistency_weights_sum
        else:
            weights_sum = torch.clamp(domain_weights.sum(dim=1), min=1e-6)  
            rel_logits = torch.sum(domain_weights * head_outputs, dim=1) / weights_sum
            task_logits = rel_logits

        outputs = {
            "task_logits": task_logits,
            "rel_logits": rel_logits,
        }
        if self.supports_domain_objective():
            outputs["domain_logits"] = self.domain_classifier(lr_features)
            outputs["domain_logits_detached"] = self.domain_classifier(lr_features.detach())
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