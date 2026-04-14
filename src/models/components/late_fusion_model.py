from typing import List, Dict

import torch
import torch.nn as nn

from models.components.branches import DualBranch
from models.components.fusion import D3GFusion, Fusion

class LateFusionModel(nn.Module):
    def __init__(self, branches: DualBranch, fusion: Fusion, num_labels: int, domain_num_labels: int):
        super().__init__()

        self.branches: DualBranch = branches
        self.fusion: Fusion = fusion
        self.task_classifier: nn.Linear = nn.Linear(fusion.out_dim, num_labels)
        self.domain_classifier: nn.Linear = nn.Linear(branches.lr_encoder.out_dim, domain_num_labels)


    def forward(
        self,
        x: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        hr_branch_out, lr_branch_out = self.branches(x)
        fused = self.fusion(hr_branch_out, lr_branch_out)
        return {
            "task_logits": self.task_classifier(fused),
            "domain_logits": self.domain_classifier(lr_branch_out),
            "domain_logits_detached": self.domain_classifier(lr_branch_out.detach())
        } 
    
    def task_parameters(self) -> List[torch.nn.Parameter]:
        return list(self.branches.parameters()) + list(self.fusion.parameters()) + list(self.task_classifier.parameters())
    
    def domain_parameters(self) -> List[torch.nn.Parameter]:
        return list(self.domain_classifier.parameters())


class D3GModel(LateFusionModel):
    def __init__(self, branches: DualBranch, fusion: D3GFusion, num_labels: int, domain_num_labels: int):
        super().__init__(branches, fusion, num_labels, domain_num_labels)

    def forward(self, x: Dict[str, torch.Tensor]):
        """
        TODO: If training, consistency loss. If not training, use all heads and average.    
        """
        hr_branch_out, lr_branch_out = self.branches(x)
        fused = self.fusion(hr_branch_out, lr_branch_out)
        return {
            "task_logits": self.task_classifier(fused), 
            "domain_logits": self.domain_classifier(lr_branch_out),
            "domain_logits_detached": self.domain_classifier(lr_branch_out.detach())
        }