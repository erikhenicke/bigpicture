import torch
import torch.nn as nn

from models.utils import REGIONS

class IndicatorDomainRelation(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, queries, key):
        return torch.eq(queries, key).float().unsqueeze(-1)


class D3GRelation(nn.Module):
    def __init__(
        self,
        learnable_relation_coeff: float,
        internal_dim: int,
        lr_features_dim: int,
    ):
        super().__init__()
        self.fixed_relations = IndicatorDomainRelation()
        self.learnable_relation_coeff = learnable_relation_coeff
        self.nnet_latlon = None

        if self.learnable_relation_coeff < 1:
            self.nnet = nn.Sequential(
                nn.Embedding(len(REGIONS), internal_dim // 2),
                nn.ReLU(),
                nn.Linear(internal_dim // 2, internal_dim)
            )

            self.nnet_latlon = nn.Sequential(
                nn.Linear(lr_features_dim, internal_dim // 2),
                nn.ReLU(),
                nn.Linear(internal_dim // 2, internal_dim)
            )

            self.weights = nn.Parameter(torch.rand(internal_dim))
            self.cos = nn.CosineSimilarity()
    
    def forward(self, domain_id, lr_features, head_id):
        """Compute D3G relations for a batch of samples and a specific head."""
        fixed_relations = self.fixed_relations(domain_id, head_id)
        if self.learnable_relation_coeff == 1:
            return fixed_relations

        head_ids = head_id.repeat(domain_id.shape[0])
        learned_relations = self.cos(self.weights * self.nnet_latlon(lr_features), self.weights * self.nnet(head_ids)).unsqueeze(-1)
        
        return self.learnable_relation_coeff * fixed_relations + (1. - self.learnable_relation_coeff) * learned_relations