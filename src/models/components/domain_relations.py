import torch
import torch.nn as nn

from models.components.branches import Branch

class IndicatorDomainRelation(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, queries, key):
        return torch.eq(queries, key).float().unsqueeze(-1)


class D3GRelation(nn.Module):
    def __init__(self, beta, out_dim, lr_encoder: Branch = None):
        super().__init__()
        self.fixed_relations = IndicatorDomainRelation()
        self.beta = beta
        self.nnet_latlon = None

        if self.beta < 1:
            self.nnet = nn.Sequential(
                nn.Linear(1, out_dim // 2),
                nn.ReLU(),
                nn.Linear(out_dim // 2, out_dim)
            )

            if lr_encoder is not None:
                self.nnet_latlon = nn.Sequential(
                    lr_encoder,
                    nn.Linear(lr_encoder.out_dim, out_dim // 2),
                    nn.ReLU(),
                    nn.Linear(out_dim // 2, out_dim)
                )

            self.weights = nn.Parameter(torch.rand(out_dim))
            self.cos = nn.CosineSimilarity()
    
    def forward(self, queries, queries_latlon, key):
        fixed_relations = self.fixed_relations(queries, key)
        if self.beta == 1:
            return fixed_relations
        keys = key.repeat(queries.shape[0]).unsqueeze(-1).float()
        
        if self.nnet_latlon is not None:
            learned_relations = self.cos(self.weights * self.nnet_latlon(queries_latlon), self.weights * self.nnet(keys)).unsqueeze(-1)
        else:
            queries = queries.unsqueeze(-1).float()
            learned_relations = self.cos(self.weights * self.nnet(queries), self.weights * self.nnet(keys)).unsqueeze(-1)
        
        return self.beta * fixed_relations + (1. - self.beta) * learned_relations