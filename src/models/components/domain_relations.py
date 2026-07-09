"""Domain-relation scoring modules for D3G (domain-gated feature fusion).

Defines the per-domain-head "relation" score that :class:`~models.components.
fusion_models.D3GModel` uses to mix its per-domain task-classifier heads:
:class:`IndicatorDomainRelation` is a fixed 0/1 match between a sample's domain
and a head's domain, and :class:`D3GRelation` optionally blends that fixed
indicator with a learned cosine-similarity relation (between the LR branch
features and a learned per-domain embedding) via ``learnable_relation_coeff``.
``D3GRelation`` composes ``IndicatorDomainRelation`` internally.
"""
import torch
import torch.nn as nn


class IndicatorDomainRelation(nn.Module):
    """Fixed 0/1 relation: 1 if a sample's domain equals the query domain, else 0."""

    def __init__(self):
        """Initialize the module (no parameters or buffers)."""
        super().__init__()
    
    def forward(self, queries, key):
        """Compare each sample's domain id against a single query domain id.

        Args:
            queries (torch.Tensor): Shape ``(batch_size,)``, integer dtype --
                per-sample domain (region) ids.
            key (torch.Tensor): 0-dim integer tensor -- the domain id to compare
                against (e.g. a task-classifier head's domain index).

        Returns:
            torch.Tensor: Shape ``(batch_size, 1)``, float32. 1.0 where
            ``queries == key``, 0.0 elsewhere.
        """
        return torch.eq(queries, key).float().unsqueeze(-1)


class D3GRelation(nn.Module):
    """Per-domain-head relation score mixing a fixed indicator and a learned term.

    Used by ``D3GModel`` (one instance shared across all domain heads) to weight
    each per-domain task-classifier head's contribution for a given sample. The
    fixed term (:class:`IndicatorDomainRelation`) is 1 exactly when the sample's
    ground-truth domain matches the head's domain. When ``learnable_relation_coeff
    < 1``, this is blended with a learned term: the cosine similarity, in a shared
    ``internal_dim``-dimensional space, between the LR branch features (mapped
    through ``nnet_latlon``) and a learned embedding of the head's domain id
    (mapped through ``nnet``, an ``Embedding`` + MLP), both scaled elementwise by a
    shared learned ``weights`` vector before the cosine similarity is taken. This
    lets a sample partially "borrow" a nearby domain's head instead of only ever
    using its own domain's head.
    """

    def __init__(
        self,
        learnable_relation_coeff: float,
        internal_dim: int,
        lr_features_dim: int,
        num_domains: int,
    ):
        """Build the fixed indicator relation and, if needed, the learned relation.

        Args:
            learnable_relation_coeff (float): Weight on the fixed indicator
                relation in ``[0, 1]``; ``1 - learnable_relation_coeff`` weights
                the learned relation. If ``== 1``, the learned branch
                (``nnet``, ``nnet_latlon``, ``weights``, ``cos``) is not built and
                :meth:`forward` short-circuits to the fixed relation alone.
            internal_dim (int): Dimensionality of the shared embedding space in
                which the learned cosine similarity is computed.
            lr_features_dim (int): Feature dimension of the LR branch output fed
                into ``nnet_latlon``.
            num_domains (int): Number of discrete domains, i.e. the size of the
                ``nnet`` embedding table.
        """
        super().__init__()
        self.fixed_relations = IndicatorDomainRelation()
        self.learnable_relation_coeff = learnable_relation_coeff
        self.nnet_latlon = None

        if self.learnable_relation_coeff < 1:
            self.nnet = nn.Sequential(
                nn.Embedding(num_domains, internal_dim // 2),
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
        """Compute D3G relations for a batch of samples and a specific head.

        Args:
            domain_id (torch.Tensor): Shape ``(batch_size,)``, integer dtype --
                each sample's ground-truth domain (region) id.
            lr_features (torch.Tensor): Shape ``(batch_size, lr_features_dim)``,
                float32 -- LR branch features for the batch, used only by the
                learned relation term.
            head_id (torch.Tensor): 0-dim integer tensor -- the domain id of the
                task-classifier head this relation score is being computed for.

        Returns:
            torch.Tensor: Shape ``(batch_size, 1)``, float32. Relation score per
            sample: the fixed indicator alone if ``learnable_relation_coeff == 1``,
            otherwise ``learnable_relation_coeff * fixed + (1 -
            learnable_relation_coeff) * learned``, where ``learned`` is the cosine
            similarity described in the class docstring.
        """
        fixed_relations = self.fixed_relations(domain_id, head_id)
        if self.learnable_relation_coeff == 1:
            return fixed_relations

        head_ids = head_id.repeat(domain_id.shape[0])
        learned_relations = self.cos(self.weights * self.nnet_latlon(lr_features), self.weights * self.nnet(head_ids)).unsqueeze(-1)
        
        return self.learnable_relation_coeff * fixed_relations + (1. - self.learnable_relation_coeff) * learned_relations