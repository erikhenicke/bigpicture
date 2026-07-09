"""Reusable model building blocks for multi-scale/multi-sensor classification.

Marks ``models.components`` as a package. The submodules provide the pieces
``models/multi_scale_classification.py`` composes into full models:

- ``branches``: per-scale encoders (HR FMoW-RGB / LR Landsat / location) and the
  ``*DualBranch`` wrappers that run an HR and an LR (or location/domain) encoder
  side by side.
- ``fusion``: mechanisms that combine an HR and an LR feature vector into a single
  representation (concatenation, FiLM, multiplicative similarity, GeoPrior) plus
  the parameter-free ``DecisionRule`` for late/decision fusion.
- ``domain_relations``: the domain-relation scoring modules used by the D3G
  (domain-gated feature fusion) model.
- ``spatial_encoding``: Fourier positional encoding of geographic coordinate grids.

No symbols are re-exported here; consumers import directly from the submodules
(e.g. ``from models.components.branches import Branch``).
"""
