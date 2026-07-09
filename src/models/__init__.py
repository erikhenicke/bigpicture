"""
models package

Model architectures and training utilities for multi-scale FMoW-WILDS /
Landsat classification.

Submodules:
    multi_scale_classification: `MultiScaleClassificationModule`, the Lightning
        training/eval harness wrapping any `MultiScaleModel` (see
        `models.components.fusion_models`).
    schedulers: `LinearWarmupCosineAnnealingLR`, a Hydra-configurable
        learning-rate scheduler.
    utils: Evaluation-state bookkeeping and metric computation (task accuracy,
        domain-classifier accuracy, calibration error, per-region/per-class and
        branch-ablation breakdowns) shared by the training module.
    components: Building blocks (encoders/branches, fusion strategies, domain
        relations, spatial encodings) composed into the concrete
        `MultiScaleModel` subclasses.
"""
