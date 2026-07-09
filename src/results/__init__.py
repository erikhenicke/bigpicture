"""Evaluation, metrics, and plotting scripts for trained FMoW/Landsat checkpoints.

This package turns Hydra run directories and Lightning checkpoints under
``log/runs`` into the result tables and figures used in the thesis. Given an
eval YAML (``src/train/configs/eval/*.yaml``) describing groups of runs to
compare, the scripts here resolve each run reference to its log directory,
load its per-seed test metrics, and render tables or plots:

- ``utils``: shared helpers for resolving eval-YAML run references to log
  directories, loading per-seed/aggregated test metrics and Hydra configs,
  and formatting experiment/metric display names from ``translations.yaml``.
- ``eval_metrics``: loads per-seed test metrics (and checkpoint parameter
  counts) and renders per-group HTML (great_tables) or LaTeX result tables,
  including a combined LaTeX document.
- ``eval_classes``: per-class accuracy gain/loss bar plots comparing runs to
  a baseline, plain and occurrence-weighted, for the overall OOD split and
  the baseline's worst-performing region.
- ``decision_plots``: grouped bar plots comparing decision-fusion ablation
  modes (prior/domain on or off) across decision rules, plus per-backbone
  context and accuracy/robustness trade-off summary plots.
- ``eval_checkpoint``: loads a single Lightning checkpoint (with its Hydra
  config) and re-runs ``Trainer.test`` on the FMoW test data for ad hoc
  inspection.
- ``eval_reproduce``, ``extract_features``, ``per_seed_compare``,
  ``region_fusion_plot``, ``retrain_lr_domain``, ``spatial_extent_plots``:
  further evaluation, feature-extraction, and plotting scripts built on the
  same ``utils`` helpers.
"""
