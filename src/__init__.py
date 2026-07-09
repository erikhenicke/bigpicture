"""Root package for the thesis codebase on multi-scale/multi-sensor satellite image classification.

Fuses FMoW-WILDS high-resolution imagery with broader-scale Landsat data for
land-use classification. Relevant submodules for this batch of files:
    `dataset`: `FMoWMultiScaleDataset`, the paired FMoW/Landsat PyTorch Dataset
        (see `dataset.fmow_multiscale_dataset`).
    `dataset_creation`: Scripts to build, preprocess and download the paired
        FMoW+Landsat data.
    `models`: Model architectures (single-branch and fusion variants) and
        their shared components.
    `statistics`: Dataset statistics and analysis utilities (e.g.
        `statistics.average_class_extent`).
    `slurm`: SLURM job script generation for cluster training runs
        (`slurm.generate`).
    `train`: Hydra-driven training entrypoint (`train.run_experiment`) and
        supporting dataset/loader construction utilities (`train.utils`).
    `results`: Result analysis, evaluation and plotting utilities.
"""
