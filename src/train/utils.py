"""Convenience constructors for `FMoWMultiScaleDataset` and its `DataLoader`.

Thin wrappers used by `run_experiment.py` to build the training/eval dataset
and per-split `DataLoader`s from Hydra config values, with defaults matching
this codebase's typical host paths.

Functions:
    `make_multiscale_dataset`: Construct an `FMoWMultiScaleDataset` with
        keyword defaults.
    `make_multiscale_loader`: Wrap a WILDS split subset of the dataset in a
        `DataLoader` using `collate_multiscale`.
"""

from __future__ import annotations

from torch.utils.data import DataLoader

from dataset.fmow_multiscale_dataset import FMoWMultiScaleDataset, collate_multiscale


DEFAULT_FMOW_DIR = "/home/henicke/data"
DEFAULT_LANDSAT_DIR = "/home/datasets4/FMoW_LandSat"
DEFAULT_NUM_WORKERS = 4
DEFAULT_LR_EXTENSION_FACTOR = 3.0


def make_multiscale_dataset(
    fmow_dir: str = DEFAULT_FMOW_DIR,
    landsat_dir: str = DEFAULT_LANDSAT_DIR,
    source: str = "raw",
    preprocessed_dir: str | None = None,
    augment: bool = False,
    image_norm: str = "fmow-statistics",
    lr_crop_km: float | None = None,
    spatial_coord_grid: bool = False,
    spatial_overlap_mask: bool = False,
    overlap_mask_type: str = "binary",
    lr_extension_factor: float = DEFAULT_LR_EXTENSION_FACTOR,
    hr_feature_run_name: str | None = None,
    lr_feature_run_name: str | None = None,
    feature_run_idx: int | None = None,
    leave_asia_out: bool = False,
) -> FMoWMultiScaleDataset:
    """Construct an `FMoWMultiScaleDataset` with this project's default paths.

    Forwards every argument to `FMoWMultiScaleDataset.__init__` unchanged;
    see that class for the full semantics of each parameter (source modes,
    spatial encodings, feature-cache options, etc.).

    Args:
        fmow_dir (str): Root directory of the FMoW-WILDS dataset. Defaults
            to `DEFAULT_FMOW_DIR`.
        landsat_dir (str): Root directory of the raw Landsat images.
            Defaults to `DEFAULT_LANDSAT_DIR`.
        source (str): Input mode: `"raw"`, `"preprocessed"`, or
            `"features"`. Defaults to `"raw"`.
        preprocessed_dir (str | None): Base dir name for
            `source="preprocessed"`. Defaults to `None`.
        augment (bool): Whether to apply random flip augmentation. Defaults
            to `False`.
        image_norm (str): Normalization statistics scheme; see
            `FMoWMultiScaleDataset.get_default_transform_rgb`/
            `get_default_transform_landsat`. Defaults to
            `"fmow-statistics"`.
        lr_crop_km (float | None): Optional spatial-extent crop (km) for the
            LR branch; only valid with `source="preprocessed"`. Defaults to
            `None`.
        spatial_coord_grid (bool): Whether to emit per-pixel coordinate grid
            tensors. Defaults to `False`.
        spatial_overlap_mask (bool): Whether to emit an HR/LR overlap mask
            tensor. Defaults to `False`.
        overlap_mask_type (str): `"binary"` or `"gaussian"` overlap mask
            shape. Defaults to `"binary"`.
        lr_extension_factor (float): Multiplier from max HR span to the
            full LR (Landsat) footprint. Defaults to
            `DEFAULT_LR_EXTENSION_FACTOR`.
        hr_feature_run_name (str | None): Cached-feature run name for the
            HR branch, when `source="features"`. Defaults to `None`.
        lr_feature_run_name (str | None): Cached-feature run name for the
            LR branch, when `source="features"`. Defaults to `None`.
        feature_run_idx (int | None): Rerun index (0/1/2) to load, required
            when `source="features"`. Defaults to `None`.
        leave_asia_out (bool): Whether to convert the split scheme into a
            leave-Asia-out geographic split. Defaults to `False`.

    Returns:
        FMoWMultiScaleDataset: The constructed dataset.
    """
    return FMoWMultiScaleDataset(
        fmow_dir=fmow_dir,
        landsat_dir=landsat_dir,
        source=source,
        preprocessed_dir=preprocessed_dir,
        augment=augment,
        image_norm=image_norm,
        lr_crop_km=lr_crop_km,
        spatial_coord_grid=spatial_coord_grid,
        spatial_overlap_mask=spatial_overlap_mask,
        overlap_mask_type=overlap_mask_type,
        lr_extension_factor=lr_extension_factor,
        hr_feature_run_name=hr_feature_run_name,
        lr_feature_run_name=lr_feature_run_name,
        feature_run_idx=feature_run_idx,
        leave_asia_out=leave_asia_out,
    )


def make_multiscale_loader(
    dataset: FMoWMultiScaleDataset,
    split: str,
    frac: float,
    batch_size: int,
    num_workers: int = DEFAULT_NUM_WORKERS,
    shuffle: bool = False,
) -> DataLoader:
    """Wrap a WILDS split subset of `dataset` in a `DataLoader`.

    Args:
        dataset (FMoWMultiScaleDataset): Dataset to draw the subset from.
        split (str): WILDS split name to select via `dataset.get_subset`
            (e.g. `"train"`, `"id_val"`, `"ood_test"`).
        frac (float): Fraction of the split to use (forwarded to
            `WILDSDataset.get_subset`), e.g. for quick debug runs on a
            subset of the data.
        batch_size (int): Batch size for the `DataLoader`.
        num_workers (int): Number of worker processes for data loading.
            Defaults to `DEFAULT_NUM_WORKERS`.
        shuffle (bool): Whether to shuffle samples each epoch. Defaults to
            `False`.

    Returns:
        DataLoader: Loader over `dataset`'s `split` subset, batched with
        `collate_multiscale`.
    """
    return DataLoader(
        dataset.get_subset(split, frac=frac),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_multiscale,
    )
