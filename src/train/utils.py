from __future__ import annotations

from torch.utils.data import DataLoader

from dataset.fmow_multiscale_dataset import FMoWMultiScaleDataset, collate_multiscale


DEFAULT_FMOW_DIR = "/home/henicke/data"
DEFAULT_LANDSAT_DIR = "/home/datasets4/FMoW_LandSat"
DEFAULT_NUM_WORKERS = 4


def make_multiscale_dataset(
    fmow_dir: str = DEFAULT_FMOW_DIR,
    landsat_dir: str = DEFAULT_LANDSAT_DIR,
    source: str = "raw",
    preprocessed_dir: str | None = None,
    augment: bool = False,
    image_norm: str = "fmow-statistics",
    spatial_coord_grid: bool = False,
    spatial_overlap_mask: bool = False,
    overlap_mask_type: str = "binary",
    lr_extension_factor: float | None = None,
    hr_feature_run: str | None = None,
    lr_feature_run: str | None = None,
    feature_run_idx: int | None = None,
) -> FMoWMultiScaleDataset:
    return FMoWMultiScaleDataset(
        fmow_dir=fmow_dir,
        landsat_dir=landsat_dir,
        source=source,
        preprocessed_dir=preprocessed_dir,
        augment=augment,
        image_norm=image_norm,
        spatial_coord_grid=spatial_coord_grid,
        spatial_overlap_mask=spatial_overlap_mask,
        overlap_mask_type=overlap_mask_type,
        lr_extension_factor=lr_extension_factor,
        hr_feature_run=hr_feature_run,
        lr_feature_run=lr_feature_run,
        feature_run_idx=feature_run_idx,
    )


def make_multiscale_loader(
    dataset: FMoWMultiScaleDataset,
    split: str,
    frac: float,
    batch_size: int,
    num_workers: int = DEFAULT_NUM_WORKERS,
    shuffle: bool = False,
) -> DataLoader:
    return DataLoader(
        dataset.get_subset(split, frac=frac),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_multiscale,
    )
