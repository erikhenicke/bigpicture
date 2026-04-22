from __future__ import annotations

import platform

from torch.utils.data import DataLoader

from dataset.fmow_multiscale_dataset import FMoWMultiScaleDataset, collate_multiscale


DEFAULT_FMOW_DIR = "/home/henicke/data"
DEFAULT_LANDSAT_DIR = "/home/datasets4/FMoW_LandSat"
DEFAULT_PREPROCESSED_DIR = "FMoW_LandSat"
DEFAULT_NUM_WORKERS = 4


def resolve_preprocessed_dir(preprocessed_dir: str | None = DEFAULT_PREPROCESSED_DIR) -> str:
    if platform.node() in {"gaia4", "gaia5", "gaia6"}:
        return f"/data/henicke/{preprocessed_dir}"
    elif platform.node() in {"nyx"}:
        return f"/home/nyx_data1/henicke/{preprocessed_dir}"
    raise ValueError(f"Unknown host {platform.node()}, cannot resolve preprocessed_dir path.")


def make_multiscale_dataset(
    fmow_dir: str = DEFAULT_FMOW_DIR,
    landsat_dir: str = DEFAULT_LANDSAT_DIR,
    preprocessed_dir: str | None = None,
    augment: bool = False,
) -> FMoWMultiScaleDataset:
    return FMoWMultiScaleDataset(
        fmow_dir=fmow_dir,
        landsat_dir=landsat_dir,
        preprocessed_dir=preprocessed_dir,
        augment=augment,
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
