from __future__ import annotations

import argparse
from functools import partial
from pathlib import Path

import torch
from lightning import Trainer, seed_everything
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from models.components.branches import DeitBranch, DualBranch
from models.components.fusion import ConcatFusion
from models.late_fusion import LateFusionModule
from train.utils import make_multiscale_dataset, make_multiscale_loader, resolve_preprocessed_dir as resolve_platform_preprocessed_dir


PROJECT_NAME = "fmow"
RUN_NAME = "late-fusion-lightning-eval"
SEED = 111
NUM_CLASSES = 62
BATCH_SIZE = 32
FRAC = 1.0
LANDSAT_IN_CHANNELS = 6
DATA_LOADER_NUM_WORKERS = 4
LR = 5e-5
WEIGHT_DECAY = 0.0
LR_DECAY = 0.5
PLATEAU_PATIENCE = 5
ECE_N_BINS = 10
MONITOR_METRIC = "val-od-worst-group-task-acc"


def make_data_loaders(
    fmow_dir: str,
    landsat_dir: str,
    preprocessed_dir: str | None,
    frac: float,
    batch_size: int,
    num_workers: int,
) -> list:
    dataset_eval = make_multiscale_dataset(
        fmow_dir=fmow_dir,
        landsat_dir=landsat_dir,
        preprocessed_dir=preprocessed_dir,
    )

    return [
        make_multiscale_loader(
            dataset_eval,
            split="test",
            frac=frac,
            batch_size=batch_size,
            num_workers=num_workers,
        ),
        make_multiscale_loader(
            dataset_eval,
            split="id_test",
            frac=frac,
            batch_size=batch_size,
            num_workers=num_workers,
        ),
    ]


def make_model() -> LateFusionModule:
    hr_encoder = DeitBranch(in_channels=3, image_net=True)
    lr_encoder = DeitBranch(in_channels=LANDSAT_IN_CHANNELS, image_net=True)
    branches = DualBranch(hr_encoder=hr_encoder, lr_encoder=lr_encoder, landsat_channels=LANDSAT_IN_CHANNELS)

    fusion = ConcatFusion(hr_dim=192, lr_dim=192, out_dim=256)

    optimizer_factory = partial(AdamW, lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler_factory = partial(
        ReduceLROnPlateau,
        mode="max",
        factor=LR_DECAY,
        patience=PLATEAU_PATIENCE,
    )

    return LateFusionModule(
        branches=branches,
        fusion=fusion,
        optimizer=optimizer_factory,
        scheduler=scheduler_factory,
        num_labels=NUM_CLASSES,
        region_index=0,
        ece_n_bins=ECE_N_BINS,
        val_loader_names=["val-od", "val-id"],
        test_loader_names=["test-od", "test-id"],
        key_metric=MONITOR_METRIC,
        compile=False,
    )


def load_model_from_checkpoint(checkpoint_path: str) -> LateFusionModule:
    model_template = make_model()
    return LateFusionModule.load_from_checkpoint(
        checkpoint_path,
        branches=model_template.branches,
        fusion=model_template.fusion,
        optimizer=model_template.optimizer_factory,
        scheduler=model_template.scheduler_factory,
        num_labels=model_template.hparams.num_labels,
        region_index=model_template.hparams.region_index,
        ece_n_bins=model_template.hparams.ece_n_bins,
        val_loader_names=list(model_template.val_loader_names),
        test_loader_names=list(model_template.test_loader_names),
        key_metric=model_template.hparams.key_metric,
        compile=False,
        map_location="cpu",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a Lightning checkpoint on FMoW test data.")
    parser.add_argument("checkpoint_path", type=str, help="Path to a .ckpt file created by run_experiment_lightning.py")
    parser.add_argument("--fmow-dir", type=str, default="/home/henicke/data", help="Root directory containing FMoW data")
    parser.add_argument(
        "--landsat-dir",
        type=str,
        default="/home/datasets4/FMoW_LandSat",
        help="Root directory containing Landsat data",
    )
    parser.add_argument(
        "--preprocessed-dir",
        type=str,
        default=None,
        help="Optional directory with preprocessed fmow_preprocessed assets",
    )
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Evaluation batch size")
    parser.add_argument("--num-workers", type=int, default=DATA_LOADER_NUM_WORKERS, help="DataLoader workers")
    parser.add_argument("--frac", type=float, default=FRAC, help="Fraction of each split to evaluate")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed")
    return parser.parse_args()


def resolve_checkpoint_path(path: Path) -> Path:
    if path.is_file():
        return path

    if not path.is_dir():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    candidates = list(path.rglob("best*.ckpt"))
    if not candidates:
        candidates = list(path.rglob("last.ckpt"))
    if not candidates:
        candidates = list(path.rglob("*.ckpt"))
    if not candidates:
        raise FileNotFoundError(f"No .ckpt files found under: {path}")

    return max(candidates, key=lambda candidate: candidate.stat().st_mtime)


def evaluate_lightning_checkpoint() -> None:
    args = parse_args()
    seed_everything(args.seed, workers=True)

    checkpoint_path = resolve_checkpoint_path(Path(args.checkpoint_path))

    preprocessed_dir = resolve_platform_preprocessed_dir(args.preprocessed_dir, "/data/henicke/FMoW_LandSat")
    test_loaders = make_data_loaders(
        fmow_dir=args.fmow_dir,
        landsat_dir=args.landsat_dir,
        preprocessed_dir=preprocessed_dir,
        frac=args.frac,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    model = load_model_from_checkpoint(str(checkpoint_path))

    trainer = Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        log_every_n_steps=25,
    )

    results = trainer.test(model=model, dataloaders=test_loaders, ckpt_path=None)
    print(results)


if __name__ == "__main__":
    evaluate_lightning_checkpoint()