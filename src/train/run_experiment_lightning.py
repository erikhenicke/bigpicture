from functools import partial
import platform

import torch
from lightning import Trainer, seed_everything
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
import wandb

from dataset.fmow_multiscale_dataset import FMoWMultiScaleDataset, collate_multiscale
from models.components.branches import DeitBranch, DualBranch
from models.components.fusion import ConcatFusion
from models.late_fusion import LateFusionModule

# Hardcoded test configuration for first Lightning integration run.
PROJECT_NAME = "fmow"
RUN_NAME = "late-fusion-lightning-test"
SEED = 111
NUM_CLASSES = 62
BATCH_SIZE = 32
EPOCHS = 20
FRAC = 1.0 
LANDSAT_IN_CHANNELS = 6
DATA_LOADER_NUM_WORKERS = 4
LR = 5e-5
WEIGHT_DECAY = 0.0
LR_DECAY = 0.5
PLATEAU_PATIENCE = 5 
ECE_N_BINS = 10
MONITOR_METRIC = "val-od-worst-group-task-acc"


def get_data_loader(dataset: FMoWMultiScaleDataset, split: str, shuffle: bool = False) -> DataLoader:
    return DataLoader(
        dataset.get_subset(split, frac=FRAC),
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=DATA_LOADER_NUM_WORKERS,
        collate_fn=collate_multiscale,
    )


def make_data_loaders():
    fmow_dir = "/home/henicke/data"
    landsat_dir = "/home/datasets4/FMoW_LandSat"
    preprocessed_dir = None

    if platform.node() in {"gaia4", "gaia5"}:
        preprocessed_dir = "/data/henicke/FMoW_LandSat"

    dataset_train = FMoWMultiScaleDataset(
        fmow_dir=fmow_dir,
        landsat_dir=landsat_dir,
        preprocessed_dir=preprocessed_dir,
        augment=True,
    )

    dataset_eval = FMoWMultiScaleDataset(
        fmow_dir=fmow_dir,
        landsat_dir=landsat_dir,
        preprocessed_dir=preprocessed_dir,
    )

    train_loader = get_data_loader(dataset_train, split="train", shuffle=True)
    val_loaders = [
        get_data_loader(dataset_eval, split="val", shuffle=False),
        get_data_loader(dataset_eval, split="id_val", shuffle=False),
    ]
    test_loaders = [
        get_data_loader(dataset_eval, split="test", shuffle=False),
        get_data_loader(dataset_eval, split="id_test", shuffle=False),
    ]
    return train_loader, val_loaders, test_loaders


def make_model() -> LateFusionModule:
    hr_encoder = DeitBranch(in_channels=3, image_net=True)
    lr_encoder = DeitBranch(in_channels=LANDSAT_IN_CHANNELS, image_net=True)
    branches = DualBranch(hr_encoder=hr_encoder, lr_encoder=lr_encoder, landsat_channels=LANDSAT_IN_CHANNELS)

    # DeiT tiny CLS embedding dim is 192 for each branch.
    fusion = ConcatFusion(hr_dim=192, lr_dim=192, out_dim=256)

    optimizer_factory = partial(AdamW, lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler_factory = partial(
        ReduceLROnPlateau,
        mode="max",
        factor=LR_DECAY,
        patience=PLATEAU_PATIENCE,
    )

    model = LateFusionModule(
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
    return model


def run_experiment_lightning() -> None:
    seed_everything(SEED, workers=True)

    wandb.login()
    wandb_logger = WandbLogger(
        project=PROJECT_NAME,
        name=RUN_NAME,
        log_model=False,
        config={
            "seed": SEED,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "frac": FRAC,
            "learning_rate": LR,
            "weight_decay": WEIGHT_DECAY,
            "ece_n_bins": ECE_N_BINS,
            "monitor_metric": MONITOR_METRIC,
            "model": "late_fusion_deit_deit_concat",
        },
    )

    train_loader, val_loaders, test_loaders = make_data_loaders()
    model = make_model()

    checkpoint_callback = ModelCheckpoint(
        monitor=MONITOR_METRIC,
        mode="max",
        save_top_k=1,
        save_last=True,
        filename="late-fusion-epoch{epoch:02d}",
    )

    trainer = Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        max_epochs=EPOCHS,
        logger=wandb_logger,
        callbacks=[checkpoint_callback, LearningRateMonitor(logging_interval="epoch")],
        log_every_n_steps=25,
    )

    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loaders)
    trainer.test(model=model, dataloaders=test_loaders, ckpt_path="best")


if __name__ == "__main__":
    run_experiment_lightning()
