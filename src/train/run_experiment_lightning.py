from functools import partial

import torch
from lightning import Trainer, seed_everything
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from torch.optim import AdamW
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
import wandb

from models.components.branches import DeitBranch, DualBranch
from models.components.fusion import ConcatFusion
from models.late_fusion import LateFusionModule
from train.utils import (
    make_multiscale_dataset,
    make_multiscale_loader,
    resolve_preprocessed_dir,
)

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
MONITOR_METRIC = "val/val-od-worst-group-task-acc"


def make_data_loaders():
    preprocessed_dir = resolve_preprocessed_dir(None)

    dataset_train = make_multiscale_dataset(preprocessed_dir=preprocessed_dir, augment=True)
    dataset_eval = make_multiscale_dataset(preprocessed_dir=preprocessed_dir)

    train_loader = make_multiscale_loader(
        dataset_train,
        split="train",
        frac=FRAC,
        batch_size=BATCH_SIZE,
        num_workers=DATA_LOADER_NUM_WORKERS,
        shuffle=True,
    )
    val_loaders = [
        make_multiscale_loader(
            dataset_eval,
            split="val",
            frac=FRAC,
            batch_size=BATCH_SIZE,
            num_workers=DATA_LOADER_NUM_WORKERS,
        ),
        make_multiscale_loader(
            dataset_eval,
            split="id_val",
            frac=FRAC,
            batch_size=BATCH_SIZE,
            num_workers=DATA_LOADER_NUM_WORKERS,
        ),
    ]
    test_loaders = [
        make_multiscale_loader(
            dataset_eval,
            split="test",
            frac=FRAC,
            batch_size=BATCH_SIZE,
            num_workers=DATA_LOADER_NUM_WORKERS,
        ),
        make_multiscale_loader(
            dataset_eval,
            split="id_test",
            frac=FRAC,
            batch_size=BATCH_SIZE,
            num_workers=DATA_LOADER_NUM_WORKERS,
        ),
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
    domain_optimizer_factory = partial(Adam, lr=LR * 0.1, weight_decay=0.0)
    domain_scheduler_factory = partial(StepLR, step_size=1, gamma=0.96)

    model = LateFusionModule(
        branches=branches,
        fusion=fusion,
        optimizer=optimizer_factory,
        scheduler=scheduler_factory,
        domain_optimizer=domain_optimizer_factory,
        domain_scheduler=domain_scheduler_factory,
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
