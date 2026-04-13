from pathlib import Path
from typing import List, Tuple

import hydra
from hydra.utils import instantiate
from lightning import Trainer, seed_everything
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
import wandb

from models.late_fusion import LateFusionModule
from train.utils import make_multiscale_dataset, make_multiscale_loader, resolve_preprocessed_dir


def _resolve_preprocessed_dir(cfg: DictConfig) -> str | None:
    return resolve_preprocessed_dir(cfg.data.preprocessed_dir_default, cfg.data.preprocessed_dir_gaia)


def _make_loader(dataset: FMoWMultiScaleDataset, split: str, cfg: DictConfig, shuffle: bool) -> DataLoader:
    return make_multiscale_loader(
        dataset,
        split=split,
        frac=cfg.data.frac,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        shuffle=shuffle,
    )


def make_data_loaders(cfg: DictConfig) -> Tuple[DataLoader, List[DataLoader], List[DataLoader]]:
    preprocessed_dir = _resolve_preprocessed_dir(cfg)

    dataset_train = make_multiscale_dataset(
        fmow_dir=cfg.data.fmow_dir,
        landsat_dir=cfg.data.landsat_dir,
        preprocessed_dir=preprocessed_dir,
        augment=cfg.data.augment_train,
    )
    dataset_eval = make_multiscale_dataset(
        fmow_dir=cfg.data.fmow_dir,
        landsat_dir=cfg.data.landsat_dir,
        preprocessed_dir=preprocessed_dir,
    )

    train_loader = _make_loader(dataset_train, split=cfg.data.train_split, cfg=cfg, shuffle=True)
    val_loaders = [
        _make_loader(dataset_eval, split=split, cfg=cfg, shuffle=False)
        for split in cfg.data.val_splits
    ]
    test_loaders = [
        _make_loader(dataset_eval, split=split, cfg=cfg, shuffle=False)
        for split in cfg.data.test_splits
    ]
    return train_loader, val_loaders, test_loaders


def make_model(cfg: DictConfig) -> LateFusionModule:
    hr_encoder = instantiate(cfg.model.hr_encoder)
    lr_encoder = instantiate(cfg.model.lr_encoder)
    branches = instantiate(cfg.model.branches, hr_encoder=hr_encoder, lr_encoder=lr_encoder)
    fusion = instantiate(cfg.model.fusion)

    optimizer_factory = instantiate(cfg.optim.optimizer)
    scheduler_factory = instantiate(cfg.optim.scheduler)

    return LateFusionModule(
        branches=branches,
        fusion=fusion,
        optimizer=optimizer_factory,
        scheduler=scheduler_factory,
        num_labels=cfg.model.num_labels,
        region_index=cfg.model.region_index,
        ece_n_bins=cfg.model.ece_n_bins,
        val_loader_names=list(cfg.data.val_loader_names),
        test_loader_names=list(cfg.data.test_loader_names),
        key_metric=cfg.model.monitor_metric,
        compile=cfg.model.compile,
    )


@hydra.main(version_base=None, config_path="configs", config_name="run_experiment_lightning")
def run_experiment_lightning_hydra(cfg: DictConfig) -> None:
    seed_everything(cfg.seed, workers=True)

    wandb.login()
    wandb_logger = WandbLogger(
        project=cfg.wandb.project,
        name=cfg.wandb.run_name,
        log_model=cfg.wandb.log_model,
        config=OmegaConf.to_container(cfg, resolve=True),
    )

    train_loader, val_loaders, test_loaders = make_data_loaders(cfg)
    model = make_model(cfg)

    checkpoint_callback = ModelCheckpoint(
        monitor=cfg.model.monitor_metric,
        mode="max",
        save_top_k=1,
        save_last=True,
        filename="late-fusion-epoch{epoch:02d}",
    )

    default_root_dir = str(Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir))
    trainer = Trainer(
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        max_epochs=cfg.trainer.max_epochs,
        logger=wandb_logger,
        callbacks=[checkpoint_callback, LearningRateMonitor(logging_interval="epoch")],
        log_every_n_steps=cfg.trainer.log_every_n_steps,
        default_root_dir=default_root_dir,
    )

    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loaders)
    trainer.test(model=model, dataloaders=test_loaders, ckpt_path="best")


if __name__ == "__main__":
    run_experiment_lightning_hydra()
