import shutil
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import hydra
from hydra.utils import instantiate
from lightning import Trainer, seed_everything
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from omegaconf import DictConfig, OmegaConf
import torch
from torch.utils.data import DataLoader
import wandb

from dataset.fmow_multiscale_dataset import FMoWMultiScaleDataset
from models.late_fusion import LateFusionModule
from train.utils import make_multiscale_dataset, make_multiscale_loader, resolve_preprocessed_dir

def _has_device_tensor_cores() -> bool:
    """Check if the current GPU supports Tensor Cores: https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html"""
    if not torch.cuda.is_available():
        return False
    device = torch.device("cuda")
    major, _ = torch.cuda.get_device_capability(device)
    return major >= 7

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

    model_target = cfg.model.model.get("_target_", "")

    if model_target.endswith("SingleBranchModel"):
        hr_encoder = instantiate(cfg.model.hr_encoder)
        model = instantiate(
            cfg.model.model,
            encoder=hr_encoder,
            num_task_labels=cfg.num_task_labels,
            num_domain_labels=cfg.num_domain_labels,
        )
    else:
        hr_encoder = instantiate(cfg.model.hr_encoder)
        lr_encoder = instantiate(cfg.model.lr_encoder)
        branches = instantiate(cfg.model.branches, hr_encoder=hr_encoder, lr_encoder=lr_encoder)
        if model_target.endswith("D3GModel"):
            model = instantiate(
                cfg.model.model,
                branches=branches,
                num_task_labels=cfg.num_task_labels,
                num_domain_labels=cfg.num_domain_labels,
                enable_domain_head=cfg.model.enable_domain_head,
                domain_loss_coeff=cfg.model.domain_loss_coeff,
                learnable_relation_coeff=cfg.model.learnable_relation_coeff,
                consistency_loss_coeff=cfg.model.d3g_loss_coeff,
                pred_domain_for_d3g=cfg.model.pred_domain_for_d3g,
            )
        else:
            fusion = instantiate(cfg.model.fusion)
            model = instantiate(
                cfg.model.model,
                branches=branches,
                fusion=fusion,
                num_task_labels=cfg.num_task_labels,
                num_domain_labels=cfg.num_domain_labels,
                enable_domain_head=cfg.model.enable_domain_head,
                domain_loss_coeff=cfg.model.domain_loss_coeff,
            )

    optimizer_factory = instantiate(cfg.optim.optimizer)
    scheduler_factory = instantiate(cfg.optim.scheduler) if cfg.optim.scheduler is not None else None
    domain_optimizer_factory = None
    domain_scheduler_factory = None
    if cfg.model.enable_domain_head:
        domain_optimizer_factory = instantiate(
            cfg.optim.domain_optimizer,
            lr=cfg.optim.optimizer.lr * cfg.optim.domain_optimizer_lr_factor,
        )
        domain_scheduler_factory = (
            instantiate(cfg.optim.domain_scheduler)
            if cfg.optim.domain_scheduler is not None
            else None
        )

    return LateFusionModule(
        model=model,
        optimizer=optimizer_factory,
        scheduler=scheduler_factory,
        domain_optimizer=domain_optimizer_factory,
        domain_scheduler=domain_scheduler_factory,
        num_task_labels=cfg.num_task_labels,
        num_domain_labels=cfg.num_domain_labels,
        domain_index=cfg.domain_index,
        ece_n_bins=cfg.trainer.ece_n_bins,
        val_loader_names=list(cfg.data.val_loader_names),
        test_loader_names=list(cfg.data.test_loader_names),
        key_metric=cfg.trainer.monitor_metric,
        compile=cfg.trainer.compile,
        label_smoothing=cfg.trainer.label_smoothing,
    )


def _run_once(
    cfg: DictConfig,
    run_idx: int,
    default_root_dir: str,
    wandb_group: str,
) -> tuple[list[dict], str]:
    """Run one training + test pass. Returns (test_results, checkpoint_dir)."""
    seed_everything(cfg.seed + run_idx, workers=True)

    run_name = f"{wandb_group}-run{run_idx}"
    checkpoint_dir = f"{default_root_dir}/checkpoints/run{run_idx}"

    wandb_logger = WandbLogger(
        project=cfg.wandb.project,
        name=run_name,
        group=wandb_group,
        log_model=cfg.wandb.log_model,
        config=OmegaConf.to_container(cfg, resolve=True),
    )
    csv_logger = CSVLogger(save_dir=default_root_dir, name=f"run{run_idx}")

    train_loader, val_loaders, test_loaders = make_data_loaders(cfg)
    model = make_model(cfg)

    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        monitor=cfg.trainer.monitor_metric,
        mode="max",
        save_top_k=1,
        save_last=True,
        filename="late-fusion-epoch{epoch:02d}",
    )

    trainer = Trainer(
        accelerator=cfg.trainer.accelerator,
        max_epochs=cfg.trainer.max_epochs,
        logger=[wandb_logger, csv_logger],
        callbacks=[checkpoint_callback, LearningRateMonitor(logging_interval="epoch")],
        log_every_n_steps=cfg.trainer.log_every_n_steps,
        default_root_dir=default_root_dir,
    )

    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loaders)
    test_results = trainer.test(model=model, dataloaders=test_loaders, ckpt_path="best")

    wandb.finish()

    return test_results, checkpoint_dir


def _best_run_score(test_results: list[dict], metric: str) -> float:
    for result_dict in test_results:
        if metric in result_dict:
            return result_dict[metric]
    raise ValueError(f"Metric '{metric}' not found in test results: {test_results}")


@hydra.main(version_base=None, config_path="configs", config_name="setup")
def run_experiment(cfg: DictConfig) -> None:
    if _has_device_tensor_cores():
        torch.set_float32_matmul_precision("medium")

    default_root_dir = str(Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir))

    wandb.login()

    timestamp = datetime.now().strftime("%y-%m-%d-%H-%M-%S")
    wandb_group = f"{cfg.run_name}-{timestamp}"

    num_reruns = cfg.get("num_reruns", 1)
    best_run_metric = cfg.trainer.get("best_run_metric", "test/test-od-worst-group-task-acc")

    run_results: list[tuple[float, str]] = []
    for run_idx in range(num_reruns):
        test_results, checkpoint_dir = _run_once(cfg, run_idx, default_root_dir, wandb_group)
        score = _best_run_score(test_results, best_run_metric)
        run_results.append((score, checkpoint_dir))
        print(f"Run {run_idx}: {best_run_metric} = {score:.4f}")

    if num_reruns > 1:
        best_score, best_dir = max(run_results, key=lambda x: x[0])
        print(f"\nBest run: {best_dir} ({best_run_metric} = {best_score:.4f})")
        for score, ckpt_dir in run_results:
            if ckpt_dir != best_dir:
                shutil.rmtree(ckpt_dir, ignore_errors=True)
                print(f"Deleted checkpoints: {ckpt_dir}")


if __name__ == "__main__":
    run_experiment()
