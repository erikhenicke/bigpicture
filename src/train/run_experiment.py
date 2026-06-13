from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import hydra
from hydra.utils import instantiate
from lightning import Trainer, seed_everything
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from omegaconf import DictConfig, OmegaConf
import pandas as pd
import torch
from torch.utils.data import DataLoader
import wandb

from dataset.fmow_multiscale_dataset import FMoWMultiScaleDataset
from models.components.spatial_encoding import SpatialEncoding
from models.late_fusion import LateFusionModule
from results.utils import find_best_checkpoints, find_run_dir
from train.utils import make_multiscale_dataset, make_multiscale_loader

REPO_ROOT = Path(__file__).parent.parent.parent
DEFAULT_PRIOR_METADATA = REPO_ROOT / "data" / "rgb_metadata_extended.csv"

def has_device_tensor_cores() -> bool:
    """Check if the current GPU supports Tensor Cores: https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html"""
    if not torch.cuda.is_available():
        return False
    device = torch.device("cuda")
    major, _ = torch.cuda.get_device_capability(device)
    return major >= 7


def _make_loader(dataset: FMoWMultiScaleDataset, split: str, cfg: DictConfig, shuffle: bool) -> DataLoader:
    return make_multiscale_loader(
        dataset,
        split=split,
        frac=cfg.data.frac,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        shuffle=shuffle,
    )


def _parse_spatial_cfg(cfg: DictConfig):
    spatial_cfg = cfg.model.get("spatial_encoding", {})
    coord_channels = spatial_cfg.get("coord_channels", False)
    overlap_mask = spatial_cfg.get("overlap_mask", False)
    fourier_bands = spatial_cfg.get("fourier_bands", 0)
    fourier_proj_dim = spatial_cfg.get("fourier_proj_dim", 0)
    overlap_mask_type = spatial_cfg.get("overlap_mask_type", "binary")

    hr_extra, lr_extra = 0, 0
    if coord_channels:
        hr_extra += 2
        lr_extra += 2
    if overlap_mask:
        lr_extra += 1
    use_fourier = fourier_proj_dim > 0 and fourier_bands > 0
    if use_fourier:
        hr_extra += fourier_proj_dim
        lr_extra += fourier_proj_dim

    needs_coord_grid = coord_channels or use_fourier
    needs_overlap_mask = overlap_mask

    return {
        "coord_channels": coord_channels,
        "overlap_mask": overlap_mask,
        "overlap_mask_type": overlap_mask_type,
        "fourier_bands": fourier_bands,
        "fourier_proj_dim": fourier_proj_dim,
        "use_fourier": use_fourier,
        "hr_extra": hr_extra,
        "lr_extra": lr_extra,
        "needs_coord_grid": needs_coord_grid,
        "needs_overlap_mask": needs_overlap_mask,
    }


def make_data_loaders(cfg: DictConfig, run_idx: int) -> Tuple[DataLoader, List[DataLoader], List[DataLoader]]:
    sc = _parse_spatial_cfg(cfg)

    spatial_kwargs = dict(
        spatial_coord_grid=sc["needs_coord_grid"],
        spatial_overlap_mask=sc["needs_overlap_mask"],
        overlap_mask_type=sc["overlap_mask_type"],
    )

    dataset_train = make_multiscale_dataset(
        fmow_dir=cfg.data.fmow_dir,
        landsat_dir=cfg.data.landsat_dir,
        source=cfg.data.get("source", "preprocessed" if "preprocessed_dir" in cfg.data else "raw"),
        preprocessed_dir=cfg.data.preprocessed_dir,
        augment=cfg.data.augment_train,
        image_norm=cfg.data.image_norm,
        lr_crop_km=cfg.data.get("lr_crop_km", None),
        lr_extension_factor=cfg.data.get("lr_extension_factor", 3.0),
        hr_feature_run_name=cfg.data.get("hr_feature_run_name", None),
        lr_feature_run_name=cfg.data.get("lr_feature_run_name", None),
        feature_run_idx=run_idx if cfg.data.get("source", None) == "features" else None,
        **spatial_kwargs,
    )
    dataset_eval = make_multiscale_dataset(
        fmow_dir=cfg.data.fmow_dir,
        landsat_dir=cfg.data.landsat_dir,
        source=cfg.data.get("source", "preprocessed" if "preprocessed_dir" in cfg.data else "raw"),
        preprocessed_dir=cfg.data.preprocessed_dir,
        augment=False,
        image_norm=cfg.data.image_norm,
        lr_crop_km=cfg.data.get("lr_crop_km", None),
        lr_extension_factor=cfg.data.get("lr_extension_factor", 3.0),
        hr_feature_run_name=cfg.data.get("hr_feature_run_name", None),
        lr_feature_run_name=cfg.data.get("lr_feature_run_name", None),
        feature_run_idx=run_idx if cfg.data.get("source", None) == "features" else None,
        **spatial_kwargs,
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
    elif model_target.endswith("SingleBranchLRModel"):
        lr_encoder = instantiate(cfg.model.lr_encoder)
        model = instantiate(
            cfg.model.model,
            encoder=lr_encoder,
            num_task_labels=cfg.num_task_labels,
            num_domain_labels=cfg.num_domain_labels,
            lr_domain_loss_coeff=cfg.model.lr_domain_loss_coeff,
            landsat_channels=cfg.model.landsat_in_channels,
        )
    elif model_target.endswith("SingleBranchStackedModel"):
        encoder = instantiate(cfg.model.encoder)
        model = instantiate(
            cfg.model.model,
            encoder=encoder,
            num_task_labels=cfg.num_task_labels,
            num_domain_labels=cfg.num_domain_labels,
            lr_domain_loss_coeff=cfg.model.lr_domain_loss_coeff,
            landsat_channels=cfg.model.landsat_in_channels,
        )
    elif model_target.endswith("DecisionFusionModel"):
        # Parameter-free decision fusion over cached features. The frozen heads are
        # built later (per seed) by model.load_heads() from the single-branch
        # checkpoints; here we only build the (untrained) decision rule.
        rule = instantiate(cfg.model.rule)
        model = instantiate(
            cfg.model.model,
            num_task_labels=cfg.num_task_labels,
            rule=rule,
        )
    elif model_target.endswith("SingleBranchLocationModel"):
        encoder = instantiate(cfg.model.encoder)
        model = instantiate(
            cfg.model.model,
            encoder=encoder,
            num_task_labels=cfg.num_task_labels,
            num_domain_labels=cfg.num_domain_labels,
            lr_domain_loss_coeff=cfg.model.lr_domain_loss_coeff,
        )
    else:
        branches_target = cfg.model.branches.get("_target_", "")
        is_dual_branch = branches_target.endswith(".DualBranch")

        if is_dual_branch:
            sc = _parse_spatial_cfg(cfg)
            hr_encoder = instantiate(cfg.model.hr_encoder, in_channels=3 + sc["hr_extra"])
            lr_encoder = instantiate(cfg.model.lr_encoder, in_channels=cfg.model.landsat_in_channels + sc["lr_extra"])

            hr_spatial_enc = None
            lr_spatial_enc = None
            if sc["use_fourier"]:
                hr_spatial_enc = SpatialEncoding(sc["fourier_bands"], sc["fourier_proj_dim"])
                lr_spatial_enc = SpatialEncoding(sc["fourier_bands"], sc["fourier_proj_dim"])

            branches = instantiate(
                cfg.model.branches,
                hr_encoder=hr_encoder,
                lr_encoder=lr_encoder,
                coord_channels=sc["coord_channels"],
                hr_spatial_encoding=hr_spatial_enc,
                lr_spatial_encoding=lr_spatial_enc,
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
                lr_domain_loss_coeff=cfg.model.lr_domain_loss_coeff,
                consistency_loss_coeff=cfg.model.consistency_loss_coeff,
                learnable_relation_coeff=cfg.model.learnable_relation_coeff,
                pred_domain_for_d3g=cfg.model.pred_domain_for_d3g,
                detach_lr_for_consistency=cfg.model.detach_lr_for_consistency,
                detach_hr_for_consistency=cfg.model.detach_hr_for_consistency,
            )
        else:
            fusion = instantiate(cfg.model.fusion)
            model = instantiate(
                cfg.model.model,
                branches=branches,
                fusion=fusion,
                num_task_labels=cfg.num_task_labels,
                num_domain_labels=cfg.num_domain_labels,
                lr_domain_loss_coeff=cfg.model.lr_domain_loss_coeff,
                detach_lr_for_task=cfg.model.detach_lr_for_task,
            )

    optimizer_factory = instantiate(cfg.optim.optimizer)
    scheduler_factory = instantiate(cfg.optim.scheduler) if cfg.optim.scheduler is not None else None
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
        label_smoothing=cfg.trainer.label_smoothing,
        branch_ablation=cfg.trainer.branch_ablation,
        alternating_freeze=cfg.trainer.alternating_freeze,
        alternating_freeze_period=cfg.trainer.alternating_freeze_period,
    )


def _is_decision_fusion(cfg: DictConfig) -> bool:
    return cfg.model.model.get("_target_", "").endswith("DecisionFusionModel")


def _compute_class_prior(metadata_path: Path, split: str, num_classes: int) -> torch.Tensor:
    """Per-class frequency vector P(y) over ``split``, indexed by the ``y`` label.

    Counts come from the FMoW metadata CSV (not model metrics); classes absent from
    the split get a count of 0 (DecisionRule clamps the log before use).
    """
    if not metadata_path.exists():
        raise FileNotFoundError(f"Class-prior metadata CSV not found: {metadata_path}")
    df = pd.read_csv(metadata_path, usecols=["split", "y"])
    sub = df[df["split"] == split]
    if sub.empty:
        raise ValueError(f"No rows with split == {split!r} in {metadata_path}")
    prior = torch.zeros(num_classes)
    for cls, n in sub["y"].value_counts().items():
        prior[int(cls)] = float(n)
    return prior


def _prepare_decision_fusion(model: LateFusionModule, cfg: DictConfig, run_idx: int) -> None:
    """Build this seed's frozen heads and set the class prior (no training involved).

    The HR/LR heads come from the same single-branch runs that produced the cached
    features, so the run keys are taken from ``data.hr_feature_run_name`` /
    ``data.lr_feature_run_name`` (these must be the experiment key, e.g.
    ``densenet_baseline``, so ``find_run_dir`` resolves ``train_<key>``). The head for
    seed ``run_idx`` is loaded and its feature width inferred from the saved weight.
    The class prior P(y) is then set on the decision rule (used by every rule).
    """
    hr_run = cfg.data.hr_feature_run_name
    lr_run = cfg.data.lr_feature_run_name
    hr_dir = find_run_dir(hr_run)
    lr_dir = find_run_dir(lr_run)
    if hr_dir is None or lr_dir is None:
        raise FileNotFoundError(
            f"Could not resolve head runs: hr_feature_run_name={hr_run} -> {hr_dir}, "
            f"lr_feature_run_name={lr_run} -> {lr_dir}"
        )
    hr_ckpts = find_best_checkpoints(hr_dir)
    lr_ckpts = find_best_checkpoints(lr_dir)
    if run_idx >= len(hr_ckpts) or run_idx >= len(lr_ckpts):
        raise IndexError(
            f"Seed {run_idx} out of range (HR has {len(hr_ckpts)} seeds, LR has {len(lr_ckpts)})."
        )

    def state(path: Path) -> dict:
        return torch.load(path, map_location="cpu", weights_only=False)["state_dict"]

    model.model.load_heads(state(hr_ckpts[run_idx]), state(lr_ckpts[run_idx]))

    if not cfg.model.get("use_class_prior", True):
        return  # prior-ignored ablation: leave the rule's uniform prior untouched

    split = cfg.model.get("class_prior_split", "train")
    metadata = cfg.model.get("class_prior_metadata", None)
    metadata_path = Path(metadata) if metadata else DEFAULT_PRIOR_METADATA
    prior = _compute_class_prior(metadata_path, split, cfg.num_task_labels)
    model.model.rule.set_class_prior(prior)


def _run_once(
    cfg: DictConfig,
    run_idx: int,
    default_root_dir: str,
    wandb_group: str,
) -> tuple[list[dict], str]:
    """Run one training + test pass. Returns (test_results, checkpoint_dir).

    Decision-fusion models are parameter-free: there is nothing to fit, so for those
    we load this seed's frozen heads + prior and run test only.
    """
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

    train_loader, val_loaders, test_loaders = make_data_loaders(cfg, run_idx)
    model = make_model(cfg)

    if _is_decision_fusion(cfg):
        _prepare_decision_fusion(model, cfg, run_idx)
        trainer = Trainer(
            accelerator=cfg.trainer.accelerator,
            logger=[wandb_logger, csv_logger],
            log_every_n_steps=cfg.trainer.log_every_n_steps,
            default_root_dir=default_root_dir,
        )
        test_results = trainer.test(model=model, dataloaders=test_loaders)
        wandb.finish()
        return test_results, checkpoint_dir

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


if __name__ == "__main__":
    run_experiment()
