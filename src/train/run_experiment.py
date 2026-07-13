"""Hydra entrypoint for training and evaluating multi-scale FMoW classification models.

Builds datasets/dataloaders and a model from a Hydra config (composed from
`configs/setup.yaml` and its defaults), trains it with a Lightning
`Trainer` (or evaluates only, for parameter-free frozen decision-fusion
models), logs to Weights & Biases and a CSV logger, and repeats the process
for two fixed reruns (`run_idx` 1 and 2), reporting each run's key test
metric.

Functions:
    `has_device_tensor_cores`: Check whether the current CUDA device
        supports Tensor Cores.
    `_make_loader` / `make_data_loaders`: Build the train/val/test
        `DataLoader`s for a run from the Hydra config.
    `_parse_spatial_cfg`: Resolve the spatial-encoding config block (coord
        channels, Fourier positional encoding, overlap mask) into the
        derived flags/channel counts used to build encoders and branches.
    `make_model`: Instantiate the model architecture (single-branch,
        early/feature/decision fusion, or D3G) and wrap it in
        `MultiScaleClassificationModule` for training/eval.
    `_run_once`: Run one seeded training + test pass (or test-only, for
        frozen decision-fusion models).
    `_best_run_score`: Extract a named metric from a Lightning
        `Trainer.test` results list.
    `run_experiment`: Hydra-decorated `main`; orchestrates the reruns and
        reports the best one.

Usage:
    uv run src/train/run_experiment.py model=<model> [key=value ...]
"""

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
from models.components.spatial_encoding import SpatialEncoding
from models.multi_scale_classification import MultiScaleClassificationModule
from models.utils import domain_label_names
from train.utils import make_multiscale_dataset, make_multiscale_loader


def has_device_tensor_cores() -> bool:
    """Check whether the current CUDA device supports Tensor Cores.

    See https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html
    for the compute-capability reference (Tensor Cores from major version
    7, i.e. Volta and later).

    Returns:
        bool: True if CUDA is available and the current device's compute
        capability major version is >= 7; False if CUDA is unavailable.
    """
    if not torch.cuda.is_available():
        return False
    device = torch.device("cuda")
    major, _ = torch.cuda.get_device_capability(device)
    return major >= 7


def _make_loader(dataset: FMoWMultiScaleDataset, split: str, cfg: DictConfig, shuffle: bool) -> DataLoader:
    """Build a `DataLoader` for one split, pulling batch/worker settings from `cfg`.

    Args:
        dataset (FMoWMultiScaleDataset): Dataset to draw the subset from.
        split (str): WILDS split name to load (e.g. `"train"`, `"id_val"`).
        cfg (DictConfig): Hydra config; reads `cfg.data.frac`,
            `cfg.data.batch_size`, `cfg.data.num_workers`.
        shuffle (bool): Whether to shuffle samples each epoch.

    Returns:
        DataLoader: Loader over `dataset`'s `split` subset.
    """
    return make_multiscale_loader(
        dataset,
        split=split,
        frac=cfg.data.frac,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        shuffle=shuffle,
    )


def _parse_spatial_cfg(cfg: DictConfig):
    """Resolve the `cfg.model.spatial_encoding` block into derived spatial-encoding flags.

    Reads the raw spatial-encoding config (raw coordinate channels, Fourier
    positional encoding, and overlap mask, each independently toggleable
    per branch) and derives: whether each branch needs raw coord channels /
    Fourier PE, how many extra input channels each branch's encoder needs
    (`hr_extra`/`lr_extra`), and whether the dataset needs to emit the
    coordinate grid and/or overlap mask tensors at all (used to set
    `FMoWMultiScaleDataset`'s `spatial_coord_grid`/`spatial_overlap_mask`).

    Args:
        cfg (DictConfig): Hydra config; reads keys under
            `cfg.model.spatial_encoding` (all optional): `coord_channels`
            (bool, default False), `coord_on_hr`/`coord_on_lr` (bool,
            default True each), `overlap_mask` (bool, default False),
            `overlap_mask_type` (str, default "binary"), `fourier_bands`
            (int, default 0), `fourier_proj_dim` (int, default 0),
            `fourier_on_hr`/`fourier_on_lr` (bool, default True each).

    Returns:
        dict: Derived spatial config with keys:
            "use_coord_hr"/"use_coord_lr" (bool): Whether to add raw
                coordinate channels to the HR/LR encoder input.
            "overlap_mask" (bool): Whether an overlap-mask channel is added
                to the LR encoder input.
            "overlap_mask_type" (str): `"binary"` or `"gaussian"`.
            "fourier_bands" (int), "fourier_proj_dim" (int): Fourier PE
                hyperparameters.
            "use_fourier" (bool): Whether Fourier PE is enabled at all.
            "use_fourier_hr"/"use_fourier_lr" (bool): Whether Fourier PE is
                applied on the HR/LR branch.
            "hr_extra"/"lr_extra" (int): Extra input channels the HR/LR
                encoder needs on top of its base image channels (2 for raw
                coords, 1 for the overlap mask on LR,
                `fourier_proj_dim` for Fourier PE).
            "needs_coord_grid" (bool): Whether the dataset must emit
                coordinate grid tensors (any coord or Fourier use).
            "needs_overlap_mask" (bool): Whether the dataset must emit the
                overlap-mask tensor (alias of "overlap_mask").
    """
    spatial_cfg = cfg.model.get("spatial_encoding", {})
    coord_channels = spatial_cfg.get("coord_channels", False)
    # Per-branch raw-coord toggles (default both on -> existing symmetric behaviour).
    # Like the Fourier toggles: the LR coord grid is constant, so coord_on_lr=false
    # drops the (useless) raw LR coords while keeping HR scene-scale coords.
    coord_on_hr = spatial_cfg.get("coord_on_hr", True)
    coord_on_lr = spatial_cfg.get("coord_on_lr", True)
    overlap_mask = spatial_cfg.get("overlap_mask", False)
    fourier_bands = spatial_cfg.get("fourier_bands", 0)
    fourier_proj_dim = spatial_cfg.get("fourier_proj_dim", 0)
    overlap_mask_type = spatial_cfg.get("overlap_mask_type", "binary")
    # Per-branch Fourier toggles (default both on -> existing symmetric behaviour).
    # The LR coord grid is constant, so LR Fourier PE is a no-op; fourier_on_lr=false
    # skips it (and its wasted input channels) while keeping HR scene-scale PE.
    fourier_on_hr = spatial_cfg.get("fourier_on_hr", True)
    fourier_on_lr = spatial_cfg.get("fourier_on_lr", True)

    use_coord_hr = coord_channels and coord_on_hr
    use_coord_lr = coord_channels and coord_on_lr

    use_fourier = fourier_proj_dim > 0 and fourier_bands > 0
    use_fourier_hr = use_fourier and fourier_on_hr
    use_fourier_lr = use_fourier and fourier_on_lr

    hr_extra, lr_extra = 0, 0
    if use_coord_hr:
        hr_extra += 2
    if use_coord_lr:
        lr_extra += 2
    if overlap_mask:
        lr_extra += 1
    if use_fourier_hr:
        hr_extra += fourier_proj_dim
    if use_fourier_lr:
        lr_extra += fourier_proj_dim

    needs_coord_grid = use_coord_hr or use_coord_lr or use_fourier_hr or use_fourier_lr
    needs_overlap_mask = overlap_mask

    return {
        "use_coord_hr": use_coord_hr,
        "use_coord_lr": use_coord_lr,
        "overlap_mask": overlap_mask,
        "overlap_mask_type": overlap_mask_type,
        "fourier_bands": fourier_bands,
        "fourier_proj_dim": fourier_proj_dim,
        "use_fourier": use_fourier,
        "use_fourier_hr": use_fourier_hr,
        "use_fourier_lr": use_fourier_lr,
        "hr_extra": hr_extra,
        "lr_extra": lr_extra,
        "needs_coord_grid": needs_coord_grid,
        "needs_overlap_mask": needs_overlap_mask,
    }


def make_data_loaders(cfg: DictConfig, run_idx: int) -> Tuple[DataLoader, List[DataLoader], List[DataLoader]]:
    """Build the train, validation, and test `DataLoader`s for one run.

    Constructs two `FMoWMultiScaleDataset`s from `cfg.data` (one with
    augmentation for training, one without for eval), sized/spatially
    configured via `_parse_spatial_cfg`, then wraps the requested splits in
    `DataLoader`s.

    Args:
        cfg (DictConfig): Hydra config; reads `cfg.data.*` (dataset paths,
            `source`, `augment_train`, `image_norm`, `lr_crop_km`,
            `lr_extension_factor`, `hr_feature_run_name`,
            `lr_feature_run_name`, `leave_asia_out`, `train_split`,
            `val_splits`, `test_splits`) and `cfg.model.spatial_encoding`
            (via `_parse_spatial_cfg`).
        run_idx (int): Rerun index; used as `feature_run_idx` when
            `cfg.data.source == "features"` to select that rerun's cached
            features.

    Returns:
        tuple[DataLoader, list[DataLoader], list[DataLoader]]:
        `(train_loader, val_loaders, test_loaders)`, where
        `val_loaders`/`test_loaders` have one loader per split in
        `cfg.data.val_splits`/`cfg.data.test_splits`.
    """
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
        leave_asia_out=cfg.data.get("leave_asia_out", False),
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
        leave_asia_out=cfg.data.get("leave_asia_out", False),
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


def make_model(cfg: DictConfig, run_idx: int = 0) -> MultiScaleClassificationModule:
    """Instantiate the configured model architecture and wrap it for training/eval.

    Dispatches on `cfg.model.model._target_` (and, for the generic fusion
    path, `cfg.model.branches._target_`) to build one of: a single-branch
    model (HR-only, LR-only, or location-only), early fusion, feature/D3G
    fusion (dual-branch encoders + a fusion module), or decision fusion
    (combining frozen or jointly-trained single-branch heads via a decision
    rule). Encoders are instantiated with extra input channels/spatial
    encoding modules as resolved by `_parse_spatial_cfg` when the
    architecture uses spatial encodings. The resulting model, its
    optimizer/scheduler factories (main + a separate domain-classifier
    optimizer/scheduler), and the task/domain label counts are then wrapped
    in a `MultiScaleClassificationModule` Lightning module.

    Args:
        cfg (DictConfig): Hydra config; reads `cfg.model.*` (model/branch/
            fusion/encoder targets and their kwargs), `cfg.data.leave_asia_out`,
            `cfg.num_task_labels`, `cfg.domain_index`, `cfg.trainer.*`
            (`ece_n_bins`, `monitor_metric`, `label_smoothing`,
            `branch_ablation`, `alternating_freeze`,
            `alternating_freeze_period`), `cfg.optim.*`
            (optimizer/scheduler and domain-optimizer/scheduler factories),
            and `cfg.data.val_loader_names`/`cfg.data.test_loader_names`.
        run_idx (int): Rerun index, forwarded to a decision-fusion model to
            pick which rerun's cached single-branch checkpoints/features to
            load. Defaults to 0.

    Returns:
        MultiScaleClassificationModule: The instantiated model wrapped in
        the shared training/eval Lightning module.
    """

    model_target = cfg.model.model.get("_target_", "")

    leave_asia_out = cfg.data.get("leave_asia_out", False)
    num_domain_labels = (
        len(domain_label_names(leave_asia_out))
    )

    if model_target.endswith("SingleBranchModel"):
        sc = _parse_spatial_cfg(cfg)
        hr_encoder = instantiate(cfg.model.hr_encoder, in_channels=3 + sc["hr_extra"])
        hr_spatial_enc = (
            SpatialEncoding(sc["fourier_bands"], sc["fourier_proj_dim"])
            if sc["use_fourier_hr"]
            else None
        )
        model = instantiate(
            cfg.model.model,
            encoder=hr_encoder,
            num_task_labels=cfg.num_task_labels,
            num_domain_labels=num_domain_labels,
            coord_channels_hr=sc["use_coord_hr"],
            hr_spatial_encoding=hr_spatial_enc,
        )
    elif model_target.endswith("SingleBranchLRModel"):
        lr_encoder = instantiate(cfg.model.lr_encoder)
        model = instantiate(
            cfg.model.model,
            encoder=lr_encoder,
            num_task_labels=cfg.num_task_labels,
            num_domain_labels=num_domain_labels,
            lr_domain_loss_coeff=cfg.model.lr_domain_loss_coeff,
            landsat_channels=cfg.model.landsat_in_channels,
        )
    elif model_target.endswith("EarlyFusionModel"):
        encoder = instantiate(cfg.model.encoder)
        model = instantiate(
            cfg.model.model,
            encoder=encoder,
            num_task_labels=cfg.num_task_labels,
            num_domain_labels=num_domain_labels,
            lr_domain_loss_coeff=cfg.model.lr_domain_loss_coeff,
            landsat_channels=cfg.model.landsat_in_channels,
        )
    elif model_target.endswith("DecisionFusionModel"):
        # Decision fusion over cached features. The decision rule computes the class
        # prior from metadata in its own constructor; the model then builds this seed's
        # HR/LR heads from the single-branch checkpoints in its constructor. With
        # train_model=True the heads are trained jointly from scratch (Trainer.fit);
        # otherwise they copy the frozen single-branch weights.
        rule = instantiate(
            cfg.model.rule,
            class_prior_metadata=cfg.data.get("metadata_path", None),
        )
        model = instantiate(
            cfg.model.model,
            num_task_labels=cfg.num_task_labels,
            rule=rule,
            hr_run_name=cfg.data.hr_feature_run_name,
            lr_run_name=cfg.data.lr_feature_run_name,
            run_idx=run_idx,
            train_model=cfg.model.get("train_model", False),
        )
    elif model_target.endswith("SingleBranchLocationModel"):
        encoder = instantiate(cfg.model.encoder)
        model = instantiate(
            cfg.model.model,
            encoder=encoder,
            num_task_labels=cfg.num_task_labels,
            num_domain_labels=num_domain_labels,
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
            if sc["use_fourier_hr"]:
                hr_spatial_enc = SpatialEncoding(sc["fourier_bands"], sc["fourier_proj_dim"])
            if sc["use_fourier_lr"]:
                lr_spatial_enc = SpatialEncoding(sc["fourier_bands"], sc["fourier_proj_dim"])

            branches = instantiate(
                cfg.model.branches,
                hr_encoder=hr_encoder,
                lr_encoder=lr_encoder,
                coord_channels_hr=sc["use_coord_hr"],
                coord_channels_lr=sc["use_coord_lr"],
                hr_spatial_encoding=hr_spatial_enc,
                lr_spatial_encoding=lr_spatial_enc,
            )
        else:
            hr_encoder = instantiate(cfg.model.hr_encoder)
            lr_target = cfg.model.lr_encoder.get("_target_", "")
            if lr_target.endswith("DomainEmbeddingBranch"):
                lr_encoder = instantiate(cfg.model.lr_encoder, num_domains=num_domain_labels)
            else:
                lr_encoder = instantiate(cfg.model.lr_encoder)
            branches = instantiate(cfg.model.branches, hr_encoder=hr_encoder, lr_encoder=lr_encoder)
        if model_target.endswith("D3GModel"):
            model = instantiate(
                cfg.model.model,
                branches=branches,
                num_task_labels=cfg.num_task_labels,
                num_domain_labels=num_domain_labels,
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
                num_domain_labels=num_domain_labels,
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

    return MultiScaleClassificationModule(
        model=model,
        optimizer=optimizer_factory,
        scheduler=scheduler_factory,
        domain_optimizer=domain_optimizer_factory,
        domain_scheduler=domain_scheduler_factory,
        num_task_labels=cfg.num_task_labels,
        num_domain_labels=num_domain_labels,
        domain_index=cfg.domain_index,
        ece_n_bins=cfg.trainer.ece_n_bins,
        val_loader_names=list(cfg.data.val_loader_names),
        test_loader_names=list(cfg.data.test_loader_names),
        key_metric=cfg.trainer.monitor_metric,
        label_smoothing=cfg.trainer.label_smoothing,
        branch_ablation=cfg.trainer.branch_ablation,
        alternating_freeze=cfg.trainer.alternating_freeze,
        alternating_freeze_period=cfg.trainer.alternating_freeze_period,
        leave_asia_out=leave_asia_out,
    )


def _run_once(
    cfg: DictConfig,
    run_idx: int,
    default_root_dir: str,
    wandb_group: str,
) -> tuple[list[dict], str]:
    """Run one seeded training + test pass, or test-only for frozen models.

    Seeds everything with `cfg.seed + run_idx`, builds the data loaders and
    model for this run, and either (a) if the model is parameter-free
    (``model.model.train_model()`` is False, e.g. a frozen decision-fusion
    model using this seed's precomputed heads/prior), runs `Trainer.test`
    directly, or (b) otherwise fits with `Trainer.fit` (checkpointing on
    `cfg.trainer.monitor_metric`) and tests the best checkpoint. Logs to
    both W&B (grouped under `wandb_group`) and a CSV logger, and calls
    `wandb.finish()` before returning.

    Args:
        cfg (DictConfig): Hydra config for this run (trainer/wandb/data/
            model settings).
        run_idx (int): Rerun index; combined with `cfg.seed` to seed this
            run, and used to select this run's checkpoint/feature-cache
            subdirectory.
        default_root_dir (str): Root output directory for the Lightning
            `Trainer` and CSV logger (the Hydra run's output dir).
        wandb_group (str): W&B run group name shared by all reruns of this
            experiment; this run's W&B run name is
            ``f"{wandb_group}-run{run_idx}"``.

    Returns:
        tuple[list[dict], str]: `(test_results, checkpoint_dir)`, where
        `test_results` is the list of per-dataloader metric dicts returned
        by `Trainer.test`, and `checkpoint_dir` is
        ``f"{default_root_dir}/checkpoints/run{run_idx}"``.
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
    model = make_model(cfg, run_idx)

    if not model.model.train_model():
        # Frozen decision fusion: nothing to fit, run test only.
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
    """Extract a named metric from a `Trainer.test` results list.

    Searches the per-dataloader result dicts in order and returns the value
    from the first dict that contains `metric`.

    Args:
        test_results (list[dict]): Per-test-dataloader metric dicts, as
            returned by `Trainer.test`.
        metric (str): Metric key to look up (e.g.
            `"test/test-od-worst-group-task-acc"`).

    Returns:
        float: The metric's value from the first matching result dict.

    Raises:
        ValueError: If `metric` is not present in any of the result dicts.
    """
    for result_dict in test_results:
        if metric in result_dict:
            return result_dict[metric]
    raise ValueError(f"Metric '{metric}' not found in test results: {test_results}")


@hydra.main(version_base=None, config_path="configs", config_name="setup")
def run_experiment(cfg: DictConfig) -> None:
    """Hydra entrypoint: run two seeded reruns of the configured experiment and report results.

    Logs into W&B, derives a shared `wandb_group` name (`cfg.run_name` +
    timestamp) for all reruns, then calls `_run_once` for `run_idx` in
    `[1, 2]` (always exactly two reruns, regardless of `cfg.num_reruns`),
    scoring each with `cfg.trainer.best_run_metric` (default
    `"test/test-od-worst-group-task-acc"`) via `_best_run_score` and
    printing each run's score. If `cfg.num_reruns > 1`, additionally prints
    the best-scoring run's checkpoint directory and score.

    Args:
        cfg (DictConfig): Full Hydra config, composed from
            `configs/setup.yaml` and its defaults (`data`, `spatial`,
            `optim`, `trainer`, `wandb`, `model`) plus any CLI overrides.
            Reads `cfg.run_name`, `cfg.seed`, `cfg.num_reruns`,
            `cfg.trainer.best_run_metric`, and (via `_run_once`) the rest
            of the config.

    Returns:
        None. Results are logged to W&B/CSV and printed; nothing is
        returned to the Hydra runner.
    """
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
