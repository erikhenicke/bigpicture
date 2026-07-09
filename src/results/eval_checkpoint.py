"""Re-evaluate a single saved Lightning checkpoint on FMoW test data.

Given a checkpoint file or a run directory (the best checkpoint is picked via
`resolve_checkpoint_path`), rebuilds the model from the run's Hydra config
(auto-discovered via `find_hydra_config` and loaded via `load_hydra_config`,
or overridden with ``--config``) and loads its weights
(`load_model_from_checkpoint`). With ``--inspect`` the model structure is
printed and evaluation is skipped; otherwise the FMoW test dataloaders are
built via `train.run_experiment.make_data_loaders` and evaluated with
`lightning.Trainer.test`. `parse_args` defines the CLI, and
`evaluate_lightning_checkpoint` is the script's entry point (invoked from
``__main__``).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from lightning import Trainer, seed_everything
from omegaconf import OmegaConf

from models.multi_scale_classification import MultiScaleClassificationModule
from train.run_experiment import make_data_loaders, make_model


SEED = 111
BATCH_SIZE = 32
FRAC = 1.0
DATA_LOADER_NUM_WORKERS = 4


def resolve_checkpoint_path(path: Path) -> Path:
    """Resolve a single checkpoint, preferring the best (``late-fusion-*.ckpt``)
    over ``last.ckpt`` — same priority as eval_reproduce.find_best_checkpoints.
    run_experiment.py saves best checkpoints as ``late-fusion-epoch*.ckpt``.

    Args:
        path (Path): Either a direct path to a ``.ckpt`` file, or a run
            directory to search recursively for checkpoints.

    Returns:
        Path: `path` itself if it is a file; otherwise the most recently
            modified ``late-fusion-*.ckpt`` under `path`, or the most
            recently modified ``last.ckpt`` if no ``late-fusion-*.ckpt``
            exists.

    Raises:
        FileNotFoundError: If `path` is neither an existing file nor an
            existing directory, or if no matching checkpoint files are found
            under it.
    """
    if path.is_file():
        return path

    if not path.is_dir():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    candidates = list(path.rglob("late-fusion-*.ckpt"))
    if not candidates:
        candidates = list(path.rglob("last.ckpt"))
    if not candidates:
        raise FileNotFoundError(f"No checkpoints found under: {path}")

    return max(candidates, key=lambda candidate: candidate.stat().st_mtime)


def find_hydra_config(checkpoint_path: Path) -> Path:
    """Locate the Hydra config for a checkpoint by walking up parent directories.

    Starting at `checkpoint_path` itself (if it is a directory) or its
    parent (if it is a file), checks up to 6 directory levels up for a
    ``.hydra/config.yaml``.

    Args:
        checkpoint_path (Path): Checkpoint file or run directory whose Hydra
            config should be found.

    Returns:
        Path: Path to the discovered ``.hydra/config.yaml``.

    Raises:
        FileNotFoundError: If no ``.hydra/config.yaml`` is found within 6
            levels above `checkpoint_path`.
    """
    current = checkpoint_path if checkpoint_path.is_dir() else checkpoint_path.parent
    for _ in range(6):
        hydra_cfg = current / ".hydra" / "config.yaml"
        if hydra_cfg.is_file():
            return hydra_cfg
        current = current.parent
    raise FileNotFoundError(
        f"Could not find .hydra/config.yaml above {checkpoint_path}. "
        "Use --config to provide the path explicitly."
    )


def load_hydra_config(config_path: Path) -> tuple:
    """Load a run's Hydra config and its optional overrides file.

    Args:
        config_path (Path): Path to a ``.hydra/config.yaml`` file.

    Returns:
        tuple: ``(cfg, overrides)`` where `cfg` is the loaded OmegaConf
            config and `overrides` is the loaded ``overrides.yaml`` from the
            same directory as `config_path`, or None if that file doesn't
            exist.
    """
    cfg = OmegaConf.load(config_path)

    overrides_path = config_path.parent / "overrides.yaml"
    overrides = OmegaConf.load(overrides_path) if overrides_path.is_file() else None

    return cfg, overrides


def load_model_from_checkpoint(checkpoint_path: Path, cfg) -> MultiScaleClassificationModule:
    """Build a model from `cfg` and load its weights from a checkpoint.

    Args:
        checkpoint_path (Path): Path to the ``.ckpt`` file to load weights
            from.
        cfg: Hydra/OmegaConf config used to construct the model via
            `train.run_experiment.make_model`.

    Returns:
        MultiScaleClassificationModule: The model with the checkpoint's
            ``state_dict`` loaded.
    """
    module = make_model(cfg)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    module.load_state_dict(checkpoint["state_dict"])
    return module


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for `evaluate_lightning_checkpoint`.

    Returns:
        argparse.Namespace: Parsed arguments -- `checkpoint_path` (str, path
            to a ``.ckpt`` file or run directory), `config` (str | None,
            explicit Hydra config path), `inspect` (bool, print model
            structure and exit), `batch_size` (int | None, override),
            `num_workers` (int | None, override), `frac` (float | None,
            override fraction of each split to evaluate), and `seed` (int,
            default `SEED`).
    """
    parser = argparse.ArgumentParser(description="Evaluate a Lightning checkpoint on FMoW test data.")
    parser.add_argument("checkpoint_path", type=str, help="Path to a .ckpt file or run directory")
    parser.add_argument("--config", type=str, default=None, help="Path to .hydra/config.yaml (auto-discovered if omitted)")
    parser.add_argument("--inspect", action="store_true", help="Print model structure and exit without evaluation")
    parser.add_argument("--batch-size", type=int, default=None, help="Override evaluation batch size")
    parser.add_argument("--num-workers", type=int, default=None, help="Override DataLoader workers")
    parser.add_argument("--frac", type=float, default=None, help="Override fraction of each split to evaluate")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed")
    return parser.parse_args()


def evaluate_lightning_checkpoint() -> None:
    """CLI entry point: load a Lightning checkpoint and evaluate it on FMoW test data.

    Resolves the checkpoint (`resolve_checkpoint_path`) and its Hydra config
    (auto-discovered via `find_hydra_config`, or from ``--config``),
    rebuilds the model and loads its weights (`load_model_from_checkpoint`),
    prints the Hydra overrides and model structure, then either returns (if
    ``--inspect``) or applies any batch-size/worker-count/split-fraction
    overrides, builds the test dataloaders via
    `train.run_experiment.make_data_loaders`, and runs
    `lightning.Trainer.test`, printing the resulting metrics.

    Returns:
        None
    """
    args = parse_args()
    seed_everything(args.seed, workers=True)

    checkpoint_path = resolve_checkpoint_path(Path(args.checkpoint_path))

    if args.config:
        config_path = Path(args.config)
    else:
        config_path = find_hydra_config(Path(args.checkpoint_path))

    cfg, overrides = load_hydra_config(config_path)
    if not hasattr(cfg, "trainer") or not hasattr(cfg.trainer, "alternating_freeze"):
        cfg.trainer.alternating_freeze = False
        cfg.trainer.alternating_freeze_period = 1

    if overrides:
        print("=== Hydra overrides ===")
        for override in overrides:
            print(f"  {override}")
        print()

    print("=== Checkpoint ===")
    print(f"  {checkpoint_path}")
    print()

    module = load_model_from_checkpoint(checkpoint_path, cfg)

    print("=== Model structure ===")
    print(module.model)
    print()

    if args.inspect:
        return

    if args.batch_size is not None:
        cfg.data.batch_size = args.batch_size
    if args.num_workers is not None:
        cfg.data.num_workers = args.num_workers
    if args.frac is not None:
        cfg.data.frac = args.frac

    _, _, test_loaders = make_data_loaders(cfg, 0)

    trainer = Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        log_every_n_steps=25,
    )

    results = trainer.test(model=module, dataloaders=test_loaders, ckpt_path=None)
    print(results)


if __name__ == "__main__":
    evaluate_lightning_checkpoint()
