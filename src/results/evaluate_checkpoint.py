from __future__ import annotations

import argparse
from pathlib import Path

import torch
from lightning import Trainer, seed_everything
from omegaconf import OmegaConf

from models.late_fusion import LateFusionModule
from train.run_experiment import make_data_loaders, make_model


SEED = 111
BATCH_SIZE = 32
FRAC = 1.0
DATA_LOADER_NUM_WORKERS = 4


def resolve_checkpoint_path(path: Path) -> Path:
    """Resolve a single checkpoint, preferring the best (``late-fusion-*.ckpt``)
    over ``last.ckpt`` — same priority as eval_reproduce.find_best_checkpoints.
    run_experiment.py saves best checkpoints as ``late-fusion-epoch*.ckpt``.
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
    cfg = OmegaConf.load(config_path)

    overrides_path = config_path.parent / "overrides.yaml"
    overrides = OmegaConf.load(overrides_path) if overrides_path.is_file() else None

    return cfg, overrides


def load_model_from_checkpoint(checkpoint_path: Path, cfg) -> LateFusionModule:
    module = make_model(cfg)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    module.load_state_dict(checkpoint["state_dict"])
    return module


def parse_args() -> argparse.Namespace:
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

    _, _, test_loaders = make_data_loaders(cfg)

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
