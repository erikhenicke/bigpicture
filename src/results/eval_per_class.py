#!/usr/bin/env python3
"""Evaluate a trained model per-class on both OOD and ID test splits.

Uses Lightning's trainer.test() to match the original training evaluation path.
Step 1: Reproduce the original metrics.csv values via trainer.test().
Step 2 (TODO): Add per-class collection once metrics match.
"""

import argparse
import sys
from pathlib import Path

import torch
import yaml
from lightning import Trainer, seed_everything
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).parent.parent.parent
LOG_RUNS = REPO_ROOT / "log" / "runs"
RUN_CONFIG_DIR = REPO_ROOT / "src" / "train" / "configs" / "run"


def find_run_dir(exp_key: str) -> Path | None:
    """Return the most recent log directory for the given experiment key."""
    job_name = f"train_{exp_key}"
    prefix = job_name + "-"

    candidates: list[tuple[str, str, Path]] = []
    for date_dir in LOG_RUNS.iterdir():
        if not date_dir.is_dir():
            continue
        for run_dir in date_dir.iterdir():
            if run_dir.name.startswith(prefix):
                candidates.append((date_dir.name, run_dir.name, run_dir))

    if not candidates:
        return None
    candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
    print(f"Found {candidates} for experiment '{exp_key}':")
    return candidates[0][2]


def find_best_checkpoints(run_dir: Path) -> list[Path]:
    """Find best checkpoint for each seed run."""
    checkpoints = []
    ckpt_root = run_dir / "checkpoints"
    for seed_dir in sorted(ckpt_root.glob("run*")):
        best = list(seed_dir.glob("late-fusion-*.ckpt"))
        if best:
            checkpoints.append(best[0])
        else:
            last = seed_dir / "last.ckpt"
            if last.exists():
                checkpoints.append(last)
    return checkpoints


def load_hydra_config(run_dir: Path):
    """Load the hydra config from a run directory."""
    config_path = run_dir / ".hydra" / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"No .hydra/config.yaml found in {run_dir}")
    cfg = OmegaConf.load(config_path)
    # Backfill trainer fields that make_model reads but older runs didn't persist.
    trainer_defaults = {
        "alternating_freeze": False,
        "alternating_freeze_period": 1,
        "branch_ablation": False,
    }
    for key, default in trainer_defaults.items():
        if key not in cfg.trainer:
            cfg.trainer[key] = default
    return cfg


def _strip_compile_prefix(state_dict: dict) -> dict:
    """Drop torch.compile's '_orig_mod.' prefix so weights load into an uncompiled model.

    Runs trained with trainer.compile=true wrap self.model in torch.compile during fit,
    which prefixes the saved keys. We test uncompiled, so normalize the keys.
    """
    return {k.replace("._orig_mod.", "."): v for k, v in state_dict.items()}


def evaluate_checkpoint(ckpt_path: Path, cfg) -> list[dict]:
    """Run trainer.test() on a checkpoint, reproducing run_experiment.py's test metrics.

    Mirrors `_run_once` in run_experiment.py: same cfg -> same model -> best weights ->
    make_data_loaders(cfg) -> trainer.test(). Metrics are computed in
    LateFusionModule.on_test_epoch_end, independent of the logger.
    """
    from train.run_experiment import make_data_loaders, make_model

    module = make_model(cfg)

    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = _strip_compile_prefix(checkpoint["state_dict"])
    module.load_state_dict(state_dict)

    # The training run logs domain confusion matrices to W&B in on_test_epoch_end.
    # We test with logger=False, so no-op it to avoid a None-logger crash for domain models.
    module._log_domain_confusion_matrix = lambda *args, **kwargs: None

    _, _, test_loaders = make_data_loaders(cfg)

    trainer = Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
    )

    return trainer.test(model=module, dataloaders=test_loaders)


def load_run_config(config_path: Path) -> tuple[dict, str]:
    """Load a run config YAML from the given path."""
    if not config_path.exists():
        print(f"Run config not found: {config_path}", file=sys.stderr)
        sys.exit(1)
    with config_path.open() as f:
        cfg = yaml.safe_load(f)
    return cfg, config_path.stem


def main() -> None:
    parser = argparse.ArgumentParser(description="Per-class evaluation of a trained model")
    parser.add_argument("--config", type=str, required=True, help="Path to run config YAML (e.g. src/train/configs/run/multsim.yaml)")
    parser.add_argument("--run-name", type=str, required=True, help="Experiment key to evaluate (e.g. film_om_bin_pe)")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--num-workers", type=int, default=None, help="Override num workers")
    args = parser.parse_args()

    sys.path.insert(0, str(REPO_ROOT / "src"))

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = Path.cwd() / config_path
    run_config, config_name = load_run_config(config_path)

    exp_key = args.run_name
    if exp_key not in run_config.get("experiments", {}):
        print(f"Experiment '{exp_key}' not found in {config_path.name}", file=sys.stderr)
        print(f"Available: {', '.join(run_config.get('experiments', {}).keys())}", file=sys.stderr)
        sys.exit(1)

    run_dir = find_run_dir(exp_key)
    print(f"\nEvaluating: {exp_key}")
    print(f"Run directory: {run_dir}")

    cfg = load_hydra_config(run_dir)

    if args.batch_size is not None:
        cfg.data.batch_size = args.batch_size
    if args.num_workers is not None:
        cfg.data.num_workers = args.num_workers

    checkpoints = find_best_checkpoints(run_dir)
    if not checkpoints:
        print("No checkpoints found!", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(checkpoints)} seed checkpoints:")
    for cp in checkpoints:
        print(f"  {cp.name}")

    seed_everything(111, workers=True)
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("medium")

    for i, ckpt_path in enumerate(checkpoints):
        print(f"\n--- Evaluating seed {i} ({ckpt_path.name}) ---")
        results = evaluate_checkpoint(ckpt_path, cfg)
        for result_dict in results:
            for key, value in sorted(result_dict.items()):
                print(f"  {key}: {value:.4f}")


if __name__ == "__main__":
    main()
