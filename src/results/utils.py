"""Shared helpers for loading trained runs (checkpoints + hydra config)."""

from __future__ import annotations

from pathlib import Path

from omegaconf import OmegaConf


def find_best_checkpoints(run_dir: Path) -> list[Path]:
    """Find the best checkpoint for each seed run under ``run_dir/checkpoints``.

    Prefers ``late-fusion-*.ckpt`` (run_experiment.py's ModelCheckpoint filename),
    falling back to ``last.ckpt``. Seed dirs with neither are skipped. Returned in
    sorted ``run*`` order.
    """
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
    """Load ``.hydra/config.yaml`` from a run directory, backfilling trainer fields
    that ``make_model`` reads but older runs didn't persist."""
    config_path = Path(run_dir) / ".hydra" / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"No .hydra/config.yaml found in {run_dir}")
    cfg = OmegaConf.load(config_path)
    trainer_defaults = {
        "alternating_freeze": False,
        "alternating_freeze_period": 1,
        "branch_ablation": False,
    }
    for key, default in trainer_defaults.items():
        if key not in cfg.trainer:
            cfg.trainer[key] = default
    return cfg