#!/usr/bin/env python3
"""Re-evaluate a trained model on both OOD and ID test splits.

Uses Lightning's trainer.test() to match the original training evaluation path.
The rerun goes through the same LateFusionModule.on_test_epoch_end as training,
which now also emits per-class top-1 accuracy (overall and per region) and top-5
task accuracy. Two things happen per seed:
  1. The rerun is compared against the original metrics.csv to confirm the shared
     metrics still reproduce.
  2. The full rerun metric set (including the new per-class / top-5 metrics) is
     written to ``metrics_rerun.csv`` next to the original ``metrics.csv``.
"""

import argparse
import csv
import sys
from pathlib import Path

import torch
import yaml
from lightning import Trainer, seed_everything

from train.run_experiment import make_data_loaders, make_model, has_device_tensor_cores
from results.utils import find_best_checkpoints, load_hydra_config 


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


def evaluate_checkpoint(ckpt_path: Path, cfg, run_idx: int) -> list[dict]:
    """Run trainer.test() on a checkpoint, reproducing run_experiment.py's test metrics.

    Mirrors `_run_once` in run_experiment.py: seed -> make_data_loaders(cfg, run_idx) -> model ->
    best weights -> trainer.test(). Metrics are computed in
    LateFusionModule.on_test_epoch_end, independent of the logger.

    Reproducing the seed and the loader-build order matters when cfg.data.frac < 1.0:
    WILDS draws the random frac subset via the global np.random RNG inside get_subset,
    so the exact test images depend on seed_everything(cfg.seed + run_idx) and the order
    of the get_subset calls. The subset is drawn before training, so it is fully
    determined by these two things (not by the trained weights). `run_idx` is the
    original run index (run{run_idx}) for this checkpoint.
    """
    # Recreate the RNG state the original run used right before it built its loaders.
    seed_everything(cfg.seed + run_idx, workers=True)
    _, _, test_loaders = make_data_loaders(cfg, run_idx)

    module = make_model(cfg)

    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint["state_dict"]
    module.load_state_dict(state_dict)

    # Disable Dropout and use BatchNorm running stats so the forward pass is
    # deterministic. on_test_epoch_start also enforces this, but set it here too
    # so the module is in the right mode regardless of how it is driven.
    module.eval()
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False 

    # The training run logs domain confusion matrices to W&B in on_test_epoch_end.
    # We test with logger=False, so no-op it to avoid a None-logger crash for domain models.
    module._log_domain_confusion_matrix = lambda *args, **kwargs: None

    trainer = Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
    )

    return trainer.test(model=module, dataloaders=test_loaders)


def load_original_test_metrics(run_dir: Path, run_idx: int) -> dict[str, float]:
    """Read the original test metrics that run_experiment.py wrote via CSVLogger.

    During training, `trainer.test(...)` logs one final row of `test/...` columns to
    `run{run_idx}/version_0/metrics.csv`. We return that row as a flat metric->value dict.
    """
    metrics_path = run_dir / f"run{run_idx}" / "version_0" / "metrics.csv"
    if not metrics_path.exists():
        return {}

    with metrics_path.open() as f:
        rows = list(csv.DictReader(f))

    test_cols = [c for c in (rows[0].keys() if rows else []) if c.startswith("test/")]
    # The test row is the last one with any non-empty test column.
    for row in reversed(rows):
        if any(row.get(c) not in (None, "") for c in test_cols):
            return {c: float(row[c]) for c in test_cols if row.get(c) not in (None, "")}
    return {}


def write_rerun_metrics(run_dir: Path, run_idx: int, rerun_metrics: dict[str, float]) -> Path:
    """Persist the full rerun test metrics next to the original metrics.csv.

    Written as a long-format ``metric,value`` CSV sorted by metric name, so the
    high-cardinality per-class rows (62 classes + region x class) stay readable
    and greppable. Lives at ``run{run_idx}/version_0/metrics_rerun.csv``,
    alongside the original ``metrics.csv`` produced during training.
    """
    out_dir = run_dir / f"run{run_idx}" / "version_0"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "metrics_rerun.csv"
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for key in sorted(rerun_metrics):
            writer.writerow([key, rerun_metrics[key]])
    return out_path


def compare_metrics(original: dict[str, float], rerun: dict[str, float]) -> None:
    """Print a side-by-side comparison of original vs. rerun test metrics."""
    if not original:
        print("  (no original test metrics found in metrics.csv - skipping comparison)")
        return

    keys = sorted(set(original) | set(rerun))
    name_w = max(len(k) for k in keys)
    print(f"  {'metric':<{name_w}}  {'original':>12}  {'rerun':>12}  {'diff':>12}")
    print(f"  {'-' * name_w}  {'-' * 12}  {'-' * 12}  {'-' * 12}")

    max_abs_diff = 0.0
    max_abs_key = ""
    n_missing = 0
    for key in keys:
        orig = original.get(key)
        new = rerun.get(key)
        if orig is None or new is None:
            n_missing += 1
            o_str = f"{orig:>12.6f}" if orig is not None else f"{'-':>12}"
            n_str = f"{new:>12.6f}" if new is not None else f"{'-':>12}"
            print(f"  {key:<{name_w}}  {o_str}  {n_str}  {'-':>12}")
            continue
        diff = new - orig
        if abs(diff) > max_abs_diff:
            max_abs_diff = abs(diff)
            max_abs_key = key
        print(f"  {key:<{name_w}}  {orig:>12.6f}  {new:>12.6f}  {diff:>+12.6f}")

    print(f"  {'-' * name_w}  {'-' * 12}  {'-' * 12}  {'-' * 12}")
    print(f"  max |diff| = {max_abs_diff:.6f} ({max_abs_key})")
    if n_missing:
        print(f"  {n_missing} metric(s) present in only one of the two sets")


CATEGORIES = ("Acc", "ECE", "Loss", "Entropy", "Other")


def categorize_metric(name: str) -> str:
    """Bin a test metric into a scale-consistent category by its name.

    Checked in this order so substrings don't collide (a domain-acc metric still
    counts as Acc, but ece/entropy/loss are matched first since none of them are
    accuracies)."""
    n = name.lower()
    if "ece" in n:
        return "ECE"
    if "entropy" in n:
        return "Entropy"
    if "loss" in n:
        return "Loss"
    if "acc" in n:
        return "Acc"
    return "Other"


def compute_abs_diffs(original: dict[str, float], rerun: dict[str, float]) -> dict[str, float]:
    """Absolute deviation per metric present in both the original and rerun results."""
    return {k: abs(rerun[k] - original[k]) for k in set(original) & set(rerun)}


def print_run_summary(exp_key: str, n_seeds: int, category_diffs: dict[str, list[float]]) -> None:
    """Print the per-category reproduction summary pooled across all seeds.

    For each category two absolute-deviation averages are reported:
      - ``avg|diff| all``: mean over every metric in the category (diluted by the
        metrics that reproduce bitwise-exactly, i.e. diff == 0).
      - ``avg|diff| dev``: mean over only the metrics that deviate (diff > 0).
    No threshold is applied; precision noise vs. significant divergence is read off
    the magnitudes directly."""
    print(f"\n=== Reproduction summary: {exp_key}  ({n_seeds} seed(s)) ===")
    print(f"  {'category':<9}  {'metrics':>7}  {'deviating':>9}  {'avg|diff| all':>14}  {'avg|diff| dev':>14}")
    print(f"  {'-' * 9}  {'-' * 7}  {'-' * 9}  {'-' * 14}  {'-' * 14}")
    for cat in CATEGORIES:
        diffs = category_diffs.get(cat, [])
        if not diffs:
            continue
        deviating = [d for d in diffs if d > 0]
        all_mean = sum(diffs) / len(diffs)
        dev_mean = sum(deviating) / len(deviating) if deviating else 0.0
        print(f"  {cat:<9}  {len(diffs):>7}  {len(deviating):>9}  {all_mean:>14.2e}  {dev_mean:>14.2e}")


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
    parser.add_argument("--config", type=str, required=True, help="Path to run config YAML (e.g. src/train/configs/run/feature_fusion.yaml)")
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

    # if has_device_tensor_cores():
    #     torch.set_float32_matmul_precision("medium")

    # Absolute deviations pooled across all seeds, grouped by metric category.
    category_diffs: dict[str, list[float]] = {c: [] for c in CATEGORIES}

    for i, ckpt_path in enumerate(checkpoints):
        # Seed i was trained as run{i} with seed_everything(cfg.seed + i); evaluate_checkpoint
        # re-seeds with the same value so the frac<1.0 test subset matches the original run.
        print(f"\n--- Evaluating seed {i} ({ckpt_path.name}) ---")
        results = evaluate_checkpoint(ckpt_path, cfg, run_idx=i)
        # trainer.test() returns one dict per dataloader; flatten into a single metric map.
        rerun_metrics: dict[str, float] = {}
        for result_dict in results:
            rerun_metrics.update(result_dict)

        out_path = write_rerun_metrics(run_dir, i, rerun_metrics)
        print(f"  Wrote {len(rerun_metrics)} rerun metrics to {out_path}")

        # Seed i was trained as run{i}; its original test metrics live in run{i}/version_0/metrics.csv.
        original_metrics = load_original_test_metrics(run_dir, i)
        print(f"\n  Original vs. rerun (seed {i}):")
        compare_metrics(original_metrics, rerun_metrics)

        for name, diff in compute_abs_diffs(original_metrics, rerun_metrics).items():
            category_diffs[categorize_metric(name)].append(diff)

    print_run_summary(exp_key, len(checkpoints), category_diffs)


if __name__ == "__main__":
    main()
