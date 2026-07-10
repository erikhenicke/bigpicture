#!/usr/bin/env python3
"""Re-evaluate a trained model on both OOD and ID test splits.

Uses Lightning's trainer.test() to match the original training evaluation path.
The rerun goes through the same MultiScaleClassificationModule.on_test_epoch_end as training,
which now also emits per-class top-1 accuracy (overall and per region) and top-5
task accuracy. Two things happen per seed:
  1. The rerun is compared against the original metrics.csv to confirm the shared
     metrics still reproduce.
  2. The full rerun metric set (including the new per-class / top-5 metrics) is
     written to ``metrics_rerun.csv`` next to the original ``metrics.csv``.

Per-seed per-sample task logits are also captured during the rerun (via
``attach_logit_collector``) and cached to ``logits_rerun.npz`` (via
``write_rerun_logits``), keyed by the dataset's stable ``file_idx`` so a later
analysis can tie a logit row back to its source image (``loader_file_indices``).
``evaluate_checkpoint`` drives one seed's rerun end to end; ``compare_metrics``,
``categorize_metric``, ``compute_abs_diffs`` and ``print_run_summary`` turn the
original-vs-rerun metric maps into the printed reproduction report. ``main``
loads the run config and checkpoints (via ``results.utils``) and loops over
every seed checkpoint.
"""

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from lightning import Trainer, seed_everything

from train.run_experiment import make_data_loaders, make_model
from results.utils import find_best_checkpoints, load_hydra_config 


REPO_ROOT = Path(__file__).parent.parent.parent
LOG_RUNS = REPO_ROOT / "log" / "runs"
RUN_CONFIG_DIR = REPO_ROOT / "src" / "train" / "configs" / "run"


def find_run_dir(exp_key: str) -> Path | None:
    """Return the most recent log directory for the given experiment key.

    Searches every ``LOG_RUNS/<date>/`` directory for entries named
    ``train_<exp_key>-*`` and returns the one with the lexicographically latest
    (date dir, run dir) pair.

    Args:
        exp_key (str): Experiment key, matched against directories named
            ``train_<exp_key>-*``.

    Returns:
        Path | None: Path to the most recent matching run directory, or None if
            no run directory matches.
    """
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


def attach_logit_collector(module) -> dict[int, dict[str, Any]]:
    """Capture the model's per-sample task logits during trainer.test().

    The module's test path only folds logits into aggregate metric counters; the
    raw logits are never kept. Here we wrap two of its methods (without touching
    the model code) to record them:

      - ``_shared_forward`` stashes the task logits of the batch it just ran.
      - ``test_step`` pairs those stashed logits with the batch's labels and
        domain (region) codes, bucketed by ``dataloader_idx``.

    Because the wrappers only read the forward result and the batch, they do not
    change what test_step computes, so the reproduced metrics are unaffected. The
    returned dict is filled in place as testing proceeds, keyed by dataloader
    index (matching ``module.test_loader_names``).

    Args:
        module (MultiScaleClassificationModule): Lightning module to instrument
            in place; its ``_shared_forward`` and ``test_step`` methods are
            monkey-patched.

    Returns:
        dict[int, dict[str, Any]]: Empty at call time, filled in place during
            ``trainer.test()``. Maps dataloader index to a dict with keys
            ``"logits"``, ``"labels"``, ``"domains"``, each a list of per-batch
            ``torch.Tensor``s (task logits ``[B, num_classes]``, labels ``[B]``,
            region codes ``[B]``) to be concatenated by the caller.
    """
    collected: dict[int, dict[str, list[torch.Tensor]]] = {}
    last_logits: dict[str, torch.Tensor] = {}

    orig_shared_forward = module._shared_forward

    def wrapped_shared_forward(x, region_ids=None):
        result = orig_shared_forward(x, region_ids=region_ids)
        last_logits["task_logits"] = result["task_logits"].detach().float().cpu()
        return result

    module._shared_forward = wrapped_shared_forward

    orig_test_step = module.test_step

    def wrapped_test_step(batch, batch_idx, dataloader_idx=0):
        out = orig_test_step(batch, batch_idx, dataloader_idx)
        _, y, metadata = batch
        regions = metadata[:, module.hparams.domain_index].long()
        bucket = collected.setdefault(
            dataloader_idx, {"logits": [], "labels": [], "domains": []}
        )
        bucket["logits"].append(last_logits["task_logits"])
        bucket["labels"].append(y.detach().cpu())
        bucket["domains"].append(regions.detach().cpu())
        return out

    module.test_step = wrapped_test_step
    return collected


def loader_file_indices(loader) -> np.ndarray:
    """Recover the dataset file index of every sample served by ``loader``, in order.

    The loader wraps a ``WILDSSubset`` over ``FMoWMultiScaleDataset`` and is built
    with ``shuffle=False``, so its rows come out in subset-index order. Subset row
    ``r`` is ``base[subset.indices[r]]``, and the dataset keys its input files
    (``rgb_img_{file_idx}.png`` etc.) by ``base.full_idxs[idx]`` (see
    FMoWMultiScaleDataset.__getitem__). Composing the two gives the file index per
    served row, which is the stable id that ties a logit row back to its inputs.

    Args:
        loader (torch.utils.data.DataLoader): Unshuffled test dataloader whose
            ``.dataset`` is a WILDS subset over an ``FMoWMultiScaleDataset``.

    Returns:
        np.ndarray: 1-D int array of length ``len(loader.dataset)``, the global
            ``file_idx`` of each served row in iteration order.
    """
    subset = loader.dataset
    base = subset.dataset
    return np.asarray(base.full_idxs)[np.asarray(subset.indices)]


def write_rerun_logits(
    run_dir: Path,
    run_idx: int,
    collected: dict[int, dict[str, Any]],
    loader_names: list[str],
) -> Path:
    """Persist the captured per-sample logits next to the rerun metrics.

    Written as a single compressed ``logits_rerun.npz`` at
    ``run{run_idx}/version_0/``, alongside ``metrics_rerun.csv``. For each test
    dataloader ``<name>`` (from ``loader_names``) four arrays are stored:
      - ``<name>/logits``   float32 ``[N, num_classes]``
      - ``<name>/labels``   int64   ``[N]`` (FMoW task labels)
      - ``<name>/domains``  int64   ``[N]`` (raw WILDS region codes)
      - ``<name>/file_idx`` int64   ``[N]`` (dataset file index, the input-file id)
    Rows are in test-iteration order and aligned across the four arrays, so
    ``file_idx[r]`` is the input-file id of the sample whose logits are ``logits[r]``.

    Args:
        run_dir (Path): Run's top-level log directory.
        run_idx (int): Seed index; output goes under ``run{run_idx}/version_0/``.
        collected (dict[int, dict[str, Any]]): Per-dataloader capture as filled by
            ``attach_logit_collector`` and augmented with a ``"file_idx"``
            ``np.ndarray`` per bucket (see ``evaluate_checkpoint``). Keys
            ``"logits"``, ``"labels"``, ``"domains"`` are lists of tensors to be
            concatenated.
        loader_names (list[str]): Display name for each dataloader index (e.g.
            ``cfg.data.test_loader_names``); indices without a name fall back to
            ``"loader{idx}"``.

    Returns:
        Path: Path to the written ``logits_rerun.npz``.

    Raises:
        ValueError: If a bucket's captured file-index count does not match its
            logit row count (loader order and captured rows out of sync).
    """
    out_dir = run_dir / f"run{run_idx}" / "version_0"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "logits_rerun.npz"

    arrays: dict[str, np.ndarray] = {}
    for idx, bucket in collected.items():
        name = loader_names[idx] if idx < len(loader_names) else f"loader{idx}"
        logits = torch.cat(bucket["logits"]).numpy()
        file_idx = np.asarray(bucket["file_idx"])
        if file_idx.shape[0] != logits.shape[0]:
            raise ValueError(
                f"{name}: {file_idx.shape[0]} file indices but {logits.shape[0]} "
                "logit rows; loader order and captured rows are out of sync."
            )
        arrays[f"{name}/logits"] = logits
        arrays[f"{name}/labels"] = torch.cat(bucket["labels"]).numpy()
        arrays[f"{name}/domains"] = torch.cat(bucket["domains"]).numpy()
        arrays[f"{name}/file_idx"] = file_idx.astype(np.int64)

    np.savez_compressed(out_path, **arrays)
    return out_path


def evaluate_checkpoint(ckpt_path: Path, cfg, run_idx: int) -> tuple[list[dict], dict]:
    """Run trainer.test() on a checkpoint, reproducing run_experiment.py's test metrics.

    Mirrors `_run_once` in run_experiment.py: seed -> make_data_loaders(cfg, run_idx) -> model ->
    best weights -> trainer.test(). Metrics are computed in
    MultiScaleClassificationModule.on_test_epoch_end, independent of the logger.

    Reproducing the seed and the loader-build order matters when cfg.data.frac < 1.0:
    WILDS draws the random frac subset via the global np.random RNG inside get_subset,
    so the exact test images depend on seed_everything(cfg.seed + run_idx) and the order
    of the get_subset calls. The subset is drawn before training, so it is fully
    determined by these two things (not by the trained weights). `run_idx` is the
    original run index (run{run_idx}) for this checkpoint.

    Args:
        ckpt_path (Path): Path to the seed's checkpoint file (``.ckpt``).
        cfg (omegaconf.DictConfig): Hydra run config, as loaded by
            ``results.utils.load_hydra_config``.
        run_idx (int): Original run index this checkpoint belongs to
            (``run{run_idx}``), used to reproduce the seed and the frac<1.0 test
            subset.

    Returns:
        tuple[list[dict], dict]: ``(results, logits)`` where ``results`` is
            trainer.test()'s list of per-dataloader metric dicts, and ``logits``
            is the per-dataloader capture from ``attach_logit_collector``, with a
            ``"file_idx"`` ``np.ndarray`` added per bucket (from
            ``loader_file_indices``).
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

    # Capture the per-sample task logits as the test runs (see attach_logit_collector).
    logits = attach_logit_collector(module)

    trainer = Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
    )

    results = trainer.test(model=module, dataloaders=test_loaders)

    # Tie each captured logit row back to its dataset file index. Loaders run in the
    # order they are passed (matching dataloader_idx) and unshuffled, so the file
    # indices of a loader line up row-for-row with that bucket's captured logits.
    for idx, loader in enumerate(test_loaders):
        if idx in logits:
            logits[idx]["file_idx"] = loader_file_indices(loader)

    return results, logits


def load_original_test_metrics(run_dir: Path, run_idx: int) -> dict[str, float]:
    """Read the original test metrics that run_experiment.py wrote via CSVLogger.

    During training, `trainer.test(...)` logs one final row of `test/...` columns to
    `run{run_idx}/version_0/metrics.csv`. We return that row as a flat metric->value dict.

    Args:
        run_dir (Path): Run's top-level log directory.
        run_idx (int): Seed index; reads ``run{run_idx}/version_0/metrics.csv``.

    Returns:
        dict[str, float]: Map of ``test/*`` metric name to value from the last
            row with any non-empty test column; empty dict if the file is
            missing or has no such row.
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

    Args:
        run_dir (Path): Run's top-level log directory.
        run_idx (int): Seed index; output goes to
            ``run{run_idx}/version_0/metrics_rerun.csv``.
        rerun_metrics (dict[str, float]): Flat metric name -> value map to write.

    Returns:
        Path: Path to the written ``metrics_rerun.csv``.
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
    """Print a side-by-side comparison of original vs. rerun test metrics.

    Prints a table (one row per metric key present in either dict, "-" for a
    missing value) followed by the maximum absolute difference and a count of
    metrics present in only one of the two sets.

    Args:
        original (dict[str, float]): Metric values from the original training
            run (see ``load_original_test_metrics``); an empty dict skips the
            comparison.
        rerun (dict[str, float]): Metric values from the rerun (see
            ``evaluate_checkpoint``).
    """
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
    accuracies).

    Args:
        name (str): Metric name (case-insensitive match against "ece", "entropy",
            "loss", "acc").

    Returns:
        str: One of ``CATEGORIES`` ("Acc", "ECE", "Loss", "Entropy", "Other").
    """
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
    """Absolute deviation per metric present in both the original and rerun results.

    Args:
        original (dict[str, float]): Original metric values (see
            ``load_original_test_metrics``).
        rerun (dict[str, float]): Rerun metric values (see ``evaluate_checkpoint``).

    Returns:
        dict[str, float]: Metric name -> ``abs(rerun[k] - original[k])``,
            restricted to keys present in both inputs.
    """
    return {k: abs(rerun[k] - original[k]) for k in set(original) & set(rerun)}


def print_run_summary(exp_key: str, n_seeds: int, category_diffs: dict[str, list[float]]) -> None:
    """Print the per-category reproduction summary pooled across all seeds.

    For each category two absolute-deviation averages are reported:
      - ``avg|diff| all``: mean over every metric in the category (diluted by the
        metrics that reproduce bitwise-exactly, i.e. diff == 0).
      - ``avg|diff| dev``: mean over only the metrics that deviate (diff > 0).
    No threshold is applied; precision noise vs. significant divergence is read off
    the magnitudes directly.

    Args:
        exp_key (str): Experiment key, used in the summary header.
        n_seeds (int): Number of seed checkpoints evaluated, used in the summary header.
        category_diffs (dict[str, list[float]]): Absolute deviations (see
            ``compute_abs_diffs``) pooled across all seeds, grouped by category
            (see ``categorize_metric``); categories with no entries are skipped.
    """
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
    """Load a run config YAML from the given path.

    Args:
        config_path (Path): Path to the run config YAML (e.g.
            ``src/train/configs/run/feature_fusion.yaml``).

    Returns:
        tuple[dict, str]: ``(cfg, config_name)`` where ``cfg`` is the parsed YAML
            (with an ``"experiments"`` key mapping experiment keys to their
            definitions) and ``config_name`` is the file's stem.
    """
    if not config_path.exists():
        print(f"Run config not found: {config_path}", file=sys.stderr)
        sys.exit(1)
    with config_path.open() as f:
        cfg = yaml.safe_load(f)
    return cfg, config_path.stem


def main() -> None:
    """CLI entry point: evaluate every seed checkpoint of an experiment and print a reproduction summary.

    Parses ``--config``/``--run-name`` (and optional batch-size/num-workers
    overrides), locates the run directory and its seed checkpoints, then for each
    checkpoint reruns evaluation (``evaluate_checkpoint``), writes the rerun
    metrics and logits, compares against the original metrics
    (``compare_metrics``), and accumulates per-category deviations for the final
    ``print_run_summary``.
    """
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
        results, logits = evaluate_checkpoint(ckpt_path, cfg, run_idx=i)
        # trainer.test() returns one dict per dataloader; flatten into a single metric map.
        rerun_metrics: dict[str, float] = {}
        for result_dict in results:
            rerun_metrics.update(result_dict)

        out_path = write_rerun_metrics(run_dir, i, rerun_metrics)
        print(f"  Wrote {len(rerun_metrics)} rerun metrics to {out_path}")

        logits_path = write_rerun_logits(run_dir, i, logits, cfg.data.test_loader_names)
        n_samples = sum(int(t.shape[0]) for b in logits.values() for t in b["logits"])
        print(f"  Wrote logits for {n_samples} samples to {logits_path}")

        # Seed i was trained as run{i}; its original test metrics live in run{i}/version_0/metrics.csv.
        original_metrics = load_original_test_metrics(run_dir, i)
        print(f"\n  Original vs. rerun (seed {i}):")
        compare_metrics(original_metrics, rerun_metrics)

        for name, diff in compute_abs_diffs(original_metrics, rerun_metrics).items():
            category_diffs[categorize_metric(name)].append(diff)

    print_run_summary(exp_key, len(checkpoints), category_diffs)


if __name__ == "__main__":
    main()
