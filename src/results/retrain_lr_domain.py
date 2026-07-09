#!/usr/bin/env python3
"""Retrain only the LR domain head of a trained run, on frozen LR features.

Motivation: when a run is trained with ``lr_domain_loss_coeff = 0`` (the
``*_no_domain`` ablations) the LR domain classifier never receives a gradient, so
its head stays at initialization and the logged ``lr-domain-acc`` is meaningless.
We still want to know how well the *LR features themselves* separate the regions
in that setting. Fully retraining the model is too expensive, so this script does
a cheap linear probe instead:

  1. Load the trained model for each seed checkpoint.
  2. Freeze everything except the LR domain head (so the LR branch -- and its
     features -- stay exactly as trained).
  3. Cache the frozen LR features for the train split once, then retrain the
     (reinitialized) LR domain head on those cached features for a number of
     epochs. If the run's ``lr_domain_loss_coeff`` is 0 it is bumped to 0.2 so the
     domain loss is actually applied; otherwise the run's own coefficient is used.
  4. Re-evaluate the LR domain accuracy on the test splits with the retrained head.

Only the LR-domain test metrics are recomputed (the task/branch metrics are
unchanged because nothing but the head moved), so the script *updates* those keys
in the run's canonical per-seed metrics file rather than rewriting it: it writes
to ``metrics_rerun.csv`` when that file exists (the long-format rerun output that
``results/utils.load_seed_test_metrics`` prefers), otherwise to the training
``metrics.csv``.

Key functions: ``build_eval_loaders`` builds the (unaugmented) train/test
loaders; ``cache_lr_features`` captures the LR domain head's input features via a
forward hook; ``train_head`` retrains the (reinitialized) head on those cached
features; ``evaluate_loader`` computes LR-domain metrics with the retrained head;
``update_metrics_file`` writes the updated metrics back into the run's metrics
file; ``main`` drives the per-seed loop end to end.
"""

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import torch
import yaml
from lightning import seed_everything

from train.run_experiment import make_model, _parse_spatial_cfg
from train.utils import make_multiscale_dataset, make_multiscale_loader
from results.utils import find_best_checkpoints, find_run_dir, load_hydra_config
from models.utils import (
    make_eval_state,
    update_lr_domain_metrics,
    compute_final_lr_domain_metrics,
)


REPO_ROOT = Path(__file__).parent.parent.parent


def build_eval_loaders(cfg, run_idx: int) -> Tuple[torch.utils.data.DataLoader, List[torch.utils.data.DataLoader]]:
    """Build an unaugmented, unshuffled train loader plus the test loaders.

    Mirrors ``run_experiment.make_data_loaders`` but uses the *eval* dataset
    (augment off) for the train split too: the probe caches LR features once, so
    they must be deterministic. A single dataset is built and ``get_subset`` is
    called per split, matching the original eval construction.

    Args:
        cfg (omegaconf.DictConfig): Hydra run config (see
            ``results.utils.load_hydra_config``).
        run_idx (int): Seed/run index, forwarded as ``feature_run_idx`` when the
            data source is ``"features"``.

    Returns:
        tuple[torch.utils.data.DataLoader, list[torch.utils.data.DataLoader]]:
            ``(train_loader, test_loaders)``, both unshuffled, built from
            ``cfg.data.train_split`` and ``cfg.data.test_splits`` respectively.
    """
    sc = _parse_spatial_cfg(cfg)
    spatial_kwargs = dict(
        spatial_coord_grid=sc["needs_coord_grid"],
        spatial_overlap_mask=sc["needs_overlap_mask"],
        overlap_mask_type=sc["overlap_mask_type"],
    )
    dataset = make_multiscale_dataset(
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

    def loader(split: str):
        return make_multiscale_loader(
            dataset,
            split=split,
            frac=cfg.data.frac,
            batch_size=cfg.data.batch_size,
            num_workers=cfg.data.num_workers,
            shuffle=False,
        )

    train_loader = loader(cfg.data.train_split)
    test_loaders = [loader(split) for split in cfg.data.test_splits]
    return train_loader, test_loaders


def move_batch(x: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    """Move every tensor value of a batch input dict to ``device``, leaving non-tensors as is.

    Args:
        x (dict[str, torch.Tensor]): Model input dict (e.g. keys like ``"rgb"``,
            ``"landsat"``, ``"coords"``).
        device (torch.device): Target device.

    Returns:
        dict[str, torch.Tensor]: Same keys, tensor values moved to ``device``.
    """
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in x.items()}


def cache_lr_features(module, loader, device) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run the frozen model over ``loader`` and collect the LR domain head's inputs.

    A forward pre-hook on ``lr_domain_classifier`` captures its input -- the LR
    feature vector -- for every model type without needing to know how each model
    routes its branches. Only samples whose domain target is in-label-space
    (``>= 0``; Asia maps to -1 under Leave-Asia-Out) are kept.

    Args:
        module (MultiScaleClassificationModule): Trained module in eval mode;
            must expose ``model.lr_domain_classifier``.
        loader (torch.utils.data.DataLoader): Data loader yielding
            ``(x, y, metadata)`` batches.
        device (torch.device): Device to run the forward pass on.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: ``(features, targets)`` — the cached
            LR domain-head input features, shape
            ``(num_valid_samples, feature_dim)``, and the corresponding
            domain-label targets, shape ``(num_valid_samples,)``, both on CPU.
    """
    captured: Dict[str, torch.Tensor] = {}

    def hook(_mod, inp):
        captured["feat"] = inp[0].detach()

    handle = module.model.lr_domain_classifier.register_forward_pre_hook(hook)
    feats: List[torch.Tensor] = []
    targets: List[torch.Tensor] = []
    module.eval()
    try:
        with torch.no_grad():
            for x, _y, metadata in loader:
                x = move_batch(x, device)
                regions = metadata[:, module.hparams.domain_index].long().to(device)
                module._shared_forward(x, region_ids=regions)
                domain_targets = module._domain_targets(regions)
                valid = domain_targets >= 0
                feats.append(captured["feat"][valid].cpu())
                targets.append(domain_targets[valid].cpu())
    finally:
        handle.remove()
    return torch.cat(feats), torch.cat(targets)


def train_head(
    head: torch.nn.Module,
    features: torch.Tensor,
    targets: torch.Tensor,
    device: torch.device,
    lr: float,
    coeff: float,
    epochs: int,
    batch_size: int,
) -> None:
    """Retrain ``head`` (a reinitialized LR domain classifier) on cached features.

    Args:
        head (torch.nn.Module): LR domain classifier head to retrain in place;
            its parameters are reset before training.
        features (torch.Tensor): Cached LR domain-head input features, shape
            ``(num_samples, feature_dim)`` (see ``cache_lr_features``).
        targets (torch.Tensor): Domain-label targets, shape ``(num_samples,)``.
        device (torch.device): Device to run training on.
        lr (float): Adam learning rate.
        coeff (float): Multiplier applied to the cross-entropy loss.
        epochs (int): Number of training epochs over the cached features.
        batch_size (int): Mini-batch size.
    """
    head.reset_parameters()
    optimizer = torch.optim.Adam(head.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()
    n = targets.shape[0]
    for epoch in range(epochs):
        perm = torch.randperm(n)
        epoch_loss = 0.0
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            fb = features[idx].to(device)
            tb = targets[idx].to(device)
            logits = head(fb)
            loss = coeff * criterion(logits, tb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * idx.shape[0]
        print(f"    epoch {epoch + 1:>3}/{epochs}  train domain loss = {epoch_loss / n:.4f}")


def evaluate_loader(module, loader, loader_name: str, device) -> Dict[str, float]:
    """Compute the LR-domain test metrics on one loader with the retrained head.

    Reuses the same aggregation the training/eval path uses
    (``compute_final_lr_domain_metrics``), so the metric keys match exactly. The
    held-out region (Asia under Leave-Asia-Out) has no in-label-space targets, so
    that loader yields an empty dict and is skipped.

    Args:
        module (MultiScaleClassificationModule): Trained module (with the
            retrained LR domain head) in eval mode.
        loader (torch.utils.data.DataLoader): Data loader yielding
            ``(x, y, metadata)`` batches.
        loader_name (str): Split/loader name, used as the
            ``test/<loader_name>-*`` key prefix in the returned metrics.
        device (torch.device): Device to run the forward pass on.

    Returns:
        dict[str, float]: ``test/<loader_name>-lr-domain-acc[-<region>]`` metric
            values, or an empty dict if the loader has no in-label-space samples.
    """
    state = make_eval_state()
    module.eval()
    with torch.no_grad():
        for x, _y, metadata in loader:
            x = move_batch(x, device)
            regions = metadata[:, module.hparams.domain_index].long().to(device)
            result = module._shared_forward(x, region_ids=regions)
            preds = result["lr_domain_logits"].argmax(dim=1)
            domain_targets = module._domain_targets(regions)
            valid = domain_targets >= 0
            if valid.any():
                update_lr_domain_metrics(state, preds[valid], domain_targets[valid])
    if not state["lr_domain_preds"]:
        return {}
    metrics = compute_final_lr_domain_metrics(state, module.domain_names)
    return {f"test/{loader_name}-{k}": v for k, v in metrics.items()}


def update_metrics_file(seed_dir: Path, metrics: Dict[str, float]) -> Path:
    """Update the LR-domain test metrics in the seed's canonical metrics file.

    Writes to ``metrics_rerun.csv`` if it exists (long ``metric,value`` format),
    otherwise the training ``metrics.csv`` (wide format -- the final test row is
    updated in place). Only the keys in ``metrics`` are touched; everything else
    is preserved.

    Args:
        seed_dir (Path): Seed's run directory (``run{run_idx}``), containing
            ``version_0/``.
        metrics (dict[str, float]): LR-domain metric name -> value map to write
            (see ``evaluate_loader``).

    Returns:
        Path: Path to the updated (or newly created) metrics file --
            ``version_0/metrics_rerun.csv`` if it existed, else
            ``version_0/metrics.csv`` if it existed, else a freshly created
            ``version_0/metrics_rerun.csv``.
    """
    version_dir = seed_dir / "version_0"
    rerun_path = version_dir / "metrics_rerun.csv"
    plain_path = version_dir / "metrics.csv"

    if rerun_path.exists():
        with rerun_path.open() as f:
            values = {row["metric"]: row["value"] for row in csv.DictReader(f)}
        values.update({k: v for k, v in metrics.items()})
        with rerun_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["metric", "value"])
            for key in sorted(values):
                writer.writerow([key, values[key]])
        return rerun_path

    if plain_path.exists():
        df = pd.read_csv(plain_path)
        test_cols = [c for c in df.columns if c.startswith("test/")]
        # The test results were logged as the last row carrying any test/ column.
        if test_cols:
            test_rows = df.dropna(subset=[test_cols[0]])
            target_idx = test_rows.index[-1] if not test_rows.empty else df.index[-1]
        else:
            target_idx = df.index[-1]
        for key, value in metrics.items():
            if key not in df.columns:
                df[key] = pd.NA
            df.at[target_idx, key] = value
        df.to_csv(plain_path, index=False)
        return plain_path

    # Neither file exists: create the long-format rerun file from scratch.
    version_dir.mkdir(parents=True, exist_ok=True)
    with rerun_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for key in sorted(metrics):
            writer.writerow([key, metrics[key]])
    return rerun_path


def main() -> None:
    """CLI entry point: retrain the LR domain head of every seed and update its metrics.

    Parses ``--config``/``--run-name`` plus head-training hyperparameters, loads
    the run config and checkpoints, then for each seed checkpoint: freezes the
    whole model except the LR domain head, caches frozen LR features on the train
    split (``cache_lr_features``), retrains the head (``train_head``), evaluates
    the LR-domain metrics on the test splits (``evaluate_loader``), and writes
    them back (``update_metrics_file``). Prints a final mean-over-seeds summary.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, required=True, help="Path to run config YAML (e.g. src/train/configs/run/feature_fusion.yaml)")
    parser.add_argument("--run-name", type=str, required=True, help="Experiment key to probe (e.g. film_no_domain)")
    parser.add_argument("--epochs", type=int, default=50, help="Head-training epochs over the cached features")
    parser.add_argument("--lr", type=float, default=None, help="Head learning rate (default: optimizer.lr * domain_optimizer_lr_factor)")
    parser.add_argument("--head-batch-size", type=int, default=512, help="Mini-batch size for head training on cached features")
    parser.add_argument("--batch-size", type=int, default=None, help="Override data batch size (feature extraction / eval)")
    parser.add_argument("--num-workers", type=int, default=None, help="Override dataloader workers")
    args = parser.parse_args()

    sys.path.insert(0, str(REPO_ROOT / "src"))

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = Path.cwd() / config_path
    with config_path.open() as f:
        run_config = yaml.safe_load(f)

    exp_key = args.run_name
    experiments = run_config.get("experiments", {})
    if exp_key not in experiments:
        print(f"Experiment '{exp_key}' not found in {config_path.name}", file=sys.stderr)
        print(f"Available: {', '.join(experiments)}", file=sys.stderr)
        sys.exit(1)

    run_dir = find_run_dir(exp_key)
    if run_dir is None:
        print(f"No run directory found for experiment '{exp_key}'", file=sys.stderr)
        sys.exit(1)
    print(f"\nProbing LR domain head: {exp_key}")
    print(f"Run directory: {run_dir}")

    cfg = load_hydra_config(run_dir)
    if args.batch_size is not None:
        cfg.data.batch_size = args.batch_size
    if args.num_workers is not None:
        cfg.data.num_workers = args.num_workers

    orig_coeff = float(cfg.model.get("lr_domain_loss_coeff", 0.0))
    coeff = 0.2 if orig_coeff == 0.0 else orig_coeff
    if orig_coeff == 0.0:
        print(f"lr_domain_loss_coeff is 0 -> using {coeff} for the probe (LR branch stays frozen)")
    else:
        print(f"lr_domain_loss_coeff is {orig_coeff} -> reusing it for the probe (LR branch stays frozen)")

    base_lr = float(cfg.optim.optimizer.lr) * float(cfg.optim.get("domain_optimizer_lr_factor", 1.0))
    head_lr = args.lr if args.lr is not None else base_lr
    print(f"Head learning rate: {head_lr}")

    checkpoints = find_best_checkpoints(run_dir)
    if not checkpoints:
        print("No checkpoints found!", file=sys.stderr)
        sys.exit(1)
    print(f"Found {len(checkpoints)} seed checkpoints.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    pooled: Dict[str, List[float]] = {}
    for ckpt_path in checkpoints:
        # Seed dir name (run{i}) doubles as the run index used by the frac<1 subset RNG.
        run_idx = int(ckpt_path.parent.name.removeprefix("run"))
        print(f"\n--- Seed {run_idx} ({ckpt_path.name}) ---")
        seed_everything(cfg.seed + run_idx, workers=True)

        module = make_model(cfg, run_idx)
        if not module.has_lr_domain_classifier:
            print(f"  Run '{exp_key}' has no LR domain classifier; nothing to probe.", file=sys.stderr)
            sys.exit(1)

        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        module.load_state_dict(checkpoint["state_dict"])
        module.to(device)

        # Linear probe: freeze the whole model, then unfreeze only the LR domain head.
        for param in module.parameters():
            param.requires_grad_(False)
        for param in module.model.lr_domain_classifier.parameters():
            param.requires_grad_(True)

        train_loader, test_loaders = build_eval_loaders(cfg, run_idx)

        print("  Caching frozen LR features for the train split ...")
        features, targets = cache_lr_features(module, train_loader, device)
        print(f"  Cached {targets.shape[0]} train features (dim {features.shape[1]}); retraining head ...")
        train_head(
            module.model.lr_domain_classifier,
            features,
            targets,
            device,
            lr=head_lr,
            coeff=coeff,
            epochs=args.epochs,
            batch_size=args.head_batch_size,
        )

        seed_metrics: Dict[str, float] = {}
        for loader, name in zip(test_loaders, cfg.data.test_loader_names):
            seed_metrics.update(evaluate_loader(module, loader, name, device))

        if not seed_metrics:
            print("  No in-label-space test samples for any loader; nothing written.")
            continue

        for key, value in seed_metrics.items():
            print(f"    {key} = {value:.4f}")
            pooled.setdefault(key, []).append(value)

        out_path = update_metrics_file(run_dir / f"run{run_idx}", seed_metrics)
        print(f"  Updated {len(seed_metrics)} LR-domain metrics in {out_path}")

    if pooled:
        print(f"\n=== Mean over {len(checkpoints)} seed(s): {exp_key} ===")
        for key in sorted(pooled):
            vals = pooled[key]
            print(f"  {key} = {sum(vals) / len(vals):.4f}  (n={len(vals)})")


if __name__ == "__main__":
    main()
