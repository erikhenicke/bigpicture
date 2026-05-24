#!/usr/bin/env python3
"""Evaluate a trained model per-class on both OOD and ID test splits.

Prompts user to select a run config and experiment, loads all seed checkpoints,
runs inference, and writes a CSV with per-class metrics into the run directory.
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from lightning import Trainer, seed_everything
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).parent.parent.parent
LOG_RUNS = REPO_ROOT / "log" / "runs"
RUN_CONFIG_DIR = REPO_ROOT / "src" / "train" / "configs" / "run"

CATEGORIES = [
    "airport", "airport_hangar", "airport_terminal", "amusement_park",
    "aquaculture", "archaeological_site", "barn", "border_checkpoint",
    "burial_site", "car_dealership", "construction_site", "crop_field",
    "dam", "debris_or_rubble", "educational_institution", "electric_substation",
    "factory_or_powerplant", "fire_station", "flooded_road", "fountain",
    "gas_station", "golf_course", "ground_transportation_station", "helipad",
    "hospital", "impoverished_settlement", "interchange", "lake_or_pond",
    "lighthouse", "military_facility", "multi-unit_residential", "nuclear_powerplant",
    "office_building", "oil_or_gas_facility", "park", "parking_lot_or_garage",
    "place_of_worship", "police_station", "port", "prison",
    "race_track", "railway_bridge", "recreational_facility", "road_bridge",
    "runway", "shipyard", "shopping_mall", "single-unit_residential",
    "smokestack", "solar_farm", "space_facility", "stadium",
    "storage_tank", "surface_mine", "swimming_pool", "toll_booth",
    "tower", "tunnel_opening", "waste_disposal", "water_treatment_facility",
    "wind_farm", "zoo",
]

FIVE_REGIONS = ["Asia", "Europe", "Africa", "Americas", "Oceania"]


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
    if not hasattr(cfg, "trainer") or not hasattr(cfg.trainer, "alternating_freeze"):
        cfg.trainer.alternating_freeze = False
        cfg.trainer.alternating_freeze_period = 1
    return cfg


def evaluate_checkpoint(ckpt_path: Path, cfg, domain_index: int) -> dict:
    """Run inference on a single checkpoint, return per-class per-split per-region results."""
    from train.run_experiment import make_data_loaders, make_model
    from train.evaluate_checkpoint import load_model_from_checkpoint

    module = load_model_from_checkpoint(ckpt_path, cfg)
    module.eval()

    _, _, test_loaders = make_data_loaders(cfg)
    test_loader_names = list(cfg.data.test_loader_names)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    module = module.to(device)

    results = {}
    for loader_idx, (loader, loader_name) in enumerate(zip(test_loaders, test_loader_names)):
        all_preds = []
        all_targets = []
        all_regions = []

        with torch.no_grad():
            for batch in loader:
                x, y, metadata = batch
                x = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in x.items()}
                y = y.to(device)
                regions = metadata[:, domain_index].long().to(device)

                logits = module(x, region_ids=regions)
                preds = logits.argmax(dim=1)

                all_preds.append(preds.cpu())
                all_targets.append(y.cpu())
                all_regions.append(regions.cpu())

        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)
        all_regions = torch.cat(all_regions)

        results[loader_name] = {
            "preds": all_preds,
            "targets": all_targets,
            "regions": all_regions,
        }

    return results


def compute_per_class_metrics(all_seed_results: list[dict]) -> pd.DataFrame:
    """Compute per-class metrics averaged over seeds."""
    num_classes = len(CATEGORIES)
    loader_names = list(all_seed_results[0].keys())

    rows = []
    for class_idx in range(num_classes):
        row = {"class_id": class_idx, "category": CATEGORIES[class_idx]}

        for loader_name in loader_names:
            seed_accs = []
            region_seed_accs = defaultdict(list)

            for seed_result in all_seed_results:
                data = seed_result[loader_name]
                preds = data["preds"]
                targets = data["targets"]
                regions = data["regions"]

                mask = targets == class_idx
                if mask.sum() == 0:
                    continue

                acc = (preds[mask] == targets[mask]).float().mean().item()
                seed_accs.append(acc)

                for rid, region_name in enumerate(FIVE_REGIONS):
                    region_mask = mask & (regions == rid)
                    if region_mask.sum() == 0:
                        continue
                    racc = (preds[region_mask] == targets[region_mask]).float().mean().item()
                    region_seed_accs[region_name].append(racc)

            prefix = loader_name.replace("test-", "")
            if seed_accs:
                row[f"{prefix}_acc_mean"] = np.mean(seed_accs)
                row[f"{prefix}_acc_std"] = np.std(seed_accs)
                row[f"{prefix}_n_samples"] = int(mask.sum().item())
            else:
                row[f"{prefix}_acc_mean"] = np.nan
                row[f"{prefix}_acc_std"] = np.nan
                row[f"{prefix}_n_samples"] = 0

            for region_name in FIVE_REGIONS:
                col_prefix = f"{prefix}_{region_name.lower()}"
                if region_seed_accs[region_name]:
                    row[f"{col_prefix}_acc_mean"] = np.mean(region_seed_accs[region_name])
                    row[f"{col_prefix}_acc_std"] = np.std(region_seed_accs[region_name])
                else:
                    row[f"{col_prefix}_acc_mean"] = np.nan
                    row[f"{col_prefix}_acc_std"] = np.nan

        rows.append(row)

    # Add worst-group row per loader
    df = pd.DataFrame(rows)
    summary_row = {"class_id": -1, "category": "WORST_GROUP_PER_CLASS"}
    for loader_name in loader_names:
        prefix = loader_name.replace("test-", "")
        for region_name in FIVE_REGIONS:
            col = f"{prefix}_{region_name.lower()}_acc_mean"
            if col in df.columns:
                valid = df[col].dropna()
                if not valid.empty:
                    summary_row[f"{prefix}_{region_name.lower()}_worst_class"] = CATEGORIES[int(valid.idxmin())]
                    summary_row[f"{prefix}_{region_name.lower()}_worst_acc"] = valid.min()

    return df


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
    domain_index = cfg.domain_index

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

    all_seed_results = []
    for i, ckpt_path in enumerate(checkpoints):
        print(f"\n--- Evaluating seed {i} ({ckpt_path.name}) ---")
        results = evaluate_checkpoint(ckpt_path, cfg, domain_index)
        all_seed_results.append(results)

    print("\nComputing per-class metrics...")
    df = compute_per_class_metrics(all_seed_results)

    output_path = run_dir / "per_class_metrics.csv"
    if output_path.exists():
        print(f"\nWARNING: {output_path} already exists, overwriting.", file=sys.stderr)
    df.to_csv(output_path, index=False)
    print(f"\nResults written to: {output_path}")

    # Print summary
    loader_names = list(all_seed_results[0].keys())
    for loader_name in loader_names:
        prefix = loader_name.replace("test-", "")
        acc_col = f"{prefix}_acc_mean"
        if acc_col in df.columns:
            overall = df[acc_col].mean()
            worst = df[acc_col].min()
            worst_cat = CATEGORIES[int(df[acc_col].idxmin())]
            print(f"\n{loader_name}:")
            print(f"  Mean per-class acc: {overall*100:.2f}%")
            print(f"  Worst class: {worst_cat} ({worst*100:.2f}%)")


if __name__ == "__main__":
    main()
