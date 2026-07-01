#!/usr/bin/env python3
"""
Print per-seed metric values for each group in an eval YAML.

PYTHONPATH=src uv run src/results/per_seed_compare.py src/train/configs/eval/feature_fusion.yaml
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

from results.eval_metrics import format_metric_name
from results.utils import (
    EVAL_CONFIG_DIR,
    find_run_dir,
    format_experiment_name,
    load_run_configs,
    load_seed_test_metrics,
    load_translations,
    parse_run_ref,
    resolve_experiments,
)

SEED_LABEL_W = 9   # fits "mean±std"
COL_W = 14         # fits "32.92±0.34" with room


def load_run_seed_metrics(
    run_dir: Path | None, metrics: list[str]
) -> list[dict[str, float | None]]:
    """Return one metric dict per seed dir under run_dir, or [] if run_dir is None."""
    if run_dir is None:
        return []
    result = []
    for seed_dir in sorted(run_dir.glob("run*")):
        m = load_seed_test_metrics(seed_dir)
        result.append({k: (m.get(k) if m else None) for k in metrics})
    return result


def fmt_val(v: float | None, is_acc: bool) -> str:
    if v is None:
        return "—"
    return f"{v * 100:.2f}" if is_acc else f"{v:.4f}"


def fmt_mean_std(vals: list[float], is_acc: bool) -> str:
    if not vals:
        return "—"
    mean, std = float(np.mean(vals)), float(np.std(vals))
    if is_acc:
        return f"{mean * 100:.2f}±{std * 100:.2f}"
    return f"{mean:.4f}±{std:.4f}"


def print_group(
    group: dict,
    primary_metrics: list[str],
    run_experiments: dict,
    translations: dict,
    default_config: str,
) -> None:
    metrics = primary_metrics + group.get("additional_metrics", [])
    is_acc = [m.endswith("acc") for m in metrics]
    metric_display = group.get("metric_display_names", {})
    model_display = group.get("model_display_names", {})
    param_display = group.get("param_display_names", {})

    run_data: list[tuple[str, list[dict[str, float | None]]]] = []
    for ref in group["runs"]:
        _, exp_key = parse_run_ref(ref, default_config)
        run_dir = find_run_dir(exp_key)
        seeds = load_run_seed_metrics(run_dir, metrics)
        name = format_experiment_name(
            ref, run_experiments, translations,
            model_overrides=model_display,
            param_overrides=param_display,
        )
        run_data.append((name, seeds))

    col_headers = [format_metric_name(m, translations, metric_display) for m in metrics]
    name_w = max(
        len("Experiment"),
        max((len(n) for n, _ in run_data), default=0),
    )

    print(f"\n=== {group['name']} ===\n")
    header = f"  {'Experiment':<{name_w}}  {'Run':<{SEED_LABEL_W}}"
    for h in col_headers:
        header += f"  {h:>{COL_W}}"
    print(header)
    print(f"  {'-' * (name_w + 2 + SEED_LABEL_W + (COL_W + 2) * len(metrics))}")

    for exp_name, seeds in run_data:
        if not seeds:
            print(f"  {exp_name:<{name_w}}  (not found)")
            print()
            continue

        for i, sm in enumerate(seeds):
            name_col = exp_name if i == 0 else ""
            row = f"  {name_col:<{name_w}}  {'seed ' + str(i):<{SEED_LABEL_W}}"
            for m, acc in zip(metrics, is_acc):
                row += f"  {fmt_val(sm.get(m), acc):>{COL_W}}"
            print(row)

        row = f"  {'': <{name_w}}  {'mean±std':<{SEED_LABEL_W}}"
        for m, acc in zip(metrics, is_acc):
            vals = [sm[m] for sm in seeds if sm.get(m) is not None]
            row += f"  {fmt_mean_std(vals, acc):>{COL_W}}"
        print(row)
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Per-seed metric comparison for eval YAML groups")
    parser.add_argument(
        "eval_yaml",
        nargs="?",
        help="Path to eval YAML (default: prompt to pick from src/train/configs/eval/)",
    )
    parser.add_argument(
        "--group",
        metavar="NAME",
        help="Print only groups whose name contains NAME (case-insensitive substring)",
    )
    args = parser.parse_args()

    if args.eval_yaml:
        eval_yaml = Path(args.eval_yaml)
        if not eval_yaml.is_absolute():
            eval_yaml = Path.cwd() / eval_yaml
    else:
        yamls = sorted(
            p for p in EVAL_CONFIG_DIR.glob("*.yaml") if p.name != "translations.yaml"
        )
        if not yamls:
            print(f"No eval YAML files found in {EVAL_CONFIG_DIR}", file=sys.stderr)
            sys.exit(1)
        print("Available eval configs:")
        for i, p in enumerate(yamls):
            print(f"  [{i}] {p.name}")
        idx = int(input("Select config index: "))
        eval_yaml = yamls[idx]

    if not eval_yaml.exists():
        print(f"File not found: {eval_yaml}", file=sys.stderr)
        sys.exit(1)

    with eval_yaml.open() as f:
        cfg = yaml.safe_load(f)

    run_name = eval_yaml.stem
    primary_metrics: list[str] = cfg["primary_metrics"]
    groups: list[dict] = cfg["groups"]

    all_refs = [ref for group in groups for ref in group["runs"]]
    config_names = {parse_run_ref(ref, run_name)[0] for ref in all_refs}
    run_configs = load_run_configs(config_names)
    run_experiments = resolve_experiments(groups, run_configs, run_name)
    translations = load_translations()

    filter_name = args.group.lower() if args.group else None
    for group in groups:
        if filter_name and filter_name not in group["name"].lower():
            continue
        print_group(group, primary_metrics, run_experiments, translations, run_name)


if __name__ == "__main__":
    main()
