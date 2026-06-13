#!/usr/bin/env python3
"""Evaluate experiment groups and generate per-group HTML result tables."""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from great_tables import GT, loc, style

from results.utils import (
    EVAL_CONFIG_DIR,
    REPO_ROOT,
    TRANSLATIONS_FILE,
    find_run_dir,
    format_experiment_name,
    load_run_configs,
    load_seed_test_metrics,
    parse_run_ref,
    resolve_experiments,
)

THESIS_ROOT = REPO_ROOT.parent / "thesis"
LATEX_OUTPUT_DIR = THESIS_ROOT / "results"


def load_test_metrics(run_dir: Path, metrics: list[str]) -> dict[str, list[float]]:
    """Load final test metrics from all seeds, preferring metrics_rerun.csv."""
    results: dict[str, list[float]] = {m: [] for m in metrics}

    for seed_dir in sorted(run_dir.glob("run*")):
        seed_metrics = load_seed_test_metrics(seed_dir)
        if not seed_metrics:
            continue
        for metric in metrics:
            val = seed_metrics.get(metric)
            if val is not None:
                results[metric].append(float(val))

    return results


def _load_param_count(run_dir: Path) -> int | None:
    """Load total parameter count, caching to model_info.json."""
    info_path = run_dir / "model_info.json"
    if info_path.exists():
        with info_path.open() as f:
            return json.load(f).get("param_count")

    ckpt_path = run_dir / "checkpoints" / "run0" / "last.ckpt"
    if not ckpt_path.exists():
        return None

    import torch
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("state_dict", {})
    param_count = sum(v.numel() for v in state_dict.values())

    with info_path.open("w") as f:
        json.dump({"param_count": param_count}, f)

    return param_count


def load_all_metrics(run_dir: Path | None, metrics: list[str]) -> dict[str, list[float]]:
    """Load metrics from CSV and param_count from checkpoint/cache."""
    csv_metrics = [m for m in metrics if m != "param_count"]
    results = load_test_metrics(run_dir, csv_metrics) if run_dir and csv_metrics else {m: [] for m in csv_metrics}
    if "param_count" in metrics:
        count = _load_param_count(run_dir) if run_dir else None
        results["param_count"] = [float(count)] if count is not None else []
    return results


def format_cell(values: list[float], format_percent: bool = False, format_count: bool = False, latex: bool = False) -> str:
    if not values:
        return "—"
    if format_count:
        count = int(values[0])
        if count >= 1_000_000:
            return f"{count / 1e6:.1f}M"
        if count >= 1_000:
            return f"{count / 1e3:.1f}K"
        return str(count)
    mean = np.mean(values)
    std = np.std(values)
    sep = r"$\pm$" if latex else "±"
    if format_percent:
        return f"{mean*100:.2f} {sep} {std*100:.2f}"
    return f"{mean:.4f} {sep} {std:.4f}"


def format_metric_name(metric: str, remove_task_prefix: bool=True, remove_acc: bool=False, remove_od: bool=False) -> str:
    if metric == "param_count":
        return "Parameters"
    if metric.startswith("test/test-"):
        metric = metric.removeprefix("test/test-")
    elif metric.startswith("val/val-"):
        metric = metric.removeprefix("val/val-")

    if remove_task_prefix:
        metric = metric.replace("-task", "")
    if remove_acc:
        metric = metric.replace("-acc", "")
    if remove_od:
        metric = metric.replace("od", "")

    if "region" in metric:
        metric = re.sub(r"-region(-.*?)", r"\1 ", metric)
    metric = (
        metric.replace("-", " ")
              .title()
              .replace("Od", "OOD")
              .replace("Id", "ID")
              .replace("Lr", "LR")
              .replace("Hr", "HR")
              .replace("Worst Group Acc", "WRA")
              .replace("Acc", "Overall Acc.")
              )

    return metric


METRIC_CHUNK_SIZE = 4


def _parse_mean(cell: str) -> float | None:
    try:
        return float(str(cell).split()[0])
    except (ValueError, IndexError):
        return None


def _best_row_per_col(df: pd.DataFrame, metric_cols: list[str], directions: dict[str, str]) -> dict[str, int]:
    best: dict[str, int] = {}
    for col in metric_cols:
        minimize = directions.get(col, "max") == "min"
        sentinel = float("inf") if minimize else -float("inf")
        best_val, best_idx = sentinel, None
        for i, cell in enumerate(df[col]):
            val = _parse_mean(cell)
            if val is None:
                continue
            if (minimize and val < best_val) or (not minimize and val > best_val):
                best_val, best_idx = val, i
        if best_idx is not None:
            best[col] = best_idx
    return best


def write_table(df: pd.DataFrame, title: str, output: Path, latex: bool, col_directions: dict[str, str]) -> None:
    exp_col = df.columns[0]
    metric_cols = list(df.columns[1:])

    if len(metric_cols) > METRIC_CHUNK_SIZE:
        mid = (len(metric_cols) + 1) // 2
        chunks = [metric_cols[:mid], metric_cols[mid:]]
    else:
        chunks = [metric_cols]

    best = _best_row_per_col(df, metric_cols, col_directions)

    if latex:
        all_lines = [f"% {title}"]
        for chunk in chunks:
            cols = [exp_col] + chunk
            col_fmt = "l" + "r" * len(chunk)
            all_lines += [
                f"\\begin{{tabular}}{{{col_fmt}}}",
                "\\toprule",
                " & ".join([""] + chunk) + " \\\\",  # no header for experiment column
                "\\midrule",
            ]
            for i, (_, row) in enumerate(df[cols].iterrows()):
                cells = []
                for col, val in zip(cols, row):
                    s = str(val)
                    if col in best and best[col] == i:
                        s = f"\\textbf{{{s}}}"
                    cells.append(s)
                all_lines.append(" & ".join(cells) + " \\\\")
            all_lines += ["\\bottomrule", "\\end{tabular}", ""]
        output.write_text("\n".join(all_lines), encoding="utf-8")
    else:
        html_parts = []
        for chunk in chunks:
            chunk_df = df[[exp_col] + chunk]
            gt = GT(chunk_df).tab_header(title=title).cols_label(**{exp_col: ""})
            for col, row_idx in best.items():
                if col in chunk:
                    gt = gt.tab_style(
                        style=style.text(weight="bold"),
                        locations=loc.body(columns=col, rows=[row_idx]),
                    )
            html_parts.append(gt.as_raw_html())
        output.write_text("\n".join(html_parts), encoding="utf-8")


def build_group_table(
    group: dict,
    primary_metrics: list[str],
    output: Path,
    latex: bool,
    run_experiments: dict,
    translations: dict,
    metric_directions: dict[str, str],
    default_config: str,
) -> bool:
    """Build and write a table for one group. Returns False if skipped."""
    metrics = primary_metrics + group.get("additional_metrics", [])
    run_refs: list[str] = group["runs"]

    rows: list[dict] = []
    for ref in run_refs:
        _, exp_key = parse_run_ref(ref, default_config)
        run_dir = find_run_dir(exp_key)
        metric_values = load_all_metrics(run_dir, metrics)
        row: dict = {"Experiment": format_experiment_name(ref, run_experiments, translations, latex=latex)}
        for m in metrics:
            row[format_metric_name(m)] = format_cell(
                metric_values[m], format_percent=m.endswith("acc"), format_count=(m == "param_count"), latex=latex,
            )
        rows.append(row)

    df = pd.DataFrame(rows)
    metric_cols = [format_metric_name(m) for m in metrics]

    if df[metric_cols].eq("").all().all():
        return False

    col_directions = {format_metric_name(m): metric_directions.get(m, "max") for m in metrics if m != "param_count"}
    write_table(df, group["name"], output, latex, col_directions)
    return True


def build_summary_table(
    groups: list[dict],
    summary_metrics: list[str],
    output: Path,
    latex: bool,
    run_experiments: dict,
    translations: dict,
    metric_directions: dict[str, str],
    default_config: str,
) -> None:
    """Build a table ranking all unique experiments by the summary metrics."""
    seen: set[str] = set()
    run_refs: list[str] = []
    for group in groups:
        for ref in group["runs"]:
            if ref not in seen:
                seen.add(ref)
                run_refs.append(ref)

    cols = [format_metric_name(m) for m in summary_metrics]
    rows: list[dict] = []
    for ref in run_refs:
        _, exp_key = parse_run_ref(ref, default_config)
        run_dir = find_run_dir(exp_key)
        all_values = load_all_metrics(run_dir, summary_metrics)
        row: dict = {"Experiment": format_experiment_name(ref, run_experiments, translations, latex=latex)}
        for m, col in zip(summary_metrics, cols):
            row[col] = format_cell(
                all_values[m], format_percent=m.endswith("acc"), format_count=(m == "param_count"), latex=latex,
            )
        first_vals = all_values[summary_metrics[0]]
        row["_sort"] = float(np.mean(first_vals)) if first_vals else -1.0
        rows.append(row)

    df = (
        pd.DataFrame(rows)
        .sort_values("_sort", ascending=False)
        .drop(columns="_sort")
        .reset_index(drop=True)
    )
    col_directions = {format_metric_name(m): metric_directions.get(m, "max") for m in summary_metrics if m != "param_count"}
    write_table(df, "All experiments — summary", output, latex, col_directions)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate experiment groups")
    parser.add_argument(
        "eval_yaml",
        nargs="?",
        help="Path to eval YAML (default: prompt user to pick from src/train/configs/eval/)",
    )
    fmt = parser.add_mutually_exclusive_group()
    fmt.add_argument("--latex", action="store_true", help="Output booktabs .tex snippets instead of HTML")
    fmt.add_argument("--both", action="store_true", help="Output both HTML and LaTeX tables")
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
    summary_metrics: list[str] = cfg.get("summary_metrics", [])
    groups: list[dict] = cfg["groups"]
    metric_directions: dict[str, str] = cfg.get("metric_directions", {})

    all_refs = [ref for group in groups for ref in group["runs"]]
    config_names = {parse_run_ref(ref, run_name)[0] for ref in all_refs}
    run_configs = load_run_configs(config_names)
    run_experiments = resolve_experiments(groups, run_configs, run_name)

    translations: dict = {"models": {}, "params": {}}
    if TRANSLATIONS_FILE.exists():
        with TRANSLATIONS_FILE.open() as f:
            translations = yaml.safe_load(f)

    if args.both:
        formats = ["html", "latex"]
    elif args.latex:
        formats = ["latex"]
    else:
        formats = ["html"]

    for fmt in formats:
        latex = fmt == "latex"
        ext = ".tex" if latex else ".html"
        output_dir = LATEX_OUTPUT_DIR / run_name if latex else REPO_ROOT / "results" / run_name
        output_dir.mkdir(parents=True, exist_ok=True)

        for group in groups:
            safe_name = group["name"].lower().replace(" ", "_").replace("-", "_")
            output = output_dir / f"{safe_name}{ext}"
            written = build_group_table(group, primary_metrics, output, latex, run_experiments, translations, metric_directions, run_name)
            if written:
                print(f"  wrote {output}")
            else:
                print(f"  skipped {group['name']} (no finished runs)")

        if summary_metrics:
            output = output_dir / f"summary{ext}"
            build_summary_table(groups, summary_metrics, output, latex, run_experiments, translations, metric_directions, run_name)
            print(f"  wrote {output}")


if __name__ == "__main__":
    main()
