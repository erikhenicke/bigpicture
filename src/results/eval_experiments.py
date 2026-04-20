#!/usr/bin/env python3
"""Evaluate experiment groups and generate per-group HTML result tables."""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from great_tables import GT

REPO_ROOT = Path(__file__).parent.parent.parent
LOG_RUNS = REPO_ROOT / "log" / "runs"
RUN_CONFIG_DIR = REPO_ROOT / "src" / "train" / "configs" / "eval"
THESIS_ROOT = REPO_ROOT.parent / "thesis"
LATEX_OUTPUT_DIR = THESIS_ROOT / "results"


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
    # Take the most recent by date, then by run name 
    candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return candidates[0][2]


def load_test_metrics(run_dir: Path, metrics: list[str]) -> dict[str, list[float]]:
    """Load final-epoch test metrics from all seeds in a run directory."""
    results: dict[str, list[float]] = {m: [] for m in metrics}

    for seed_dir in sorted(run_dir.glob("run*")):
        csv_path = seed_dir / "version_0" / "metrics.csv"
        if not csv_path.exists():
            continue

        df = pd.read_csv(csv_path)
        test_cols = [c for c in df.columns if c.startswith("test/")]
        if not test_cols:
            continue

        test_rows = df.dropna(subset=[test_cols[0]])
        if test_rows.empty:
            continue

        row = test_rows.iloc[-1]
        for metric in metrics:
            val = row.get(metric)
            if val is not None and pd.notna(val):
                results[metric].append(float(val))

    return results


def format_cell(values: list[float], format_percent: bool, latex: bool = False) -> str:
    if not values:
        return ""
    mean = np.mean(values)
    std = np.std(values)
    sep = r"$\pm$" if latex else "±"
    if format_percent:
        return f"{mean*100:.2f} {sep} {std*100:.2f}"
    return f"{mean:.4f} {sep} {std:.4f}"


def short_metric_name(metric: str) -> str:
    if metric.startswith("test/"):
        return metric.removeprefix("test/")
    elif metric.startswith("val/"):
        return metric.removeprefix("val/")
    return metric


def write_table(df: pd.DataFrame, title: str, output: Path, latex: bool) -> None:
    if latex:
        cols = list(df.columns)
        col_fmt = "l" + "r" * (len(cols) - 1)
        lines = [
            f"% {title}",
            f"\\begin{{tabular}}{{{col_fmt}}}",
            "\\toprule",
            " & ".join(cols) + " \\\\",
            "\\midrule",
        ]
        for _, row in df.iterrows():
            lines.append(" & ".join(str(v) for v in row) + " \\\\")
        lines += ["\\bottomrule", "\\end{tabular}"]
        output.write_text("\n".join(lines), encoding="utf-8")
    else:
        gt = GT(df).tab_header(title=title).cols_label(experiment="Experiment")
        output.write_text(gt.as_raw_html(), encoding="utf-8")


def build_group_table(
    group: dict,
    primary_metrics: list[str],
    output: Path,
    latex: bool,
) -> bool:
    """Build and write a table for one group. Returns False if skipped."""
    metrics = primary_metrics + group.get("additional_metrics", [])
    exp_keys: list[str] = group["runs"]

    rows: list[dict] = []
    for key in exp_keys:
        run_dir = find_run_dir(key)
        metric_values = load_test_metrics(run_dir, metrics) if run_dir else {m: [] for m in metrics}
        row: dict = {"experiment": f"train_{key}"}
        for m in metrics:
            row[short_metric_name(m)] = format_cell(metric_values[m], format_percent=m.endswith("acc"), latex=latex)
        rows.append(row)

    df = pd.DataFrame(rows)
    metric_cols = [short_metric_name(m) for m in metrics]

    if df[metric_cols].eq("").all().all():
        return False

    write_table(df, group["name"], output, latex)
    return True


def build_summary_table(groups: list[dict], summary_metrics: list[str], output: Path, latex: bool) -> None:
    """Build a table ranking all unique experiments by the summary metrics."""
    seen: set[str] = set()
    exp_keys: list[str] = []
    for group in groups:
        for key in group["runs"]:
            if key not in seen:
                seen.add(key)
                exp_keys.append(key)

    cols = [short_metric_name(m) for m in summary_metrics]
    rows: list[dict] = []
    for key in exp_keys:
        run_dir = find_run_dir(key)
        all_values = load_test_metrics(run_dir, summary_metrics) if run_dir else {m: [] for m in summary_metrics}
        row: dict = {"experiment": f"train_{key}"}
        for m, col in zip(summary_metrics, cols):
            row[col] = format_cell(all_values[m], format_percent=m.endswith("acc"), latex=latex)
        first_vals = all_values[summary_metrics[0]]
        row["_sort"] = float(np.mean(first_vals)) if first_vals else -1.0
        rows.append(row)

    df = (
        pd.DataFrame(rows)
        .sort_values("_sort", ascending=False)
        .drop(columns="_sort")
        .reset_index(drop=True)
    )
    write_table(df, "All experiments — summary", output, latex)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate experiment groups")
    parser.add_argument(
        "eval_yaml",
        nargs="?",
        help="Path to eval YAML (default: prompt user to pick from src/train/configs/eval/)",
    )
    parser.add_argument("--latex", action="store_true", help="Output booktabs .tex snippets instead of HTML")
    args = parser.parse_args()

    if args.eval_yaml:
        eval_yaml = Path(args.eval_yaml)
        if not eval_yaml.is_absolute():
            eval_yaml = Path.cwd() / eval_yaml
    else:
        yamls = sorted(RUN_CONFIG_DIR.glob("*.yaml"))
        if not yamls:
            print(f"No eval YAML files found in {RUN_CONFIG_DIR}", file=sys.stderr)
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

    primary_metrics: list[str] = cfg["primary_metrics"]
    summary_metrics: list[str] = cfg.get("summary_metrics", [])
    groups: list[dict] = cfg["groups"]

    latex: bool = args.latex
    ext = ".tex" if latex else ".html"
    run_name = eval_yaml.stem

    output_dir = LATEX_OUTPUT_DIR / run_name if latex else REPO_ROOT / "results" / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    for group in groups:
        safe_name = group["name"].lower().replace(" ", "_").replace("-", "_")
        output = output_dir / f"{safe_name}{ext}"
        written = build_group_table(group, primary_metrics, output, latex)
        if written:
            print(f"  wrote {output}")
        else:
            print(f"  skipped {group['name']} (no finished runs)")

    if summary_metrics:
        output = output_dir / f"summary{ext}"
        build_summary_table(groups, summary_metrics, output, latex)
        print(f"  wrote {output}")


if __name__ == "__main__":
    main()
