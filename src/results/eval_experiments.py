#!/usr/bin/env python3
"""Evaluate experiment groups and generate per-group HTML result tables."""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from great_tables import GT, loc, style

REPO_ROOT = Path(__file__).parent.parent.parent
LOG_RUNS = REPO_ROOT / "log" / "runs"
EVAL_CONFIG_DIR = REPO_ROOT / "src" / "train" / "configs" / "eval"
RUN_CONFIG_DIR = REPO_ROOT / "src" / "train" / "configs" / "run"
TRANSLATIONS_FILE = EVAL_CONFIG_DIR / "translations.yaml"
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
        return "—"
    mean = np.mean(values)
    std = np.std(values)
    sep = r"$\pm$" if latex else "±"
    if format_percent:
        return f"{mean*100:.2f} {sep} {std*100:.2f}"
    return f"{mean:.4f} {sep} {std:.4f}"


def format_experiment_name(
    exp_key: str,
    run_experiments: dict,
    translations: dict,
    latex: bool = False,
) -> str:
    exp_def = run_experiments.get(exp_key)
    if exp_def is None:
        return exp_key.replace("_", " ").title()

    symbol_key = "latex" if latex else "plain"
    model_key = exp_def.get("model", exp_key)
    base_name = translations["models"].get(model_key, model_key.replace("_", " ").title())

    overrides: dict = exp_def.get("overrides") or {}
    parts = [base_name]
    for param_key, value in overrides.items():
        print(param_key, value)
        param_trans = translations["params"].get(param_key)
        label = param_trans[symbol_key] if param_trans else param_key.split(".")[-1]
        no_value = param_trans.get("no_value", False) if param_trans else False
        if isinstance(value, bool):
            if not value:
                parts.append(f"no {label}")
            else:
                parts.append(label)
        elif no_value:
            parts.append(label)
        else:
            val_str = f"{value:g}" if isinstance(value, float) else str(value)
            parts.append(f"{label}$={val_str}$" if latex else f"{label}={val_str}")

    return ", ".join(parts)


def format_metric_name(metric: str, remove_task_prefix: bool=True, remove_acc: bool=False, remove_od: bool=False) -> str:
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
    metric = metric.replace("-", " ").title().replace("Od", "OD").replace("Id", "ID").replace("Lr", "LR").replace("Hr", "HR")

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
                " & ".join(cols) + " \\\\",
                "\\midrule",
            ]
            for i, (_, row) in enumerate(df[[exp_col] + chunk].iterrows()):
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
            gt = GT(chunk_df).tab_header(title=title)
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
) -> bool:
    """Build and write a table for one group. Returns False if skipped."""
    metrics = primary_metrics + group.get("additional_metrics", [])
    exp_keys: list[str] = group["runs"]

    rows: list[dict] = []
    for key in exp_keys:
        run_dir = find_run_dir(key)
        metric_values = load_test_metrics(run_dir, metrics) if run_dir else {m: [] for m in metrics}
        row: dict = {"Experiment": format_experiment_name(key, run_experiments, translations, latex=latex)}
        for m in metrics:
            row[format_metric_name(m)] = format_cell(metric_values[m], format_percent=m.endswith("acc"), latex=latex)
        rows.append(row)

    df = pd.DataFrame(rows)
    metric_cols = [format_metric_name(m) for m in metrics]

    if df[metric_cols].eq("").all().all():
        return False

    col_directions = {format_metric_name(m): metric_directions.get(m, "max") for m in metrics}
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
) -> None:
    """Build a table ranking all unique experiments by the summary metrics."""
    seen: set[str] = set()
    exp_keys: list[str] = []
    for group in groups:
        for key in group["runs"]:
            if key not in seen:
                seen.add(key)
                exp_keys.append(key)

    cols = [format_metric_name(m) for m in summary_metrics]
    rows: list[dict] = []
    for key in exp_keys:
        run_dir = find_run_dir(key)
        all_values = load_test_metrics(run_dir, summary_metrics) if run_dir else {m: [] for m in summary_metrics}
        row: dict = {"Experiment": format_experiment_name(key, run_experiments, translations, latex=latex)}
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
    col_directions = {format_metric_name(m): metric_directions.get(m, "max") for m in summary_metrics}
    write_table(df, "All experiments — summary", output, latex, col_directions)


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
        yamls = sorted(EVAL_CONFIG_DIR.glob("*.yaml"))
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
    run_yaml = RUN_CONFIG_DIR / f"{run_name}.yaml"
    run_experiments: dict = {}
    if run_yaml.exists():
        with run_yaml.open() as f:
            run_cfg = yaml.safe_load(f)
        run_experiments = run_cfg.get("experiments", {})
    else:
        print(f"Warning: no run config found at {run_yaml}", file=sys.stderr)

    translations: dict = {"models": {}, "params": {}}
    if TRANSLATIONS_FILE.exists():
        with TRANSLATIONS_FILE.open() as f:
            translations = yaml.safe_load(f)

    primary_metrics: list[str] = cfg["primary_metrics"]
    summary_metrics: list[str] = cfg.get("summary_metrics", [])
    groups: list[dict] = cfg["groups"]
    metric_directions: dict[str, str] = cfg.get("metric_directions", {})

    latex: bool = args.latex
    ext = ".tex" if latex else ".html"

    output_dir = LATEX_OUTPUT_DIR / run_name if latex else REPO_ROOT / "results" / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    for group in groups:
        safe_name = group["name"].lower().replace(" ", "_").replace("-", "_")
        output = output_dir / f"{safe_name}{ext}"
        written = build_group_table(group, primary_metrics, output, latex, run_experiments, translations, metric_directions)
        if written:
            print(f"  wrote {output}")
        else:
            print(f"  skipped {group['name']} (no finished runs)")

    if summary_metrics:
        output = output_dir / f"summary{ext}"
        build_summary_table(groups, summary_metrics, output, latex, run_experiments, translations, metric_directions)
        print(f"  wrote {output}")


if __name__ == "__main__":
    main()
