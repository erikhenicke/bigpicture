#!/usr/bin/env python3
"""Line plots of test metrics vs. LR spatial extent for the spatial-extent study.

Reads an eval YAML (e.g. ``spatial_extent.yaml``) whose runs vary only in their
``data.lr_crop_km`` override. Each (group, model) pair becomes one line series
— e.g. the "Spatial Extent" group's FiLM and D3G runs are split into separate
FiLM/D3G lines rather than interleaved — plotted against the crop extent (km) on
the x-axis, with a shaded band for the across-seed std. One figure is written
per primary metric; the OOD WRA figure is the headline one.

``style_for`` assigns a consistent color/marker/linestyle per model series;
``baseline_mean_std`` reads the HR-only baseline's stats. ``plot_stacked_accuracy``
and ``plot_single_metric`` draw the actual figures (one stacked-subplot figure
for the three ``COMBINED_METRICS``, one figure per remaining metric);
``plot_legend`` draws the single legend shared by all of them. ``crop_km_for_ref``
and ``model_for_ref`` pull the sweep x-value and model key out of a resolved
experiment definition. ``main`` loads the eval YAML, builds one series per
(group, model) pair, and calls the plotting functions.
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.lines import Line2D

from results.eval_metrics import format_metric_name, load_test_metrics
from results.utils import (
    EVAL_CONFIG_DIR,
    REPO_ROOT,
    find_run_dir,
    load_run_configs,
    load_translations,
    parse_run_ref,
    resolve_experiments,
)

CROP_KM_KEY = "data.lr_crop_km"
# Display label for the HR-only baseline reference line.
BASELINE_LABEL = "HR only"
# Mirror every figure into the thesis repo's images dir, under a per-run
# subfolder matching the repo's ``figures/<run_name>/`` layout.
THESIS_IMAGES_DIR = Path.home() / "git" / "thesis" / "images"

# Accuracy metrics drawn together on a single figure, stacked as subplots.
COMBINED_METRICS = {
    "test/test-id-task-acc",
    "test/test-od-task-acc",
    "test/test-od-worst-group-task-acc",
}

# One consistent color/marker/linestyle per model series (by translated display
# name), reused across every figure so "FiLM" and "D3G" read the same way
# everywhere. Falls back to FALLBACK_STYLES, cycled in first-seen order, for any
# series this sweep doesn't already know about.
SERIES_STYLES = {
    "FiLM": dict(color="#115fb0", marker="o", linestyle="-"),
    "D3G": dict(color="#cf440a", marker="s", linestyle="-."),
}
FALLBACK_STYLES = [dict(color="#5a3d99", marker="^", linestyle=":")]
_fallback_assigned: dict[str, dict] = {}


def style_for(name: str) -> dict:
    """Consistent color/marker/linestyle for a series name, assigning unknown
    names a fallback style (cycled, cached) so new series don't collide.

    Args:
        name (str): Series display name (e.g. "FiLM", "D3G").

    Returns:
        dict: Style kwargs with keys ``"color"``, ``"marker"``, ``"linestyle"``.
    """
    if name in SERIES_STYLES:
        return SERIES_STYLES[name]
    if name not in _fallback_assigned:
        idx = len(_fallback_assigned) % len(FALLBACK_STYLES)
        _fallback_assigned[name] = FALLBACK_STYLES[idx]
    return _fallback_assigned[name]


def baseline_mean_std(ref: str, run_name: str, metric: str) -> tuple[float, float] | None:
    """Mean and std (in %) of `metric` across seeds for the baseline run, or None.

    Args:
        ref (str): Baseline run reference (``"config@exp_key"`` or ``"exp_key"``).
        run_name (str): Default run-config name for refs without an explicit
            ``config@`` prefix.
        metric (str): Metric key to look up.

    Returns:
        tuple[float, float] | None: ``(mean, std)`` in percent across seeds, or
            None if the run directory or metric values are missing.
    """
    _, exp_key = parse_run_ref(ref, run_name)
    run_dir = find_run_dir(exp_key)
    vals = load_test_metrics(run_dir, [metric])[metric] if run_dir else []
    if not vals:
        return None
    return float(np.mean(vals)) * 100.0, float(np.std(vals)) * 100.0


def plot_stacked_accuracy(combined_metrics, group_series, baseline_ref, run_name,
                          translations, write_figure, label_size, tick_size) -> None:
    """Accuracy metrics as distinct subplots stacked vertically, sharing one x-axis.
    No per-subplot legend: the series/baseline encoding is identical across this
    figure and every single-metric figure, so one legend (``plot_legend``) covers
    all of them.

    Args:
        combined_metrics (list[str]): Metric keys to draw, one subplot each, in order.
        group_series (list[tuple[str, dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]]]):
            ``(series_name, {metric: (x, mean, std)})`` entries, as built in ``main``.
        baseline_ref (str | None): Baseline run reference, or None to skip the
            baseline line.
        run_name (str): Default run-config name for the baseline ref.
        translations (dict): Display-name translations (see
            ``results.utils.load_translations``), used for the y-axis labels.
        write_figure (Callable[..., None]): Callback that saves and closes the
            figure (see ``main``'s local ``write_figure``).
        label_size (int): Font size for axis labels.
        tick_size (int): Font size for tick labels.
    """
    fig, axes = plt.subplots(
        len(combined_metrics), 1, figsize=(6.0, 1.8 * len(combined_metrics)),
        sharex=True, squeeze=False,
    )
    axes = axes[:, 0]
    for ax, metric in zip(axes, combined_metrics):
        contributors = [(name, s[metric]) for name, s in group_series if metric in s]
        for name, (x, mean, std) in contributors:
            style = style_for(name)
            ax.plot(x, mean, marker=style["marker"], markersize=5, linewidth=1.6,
                    color=style["color"], linestyle=style["linestyle"], label=name)
            ax.fill_between(x, mean - std, mean + std, color=style["color"], alpha=0.15, linewidth=0)

        # HR-only baseline for this metric, dashed and shaded, mirroring decision_plots.py.
        if baseline_ref:
            base_ms = baseline_mean_std(baseline_ref, run_name, metric)
            if base_ms:
                base_mean, base_std = base_ms
                ax.axhspan(base_mean - base_std, base_mean + base_std, color="gray", alpha=0.1, linewidth=0)
                ax.axhline(base_mean, color="gray", linestyle="--", linewidth=1.2)

        ax.set_ylabel(f"{format_metric_name(metric, translations)} (%)", fontsize=label_size)
        ax.tick_params(axis="both", labelsize=tick_size)
        ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.6)

    axes[-1].set_xlabel(r"Spatial Extent of $c$ (km)", fontsize=label_size)
    fig.tight_layout()
    write_figure(fig, "task-acc")


def plot_single_metric(metric, group_series, baseline_ref, run_name, translations,
                       write_figure, label_size, tick_size) -> None:
    """Single line figure for one metric (e.g. ECE) across the spatial-extent sweep.
    No legend: the series/baseline encoding is identical across this figure and
    the stacked-accuracy figure, so one legend (``plot_legend``) covers both.

    Args:
        metric (str): Metric key to plot.
        group_series (list[tuple[str, dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]]]):
            ``(series_name, {metric: (x, mean, std)})`` entries, as built in ``main``.
        baseline_ref (str | None): Baseline run reference, or None to skip the
            baseline line.
        run_name (str): Default run-config name for the baseline ref.
        translations (dict): Display-name translations, used for axis labels/title.
        write_figure (Callable[..., None]): Callback that saves and closes the figure.
        label_size (int): Font size for axis labels.
        tick_size (int): Font size for tick labels.
    """
    contributors = [(name, s[metric]) for name, s in group_series if metric in s]
    if not contributors:
        print(f"Warning: no data for {metric}, skipping figure", file=sys.stderr)
        return

    fig, ax = plt.subplots(figsize=(6.0, 4.5))
    for name, (x, mean, std) in contributors:
        style = style_for(name)
        line = ax.plot(x, mean, marker=style["marker"], markersize=5, linewidth=1.6,
                       color=style["color"], linestyle=style["linestyle"], label=name)[0]
        ax.fill_between(x, mean - std, mean + std, color=line.get_color(), alpha=0.15, linewidth=0)

    # HR-only baseline as a dashed horizontal line with a shaded std band
    # spanning the full width, mirroring decision_plots.py.
    if baseline_ref:
        base_ms = baseline_mean_std(baseline_ref, run_name, metric)
        if base_ms:
            base_mean, base_std = base_ms
            ax.axhspan(base_mean - base_std, base_mean + base_std, color="gray", alpha=0.1, linewidth=0)
            ax.axhline(base_mean, color="gray", linestyle="--", linewidth=1.2)

    ax.set_xlabel(r"Spatial Extent of $c$ (km)", fontsize=label_size)
    ax.set_ylabel(f"{format_metric_name(metric, translations)} (%)", fontsize=label_size)
    ax.set_title(f"Spatial Extent — {format_metric_name(metric, translations)}")
    ax.tick_params(axis="both", labelsize=tick_size)
    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.6)

    fig.tight_layout()
    safe_metric = metric.rsplit("/", 1)[-1]
    write_figure(fig, safe_metric)


def plot_legend(group_series, baseline_ref, write_figure, legend_size) -> None:
    """Standalone legend for the spatial-extent line plots: one entry per model
    series (FiLM, D3G, ...) plus the HR-only baseline, laid out in a single
    horizontal row -- identical across the stacked-accuracy figure and every
    single-metric figure, so one legend covers all of them.

    Args:
        group_series (list[tuple[str, dict]]): ``(series_name, ...)`` entries;
            only the names are used, in first-seen order.
        baseline_ref (str | None): Baseline run reference; if set, adds a dashed
            gray "HR only" legend entry.
        write_figure (Callable[..., None]): Callback that saves and closes the figure.
        legend_size (int): Font size for the legend entries.
    """
    seen_labels = list(dict.fromkeys(name for name, _ in group_series))
    handles = [
        Line2D([0], [0], marker=style_for(name)["marker"], markersize=5, linewidth=1.6,
               color=style_for(name)["color"], linestyle=style_for(name)["linestyle"], label=name)
        for name in seen_labels
    ]
    if baseline_ref:
        handles.append(Line2D([0], [0], color="gray", linestyle="--", label=BASELINE_LABEL))
    if not handles:
        return

    fig = plt.figure(figsize=(6.0, 0.6))
    fig.legend(handles=handles, loc="center", ncol=len(handles), fontsize=legend_size, frameon=False)
    write_figure(fig, "legend", bbox_inches="tight")


def crop_km_for_ref(ref: str, run_experiments: dict) -> float | None:
    """Pull the ``data.lr_crop_km`` override for a run ref, or ``None`` if absent.

    Args:
        ref (str): Run reference.
        run_experiments (dict): Run-ref -> experiment-definition map (see
            ``results.utils.resolve_experiments``).

    Returns:
        float | None: The ``data.lr_crop_km`` override value, or None if the ref
            is unresolved or has no such override.
    """
    exp_def = run_experiments.get(ref)
    if exp_def is None:
        return None
    overrides = exp_def.get("overrides") or {}
    value = overrides.get(CROP_KM_KEY)
    return float(value) if value is not None else None


def model_for_ref(ref: str, run_experiments: dict) -> str | None:
    """Pull the ``model`` key for a run ref, or ``None`` if absent.

    Args:
        ref (str): Run reference.
        run_experiments (dict): Run-ref -> experiment-definition map (see
            ``results.utils.resolve_experiments``).

    Returns:
        str | None: The experiment's ``model`` key, or None if the ref is unresolved.
    """
    exp_def = run_experiments.get(ref)
    return exp_def.get("model") if exp_def else None


def main() -> None:
    """CLI entry point: build per-(group, model) line series and render every figure.

    Parses the eval YAML path (default ``spatial_extent.yaml``), resolves every
    referenced run's experiment definition, groups runs into one series per
    (eval-group, model) pair keyed by their ``data.lr_crop_km`` value (dropping
    sub-series with fewer than two distinct crop-km values, since those aren't a
    sweep), then draws the stacked-accuracy figure, one figure per remaining
    primary metric, and the shared legend.
    """
    parser = argparse.ArgumentParser(description="Spatial-extent line plots")
    parser.add_argument(
        "eval_yaml",
        nargs="?",
        default=str(EVAL_CONFIG_DIR / "spatial_extent.yaml"),
        help="Path to eval YAML (default: src/train/configs/eval/spatial_extent.yaml)",
    )
    args = parser.parse_args()

    eval_yaml = Path(args.eval_yaml)
    if not eval_yaml.is_absolute():
        eval_yaml = Path.cwd() / eval_yaml
    if not eval_yaml.exists():
        print(f"File not found: {eval_yaml}", file=sys.stderr)
        sys.exit(1)

    with eval_yaml.open() as f:
        cfg = yaml.safe_load(f)

    run_name = eval_yaml.stem
    primary_metrics: list[str] = cfg["primary_metrics"]
    groups: list[dict] = cfg["groups"]
    baseline_ref: str | None = cfg.get("baseline")

    translations = load_translations()

    all_refs = [ref for group in groups for ref in group["runs"]]
    config_names = {parse_run_ref(ref, run_name)[0] for ref in all_refs}
    run_configs = load_run_configs(config_names)
    run_experiments = resolve_experiments(groups, run_configs, run_name)

    figures_dir = REPO_ROOT / "figures" / run_name
    figures_dir.mkdir(parents=True, exist_ok=True)

    thesis_dir = THESIS_IMAGES_DIR / run_name if THESIS_IMAGES_DIR.parent.exists() else None
    if thesis_dir is not None:
        thesis_dir.mkdir(parents=True, exist_ok=True)
    else:
        print(f"Warning: {THESIS_IMAGES_DIR.parent} not found, skipping thesis copy", file=sys.stderr)

    # A group's runs may span more than one model (e.g. "Spatial Extent" holds
    # both FiLM and D3G sweeps) — split into one series per (group, model) pair
    # so each model gets its own line instead of interleaving their points.
    # Sub-series that don't actually vary in extent (e.g. "Channel Ablation",
    # which fixes lr_crop_km and only varies band count) aren't a sweep and are
    # dropped rather than drawn as a spurious near-vertical segment.
    model_names: dict[str, str] = translations.get("models", {})
    group_series: list[tuple[str, dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]]] = []
    for group in groups:
        per_model_points: dict[str | None, dict[str, list[tuple[float, float, float]]]] = {}
        for ref in group["runs"]:
            km = crop_km_for_ref(ref, run_experiments)
            if km is None:
                print(f"Warning: no {CROP_KM_KEY} for '{ref}', skipping", file=sys.stderr)
                continue
            _, exp_key = parse_run_ref(ref, run_name)
            run_dir = find_run_dir(exp_key)
            if run_dir is None:
                print(f"Warning: no run dir for '{ref}', skipping", file=sys.stderr)
                continue
            model_key = model_for_ref(ref, run_experiments)
            per_metric_points = per_model_points.setdefault(model_key, {m: [] for m in primary_metrics})
            metric_values = load_test_metrics(run_dir, primary_metrics)
            for metric in primary_metrics:
                vals = metric_values[metric]
                if not vals:
                    continue
                per_metric_points[metric].append(
                    (km, float(np.mean(vals)) * 100.0, float(np.std(vals)) * 100.0)
                )

        for model_key, per_metric_points in per_model_points.items():
            label = model_names.get(model_key, model_key) if model_key else group["name"]
            unique_kms = {p[0] for points in per_metric_points.values() for p in points}
            if len(unique_kms) < 2:
                print(
                    f"Warning: skipping '{group['name']}' / '{label}' — fewer than 2 distinct "
                    f"{CROP_KM_KEY} values, not a sweep",
                    file=sys.stderr,
                )
                continue
            series: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
            for metric, points in per_metric_points.items():
                if not points:
                    continue
                points.sort(key=lambda p: p[0])
                x, mean, std = (np.array(c) for c in zip(*points))
                series[metric] = (x, mean, std)
            group_series.append((label, series))

    def write_figure(fig, stem: str, **savefig_kwargs) -> None:
        out_path = figures_dir / f"{stem}.svg"
        fig.savefig(out_path, format="svg", **savefig_kwargs)
        print(f"  wrote {out_path}")
        if thesis_dir is not None:
            thesis_path = thesis_dir / f"{stem}.svg"
            fig.savefig(thesis_path, format="svg", **savefig_kwargs)
            print(f"  wrote {thesis_path}")
        plt.close(fig)

    # Font sizes per figure type — tune these to taste.
    stacked_sizes = dict(label_size=10, tick_size=8)
    single_sizes = dict(label_size=13, tick_size=13)
    legend_size = 8

    # Accuracy metrics are drawn as distinct subplots stacked vertically, sharing
    # one x-axis; every other metric (e.g. ECE) keeps its own figure below.
    combined_metrics = [
        m for m in primary_metrics
        if m in COMBINED_METRICS and any(m in s for _, s in group_series)
    ]
    if combined_metrics:
        plot_stacked_accuracy(combined_metrics, group_series, baseline_ref, run_name,
                              translations, write_figure, **stacked_sizes)

    for metric in primary_metrics:
        if metric in COMBINED_METRICS:
            continue
        plot_single_metric(metric, group_series, baseline_ref, run_name, translations,
                           write_figure, **single_sizes)

    # One legend shared by the stacked-accuracy figure and every single-metric figure.
    plot_legend(group_series, baseline_ref, write_figure, legend_size)


if __name__ == "__main__":
    main()
