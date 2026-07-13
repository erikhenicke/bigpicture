#!/usr/bin/env python3
"""Region-wise OOD accuracy delta bars for the fusion-model comparison groups.

Reads the hardcoded ``GROUP_NAMES`` groups from an eval YAML (default
``feature_fusion.yaml``), each shaped as ``[baseline, *fusion_runs]`` (the
DenseNet-121 baseline followed by Concat/FiLM/D3G, with or without the domain
loss -- see ``Fusion Comparison`` vs. ``Fusion Comparison with Domain``), and
writes one standalone figure per group, each scaled to its own y-axis range.
(``YLIM_GROUPS`` lets sublists of groups share one range instead, computed
across every group first in ``main`` before any figure is drawn, for cases
where two figures are meant to sit side by side in the thesis -- unused by the
current ``GROUP_NAMES``, which renders three independent figures.) For each of
the five OOD test regions, draws one bar per fusion model showing its
``test-od-region-<r>-task-acc``
delta vs. the baseline (mean across seeds, in percentage points), with an
error bar (seed std, propagated through the difference). Two extra, uncolored
slots per region -- one on each side of the model bars -- show the baseline's
own seed std as a gray whisker centered at zero, and that same std is filled
in behind the model bars as a translucent gray band, so the baseline's noise
floor is visible without overlapping the model bars themselves. No per-panel
legend: each group's standalone legend (``plot_legend``) covers its one figure.

Styling helpers ``model_color`` and ``wrap_label`` pick a bar's color and wrap
its legend label. ``GroupLayout`` is a dataclass holding one group's resolved
runs and computed bar/whisker/fill geometry, built by ``build_group_layout``
(data only, no drawing). ``data_extent``/``group_extent`` derive a group's y-axis
range from its ``GroupLayout``; ``render_group_figure`` draws and writes the
figure (plus its legend, via ``plot_legend``), using a range possibly shared
with other groups. ``save_figure`` writes a figure to both the repo's and the
thesis repo's image directories. ``main`` resolves every group's layout first,
computes the shared y-ranges from ``YLIM_GROUPS``, then renders each figure.
"""

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

from results.eval_metrics import load_test_metrics
from results.utils import (
    EVAL_CONFIG_DIR,
    REPO_ROOT,
    find_run_dir,
    format_experiment_name,
    load_run_configs,
    load_translations,
    parse_run_ref,
    resolve_experiments,
)

# Mirror the figure into the thesis repo's images dir, matching every other
# plotting script in this package.
THESIS_IMAGES_DIR = Path.home() / "git" / "thesis" / "images"

# Groups in the eval YAML to render, each shaped [baseline, *fusion_runs].
GROUP_NAMES = ["Fusion Comparison", "Fusion Comparison with Domain", "Best Spatial Encodings"]

# A second layer of grouping on top of GROUP_NAMES: every group in the same
# sublist shares one y-axis range (computed across just those members), so they
# stay visually comparable side by side, while groups in different sublists are
# each scaled to their own data. Every group currently stands alone as its own
# figure in the thesis, so each gets its own extent; any group missing here
# falls back to its own extent too.
YLIM_GROUPS = [
    ["Fusion Comparison"],
    ["Fusion Comparison with Domain"],
    ["Best Spatial Encodings"],
]

REGIONS = ["europe", "asia", "americas", "africa", "oceania"]
REGION_LABELS = {"europe": "Europe", "asia": "Asia", "americas": "Americas", "africa": "Africa", "oceania": "Oceania"}

# Reuses decision_plots.py's RULE_COLORS (Min/GeoPrior/Sum) and its neutral
# MODE_LEGEND_COLOR gray, so this figure reads consistently with the decision-
# fusion plots elsewhere in the thesis.
MODEL_COLORS = {
    "Concat": "#3f93c9",
    "FiLM": "#115fb0",
    "D3G": "#cf440a",
    "SatCLIP": "#e8701a", 
    "DenseNet-121": "#dddddd"
}
BASELINE_COLOR = "#888888"


def model_color(label: str, fallback: str) -> str:
    """Color for a model, matched by name prefix: the MODEL_COLORS entry whose key
    the label starts with, so a spatial-encoding label like "FiLM, OM_gauss" still
    picks up FiLM's color. Longest matching key wins (so "Concat w. LE" would beat
    "Concat" if both were present); falls back to ``fallback`` when nothing matches.

    Args:
        label (str): Formatted model/experiment display label.
        fallback (str): Color to use if no ``MODEL_COLORS`` key is a prefix of ``label``.

    Returns:
        str: Hex color string.
    """
    for key in sorted(MODEL_COLORS, key=len, reverse=True):
        if label.startswith(key):
            return MODEL_COLORS[key]
    return fallback


def wrap_label(label: str, per_line: int = 2) -> str:
    """Regroup a comma-joined model label onto lines of ``per_line`` items each, so
    long legend entries (model + several spatial-encoding terms) stack vertically
    and pack tighter instead of running wide. Splits on every visual comma --
    including ones inside a ``$...$`` math span (e.g. the comma in
    ``$\\text{OM}_{gauss}, \\text{PE}_{freq}$``) -- re-closing and re-opening the
    span around each split so every resulting item stays valid mathtext.

    A lone leftover item (an odd tail of a single item) is indented four spaces so
    it reads as a continuation of the line above rather than a new entry. The gap
    between the stacked lines is tuned via the legend text ``linespacing`` (see
    ``LINE_SPACING`` in ``plot_legend``), not by padding with blank lines here.

    Args:
        label (str): Comma-joined model/experiment display label, possibly
            containing ``$...$`` mathtext spans.
        per_line (int): Number of comma-separated items to pack per line.

    Returns:
        str: The label re-wrapped onto newline-separated lines.
    """
    items: list[str] = []
    in_math = False
    buf: list[str] = []
    for ch in label:
        if ch == "$":
            in_math = not in_math
            buf.append(ch)
        elif ch == ",":
            if in_math:
                buf.append("$")          # close this item's math span
                items.append("".join(buf).strip())
                buf = ["$"]              # reopen it for the next item
            else:
                items.append("".join(buf).strip())
                buf = []
        else:
            buf.append(ch)
    items.append("".join(buf).strip())
    chunks = [items[i:i + per_line] for i in range(0, len(items), per_line)]
    lines = []
    for chunk in chunks:
        line = ", ".join(chunk)
        if len(chunk) == 1 and len(items) > 1:
            line = "    " + line
        lines.append(line)
    return "\n".join(lines)

# Distance between adjacent region-group centers (1.0 = default matplotlib
# spacing). The figure's physical width (see figsize below) does NOT scale
# with this, so shrinking it packs the same fixed-width figure with a smaller
# data range -- every bar renders bigger, not the whole plot smaller.
GROUP_SPACING = 0.9

# Total width spanned by one region's bar cluster (all model bars + the
# baseline whisker slot), in the same data-coordinate units as GROUP_SPACING.
# Capped to a fraction of GROUP_SPACING so tightening GROUP_SPACING can never
# make adjacent clusters overlap -- bars simply get as wide as they can while
# still leaving a small gap between groups.
BAR_CLUSTER_WIDTH = 0.7
MAX_GROUP_FILL = 0.95

# Font sizes, tune these to taste. REGION_LABEL_SIZE (the x-axis region names)
# is independent of TICK_SIZE (the y-axis tick numbers) -- see the tick_params
# calls below, which set each axis's label size separately so tuning one never
# touches the other.
LABEL_SIZE = 28
TICK_SIZE = 24
REGION_LABEL_SIZE = 28
ANNOTATION_SIZE = 24
LEGEND_SIZE = 10
# Vertical gap between the stacked lines of a wrapped legend label, as a multiple
# of the font size (matplotlib Text default is 1.2). Bump this instead of padding
# wrap_label with blank lines.
LINE_SPACING = 1.4

# The no-domain ablation overrides model.lr_domain_loss_coeff=0 on every run in
# that group; showing "λ_dom=0" on every legend entry would be redundant, so it
# is hidden from the formatted experiment name (falsy override -> hidden, same
# mechanism build_group_table uses for per-group param_display_names). Groups
# without this override (e.g. the with-domain group) are unaffected.
HIDE_DOMAIN_OVERRIDE = {"model.lr_domain_loss_coeff": None}


@dataclass
class GroupLayout:
    """Resolved runs and computed bar/whisker/fill geometry for one group's figure.

    Built by ``build_group_layout`` (no drawing); consumed by ``data_extent``,
    ``group_extent``, and ``render_group_figure``.

    Attributes:
        group (dict): The group definition from the eval YAML.
        models (list[tuple[str, Path]]): ``(display_label, run_dir)`` pairs for
            the fusion models in this group.
        base_name (str): Display name of the baseline model.
        region_base (dict[str, tuple[float, float]]): Per-region baseline
            ``(mean, std)`` in percent, keyed by region name.
        x_base (np.ndarray): X-axis center position of each region's bar
            cluster, shape ``(len(REGIONS),)``.
        bar_width (float): Width of a single bar/whisker slot.
        bar_infos (list[tuple[float, float, float, float, str]]): One entry per
            model bar: ``(xpos, delta_mean, delta_std, run_abs_mean, color)``.
        baseline_infos (list[tuple[float, float]]): One entry per baseline
            whisker (two per region, left/right of the cluster):
            ``(xpos, base_std)``.
        fill_infos (list[tuple[float, float, float]]): One entry per region's
            variance-fill band: ``(left_edge, right_edge, base_std)``.
    """

    group: dict
    models: list[tuple[str, Path]]
    base_name: str
    region_base: dict[str, tuple[float, float]]
    x_base: np.ndarray
    bar_width: float
    bar_infos: list[tuple[float, float, float, float, str]]
    baseline_infos: list[tuple[float, float]]
    fill_infos: list[tuple[float, float, float]]


def save_figure(fig, figures_dir: Path, filename: str, run_name: str, **savefig_kwargs) -> None:
    """Write a figure to the repo's figures dir and mirror it into the thesis repo.

    Args:
        fig (matplotlib.figure.Figure): Figure to save.
        figures_dir (Path): Destination directory under the repo (e.g.
            ``REPO_ROOT / "figures" / run_name``).
        filename (str): Output filename (e.g. ``"<group>_region_deltas.svg"``).
        run_name (str): Eval YAML stem, used as the mirrored subfolder name under
            ``THESIS_IMAGES_DIR``.
        **savefig_kwargs: Forwarded to ``fig.savefig`` (e.g. ``bbox_inches``).
    """
    out_path = figures_dir / filename
    fig.savefig(out_path, format="svg", **savefig_kwargs)
    print(f"  wrote {out_path}")
    if THESIS_IMAGES_DIR.parent.exists():
        thesis_dir = THESIS_IMAGES_DIR / run_name
        thesis_dir.mkdir(parents=True, exist_ok=True)
        thesis_path = thesis_dir / filename
        fig.savefig(thesis_path, format="svg", **savefig_kwargs)
        print(f"  wrote {thesis_path}")


def mean_std(run_dir: Path | None, metric: str) -> tuple[float, float] | None:
    """Mean and std (in %) of `metric` across seeds for a run dir, or None.

    Args:
        run_dir (Path | None): Run's top-level log directory, or None.
        metric (str): Metric key to look up (e.g.
            ``"test/test-od-region-europe-task-acc"``).

    Returns:
        tuple[float, float] | None: ``(mean, std)`` in percent across seeds, or
            None if ``run_dir`` is None or the metric has no values.
    """
    vals = load_test_metrics(run_dir, [metric])[metric] if run_dir else []
    if not vals:
        return None
    return float(np.mean(vals)) * 100.0, float(np.std(vals)) * 100.0


def plot_legend(models: list[tuple[str, Path]], base_name: str, figures_dir: Path, run_name: str,
                filename: str, legend_size: int = LEGEND_SIZE) -> None:
    """Standalone legend: one color swatch per fusion model plus the baseline's
    gray diamond+whisker marker, laid out in a single horizontal row.

    Args:
        models (list[tuple[str, Path]]): ``(display_label, run_dir)`` pairs for
            the fusion models in the group, as built by ``build_group_layout``.
        base_name (str): Display name of the baseline model.
        figures_dir (Path): Destination directory for the saved figure (see
            ``save_figure``).
        run_name (str): Eval YAML stem, forwarded to ``save_figure`` for the
            thesis-repo mirror subfolder.
        filename (str): Output filename for the legend figure.
        legend_size (int): Font size for the legend entries.
    """
    # Baseline entry first so it sits on the left of the row, ahead of the fusion
    # model swatches.
    handles = [
        Line2D([0], [0], marker="D", linestyle="none", markerfacecolor=BASELINE_COLOR,
               markeredgecolor="black", markeredgewidth=0.5, markersize=7, label=base_name)
    ]
    handles += [
        Patch(facecolor=model_color(label, f"C{mi}"), edgecolor="black", linewidth=0.6, label=wrap_label(label))
        for mi, (label, _) in enumerate(models)
    ]
    fig = plt.figure(figsize=(6.0, 0.6))
    leg = fig.legend(handles=handles, loc="center", ncol=len(handles), fontsize=legend_size, frameon=False)
    for txt in leg.get_texts():
        txt.set_linespacing(LINE_SPACING)
    save_figure(fig, figures_dir, filename, run_name, bbox_inches="tight")
    plt.close(fig)


def build_group_layout(group: dict, run_name: str, translations: dict) -> GroupLayout | None:
    """Resolve a group's runs and compute all bar/whisker/fill geometry, without
    drawing anything -- kept separate from rendering so ``main`` can inspect every
    group's data extent first and derive one shared y-axis range.

    Args:
        group (dict): Group definition from the eval YAML; reads ``"runs"``
            (``[baseline_ref, *model_refs]``) and ``"name"``.
        run_name (str): Eval YAML stem, used as the default run-config name for
            refs without an explicit ``config@`` prefix.
        translations (dict): Display-name translations (see
            ``results.utils.load_translations``).

    Returns:
        GroupLayout | None: The resolved layout, or None if the baseline's run
            directory or every fusion model's run directory is missing.
    """
    run_refs: list[str] = group["runs"]
    baseline_ref, *model_refs = run_refs

    config_names = {parse_run_ref(ref, run_name)[0] for ref in run_refs}
    run_configs = load_run_configs(config_names)
    run_experiments = resolve_experiments([group], run_configs, run_name)

    _, base_key = parse_run_ref(baseline_ref, run_name)
    base_dir = find_run_dir(base_key)
    if base_dir is None:
        print(f"No run dir for baseline '{baseline_ref}' ('{group['name']}'), skipping", file=sys.stderr)
        return None
    base_name = format_experiment_name(baseline_ref, run_experiments, translations, latex=True)

    models: list[tuple[str, Path]] = []
    for ref in model_refs:
        _, exp_key = parse_run_ref(ref, run_name)
        run_dir = find_run_dir(exp_key)
        if run_dir is None:
            print(f"Warning: no run dir for '{ref}', skipping", file=sys.stderr)
            continue
        label = format_experiment_name(ref, run_experiments, translations, latex=True, param_overrides=HIDE_DOMAIN_OVERRIDE)
        models.append((label, run_dir))

    if not models:
        print(f"No fusion models found for '{group['name']}'; skipping.", file=sys.stderr)
        return None

    # Baseline's own per-region mean/std, needed both to compute model deltas
    # and to draw the baseline's noise-floor whisker.
    region_base: dict[str, tuple[float, float]] = {}
    for region in REGIONS:
        metric = f"test/test-od-region-{region}-task-acc"
        ms = mean_std(base_dir, metric)
        if ms is None:
            print(f"Warning: missing '{metric}' for baseline, skipping region", file=sys.stderr)
            continue
        region_base[region] = ms

    # One extra slot on each side (left and right of the model bars) for a
    # baseline whisker, so the baseline's noise floor brackets the group
    # without overlapping the colored model bars.
    n_slots = len(models) + 2
    cluster_width = min(BAR_CLUSTER_WIDTH, GROUP_SPACING * MAX_GROUP_FILL)
    bar_width = cluster_width / n_slots
    x_base = np.arange(len(REGIONS)) * GROUP_SPACING

    def slot_x(gi: int, slot_idx: int) -> float:
        return x_base[gi] + (slot_idx - (n_slots - 1) / 2) * bar_width

    # (xpos, delta_mean, delta_std, run_abs_mean, color) per model bar.
    bar_infos: list[tuple[float, float, float, float, str]] = []
    # (xpos, base_std) per region, one entry per side, for the baseline whiskers.
    baseline_infos: list[tuple[float, float]] = []
    # (left_edge, right_edge, base_std) per region for the gray variance fill,
    # spanning the full cluster width out to the baseline whiskers.
    fill_infos: list[tuple[float, float, float]] = []

    for gi, region in enumerate(REGIONS):
        if region not in region_base:
            continue
        base_std = region_base[region][1]
        baseline_infos.append((slot_x(gi, 0), base_std))
        baseline_infos.append((slot_x(gi, n_slots - 1), base_std))
        left_edge = slot_x(gi, 0)
        right_edge = slot_x(gi, n_slots - 1)
        fill_infos.append((left_edge, right_edge, base_std))

    for mi, (label, run_dir) in enumerate(models):
        color = model_color(label, f"C{mi}")
        for gi, region in enumerate(REGIONS):
            if region not in region_base:
                continue
            metric = f"test/test-od-region-{region}-task-acc"
            run_ms = mean_std(run_dir, metric)
            if run_ms is None:
                print(f"Warning: missing '{metric}' for '{label}', skipping bar", file=sys.stderr)
                continue
            run_mean, run_std = run_ms
            base_mean, base_std = region_base[region]
            delta = run_mean - base_mean
            # Independent-variance approximation for the difference of two means
            # (ignores any positive correlation between seeds of the two runs).
            delta_std = float(np.hypot(run_std, base_std))
            xpos = slot_x(gi, mi + 1)
            bar_infos.append((xpos, delta, delta_std, run_mean, color))

    return GroupLayout(
        group=group, models=models, base_name=base_name, region_base=region_base,
        x_base=x_base, bar_width=bar_width, bar_infos=bar_infos,
        baseline_infos=baseline_infos, fill_infos=fill_infos,
    )


def data_extent(layout: GroupLayout) -> tuple[float, float]:
    """A group's raw (min, max) across every bar's error-bar span and every
    baseline whisker, before any padding is added.

    Args:
        layout (GroupLayout): Resolved group geometry (see ``build_group_layout``).

    Returns:
        tuple[float, float]: ``(min, max)`` of the data, or ``(0.0, 0.0)`` if the
            layout has no bars or baseline whiskers.
    """
    extents = [delta - delta_std for _, delta, delta_std, _, _ in layout.bar_infos] + \
              [delta + delta_std for _, delta, delta_std, _, _ in layout.bar_infos] + \
              [-base_std for _, base_std in layout.baseline_infos] + \
              [base_std for _, base_std in layout.baseline_infos]
    return (min(extents), max(extents)) if extents else (0.0, 0.0)


def group_extent(layout: GroupLayout, show_baseline_header: bool) -> tuple[float, float]:
    """A group's own (bottom, top) y-limit candidate: the data extent padded for
    breathing room, plus headroom for the baseline header text when shown.

    Args:
        layout (GroupLayout): Resolved group geometry (see ``build_group_layout``).
        show_baseline_header (bool): Whether extra headroom for the baseline's
            per-region accuracy header text is needed.

    Returns:
        tuple[float, float]: ``(bottom, top)`` y-limit candidate for this group alone.
    """
    lo, hi = data_extent(layout)
    span = max(hi - lo, 1.0)
    pad = span * 0.1
    if show_baseline_header:
        top = hi + pad + span * 0.12
    else:
        top = hi + pad
    return lo - pad, top


def render_group_figure(layout: GroupLayout, run_name: str, figures_dir: Path, ylim: tuple[float, float],
                        show_baseline_header: bool, show_region_labels: bool) -> None:
    """Draw and write one group's figure (+ legend), using the shared ``ylim``
    computed across every group in ``main`` rather than this group's own extent.

    Args:
        layout (GroupLayout): Resolved group geometry (see ``build_group_layout``).
        run_name (str): Eval YAML stem, forwarded to ``save_figure`` for the
            thesis-repo mirror subfolder and used in the output filenames.
        figures_dir (Path): Destination directory for the saved figures.
        ylim (tuple[float, float]): Shared ``(bottom, top)`` y-axis limits to apply.
        show_baseline_header (bool): Whether to draw the baseline's per-region
            absolute-accuracy header text above the bars.
        show_region_labels (bool): Whether to draw region names on the x-axis
            (vs. leaving it unlabeled).
    """
    fig, ax = plt.subplots(figsize=(3.0 * len(REGIONS) + 3.0, 8.5))

    for left_edge, right_edge, base_std in layout.fill_infos:
        ax.add_patch(Rectangle(
            (left_edge, -base_std), right_edge - left_edge, 2 * base_std,
            facecolor=BASELINE_COLOR, alpha=0.15, edgecolor="none", zorder=1,
        ))

    for xpos, base_std in layout.baseline_infos:
        ax.errorbar(
            xpos, 0.0, yerr=base_std, fmt="D", color=BASELINE_COLOR, ecolor=BASELINE_COLOR,
            markeredgecolor="black", markeredgewidth=0.5, markersize=8,
            capsize=5, elinewidth=1.4, zorder=3,
        )

    for xpos, delta, delta_std, _, color in layout.bar_infos:
        ax.bar(
            xpos, delta, width=layout.bar_width,
            yerr=delta_std, capsize=5, error_kw=dict(elinewidth=0.8, ecolor="black"),
            color=color, edgecolor="black", linewidth=0.6,
        )

    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.set_xticks(layout.x_base)
    if show_region_labels:
        ax.set_xticklabels([REGION_LABELS[r] for r in REGIONS])
        ax.tick_params(axis="x", labelsize=REGION_LABEL_SIZE)
    else:
        ax.set_xticklabels(["" for _ in REGIONS])
        ax.tick_params(axis="x", length=0)
    ax.set_ylabel("OOD Acc. (%) vs. Baseline", fontsize=LABEL_SIZE)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", labelsize=TICK_SIZE)
    ax.grid(True, axis="y", linestyle=":", linewidth=0.5, alpha=0.6)

    # Per-region header: the baseline's own absolute accuracy, placed just
    # above this group's own (unpadded-by-the-shared-range) data extent, so it
    # tracks this group's bars regardless of how tall the shared axis ended up.
    if show_baseline_header:
        lo, hi = data_extent(layout)
        span = max(hi - lo, 1.0)
        y_header = hi + span * 0.1
        for gi, region in enumerate(REGIONS):
            if region not in layout.region_base:
                continue
            base_mean, _ = layout.region_base[region]
            ax.text(
                layout.x_base[gi], y_header, f"{base_mean:.1f}%",
                ha="center", va="bottom", fontsize=ANNOTATION_SIZE, fontweight="bold",
                color=BASELINE_COLOR,
            )

    ax.set_ylim(*ylim)

    fig.tight_layout()
    safe_group = layout.group["name"].lower().replace(" ", "_").replace("-", "_")
    save_figure(fig, figures_dir, f"{safe_group}_region_deltas.svg", run_name)
    plt.close(fig)

    plot_legend(layout.models, layout.base_name, figures_dir, run_name, f"{safe_group}_region_deltas_legend.svg")


def main() -> None:
    """CLI entry point: render the region-wise delta-bar figures for GROUP_NAMES.

    Parses the eval YAML path (default ``feature_fusion.yaml``), resolves every
    listed group's geometry (``build_group_layout``) in a first pass, computes
    shared y-axis ranges per ``YLIM_GROUPS``, then renders each group's figure and
    legend (``render_group_figure``) in a second pass.
    """
    parser = argparse.ArgumentParser(
        description="Region-wise OOD accuracy delta bars for the hardcoded fusion-comparison groups"
    )
    parser.add_argument(
        "eval_yaml",
        nargs="?",
        default=str(EVAL_CONFIG_DIR / "feature_fusion.yaml"),
        help="Path to eval YAML (default: src/train/configs/eval/feature_fusion.yaml)",
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
    groups: list[dict] = cfg["groups"]
    translations = load_translations()

    figures_dir = REPO_ROOT / "figures" / run_name
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Pass 1: resolve every group's data (no drawing yet) so its extent can feed
    # into the shared y-axis ranges computed per YLIM_GROUPS below.
    layouts: dict[str, GroupLayout] = {}
    for group_name in GROUP_NAMES:
        group = next((g for g in groups if g["name"] == group_name), None)
        if group is None:
            print(f"Warning: group '{group_name}' not found in {eval_yaml.name}, skipping", file=sys.stderr)
            continue
        layout = build_group_layout(group, run_name, translations)
        if layout is not None:
            layouts[group_name] = layout

    if not layouts:
        return

    def shared_range(names: list[str]) -> tuple[float, float]:
        bottoms, tops = zip(*(group_extent(layouts[n], show_baseline_header=True) for n in names))
        return min(bottoms), max(tops)

    # Each YLIM_GROUPS sublist shares one y-range across just its members; any
    # rendered group not listed there falls back to its own extent.
    ylim_by_group: dict[str, tuple[float, float]] = {}
    for names in YLIM_GROUPS:
        members = [n for n in names if n in layouts]
        if not members:
            continue
        rng = shared_range(members)
        for n in members:
            ylim_by_group[n] = rng
    for n in layouts:
        ylim_by_group.setdefault(n, shared_range([n]))

    # Pass 2: draw every group's figure against its YLIM_GROUPS-shared range.
    for group_name, layout in layouts.items():
        print(f"=== {group_name} ===")
        render_group_figure(layout, run_name, figures_dir, ylim_by_group[group_name],
                            show_baseline_header=True, show_region_labels=True)


if __name__ == "__main__":
    main()
