#!/usr/bin/env python3
"""Grouped bar plots comparing decision-fusion ablation modes across decision rules.

Reads an eval YAML (e.g. ``decision_fusion.yaml``) whose groups are named
``"<family> <DecisionRule>"`` (e.g. "Decision Fusion DenseNet121 Trained Sum")
and whose ``runs`` list is ``[baseline, full, no_prior, no_domain,
no_prior_no_domain]``. Groups sharing the same family prefix (i.e. differing
only in the trailing decision rule) are combined into one figure per metric:
one tight bar group per decision rule, with the four ablation modes as bars
within it (each mode has its own color, consistent across decision rules),
and the shared baseline drawn as a dashed horizontal line across the full
width.
"""

import argparse
import colorsys
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from results.eval_metrics import format_metric_name, load_test_metrics
from results.utils import (
    EVAL_CONFIG_DIR,
    REPO_ROOT,
    find_run_dir,
    load_translations,
    parse_run_ref,
)

# Mirror every figure into the thesis repo's images dir, under a per-run
# subfolder matching the repo's ``figures/<run_name>/`` layout.
THESIS_IMAGES_DIR = Path.home() / "git" / "thesis" / "images"


def save_figure(fig, figures_dir: Path, filename: str, run_name: str, **savefig_kwargs) -> None:
    """Write `fig` to the repo figures dir and mirror it into the thesis images
    dir under a ``<run_name>/`` subfolder when the thesis repo is present."""
    out_path = figures_dir / filename
    fig.savefig(out_path, format="svg", **savefig_kwargs)
    print(f"  wrote {out_path}")
    if THESIS_IMAGES_DIR.parent.exists():
        thesis_dir = THESIS_IMAGES_DIR / run_name
        thesis_dir.mkdir(parents=True, exist_ok=True)
        thesis_path = thesis_dir / filename
        fig.savefig(thesis_path, format="svg", **savefig_kwargs)
        print(f"  wrote {thesis_path}")


DECISION_RULES = ["Sum", "Max", "Min", "GeoPrior"]
# Group names still end in "GeoPrior" (matching the YAML), but it should display as "Product".
RULE_DISPLAY = {"GeoPrior": "Product"}

# Family names are "Decision Fusion <HR model> [<suffix>]". The HR model name maps
# directly to its display name; the suffix (if any) says where the LR side comes
# from. A suffix not listed here (e.g. "Trained") means LR mirrors HR.
HR_MODEL_DISPLAY = {
    "DenseNet121": "DenseNet-121",
    "DINOv3": "DINOv3",
}
LR_DISPLAY_BY_SUFFIX = {
    "SatCLIP": "SatCLIP, L=40",
}


def split_family(family: str) -> tuple[str, str]:
    """Split a family like "Decision Fusion DenseNet121 SatCLIP" into
    ``(hr_display, lr_display)``."""
    body = family.removeprefix("Decision Fusion ").strip()
    hr_key, _, suffix = body.partition(" ")
    hr_display = HR_MODEL_DISPLAY.get(hr_key, hr_key)
    lr_display = LR_DISPLAY_BY_SUFFIX.get(suffix, hr_display)
    return hr_display, lr_display

# Visual encoding: the "prior" axis (use_class_prior) maps to shade (dark = prior,
# light = no prior); the "domain" axis (lr_domain_loss_coeff) maps to hatching
# (plain = domain, striped = no domain). The two axes are independent so the
# encoding stays consistent regardless of decision rule.
BASE_COLOR = "#1f78c5"
NO_DOMAIN_HATCH = "//"


def _lighten(hex_color: str, amount: float = 0.55) -> tuple[float, float, float]:
    """Blend a color toward white by `amount` (0 = unchanged, 1 = white)."""
    r, g, b = mcolors.to_rgb(hex_color)
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    l = l + (1 - l) * amount
    return colorsys.hls_to_rgb(h, l, s)


# (label, has_prior, hatch) per mode, in the order the runs list provides them:
# [full, no_prior, no_domain, no_prior_no_domain]. The actual fill color is the
# rule's base color (dark = prior, lightened = no prior), tinted per decision rule.
MODES = [
    ("Prior + Domain", True, None),
    ("No Prior + Domain", False, None),
    ("Prior + No Domain", True, NO_DOMAIN_HATCH),
    ("No Prior + No Domain", False, NO_DOMAIN_HATCH),
]
MODE_LABELS = [m[0] for m in MODES]
# Neutral swatch color for the mode legend, where hue carries no meaning (it's
# the rule's base color in the plot) and only shade + hatch encode the mode.
MODE_LEGEND_COLOR = "#888888"


def mode_color(base: str | tuple, has_prior: bool) -> tuple[float, float, float]:
    """Fill color for a mode: the base color for prior, lightened for no prior."""
    return mcolors.to_rgb(base) if has_prior else _lighten(base)

# --- Context-comparison plots (image vs. location context) -------------------
# These summarise, per backbone, the "Prior + Domain" (full) variant of each
# decision rule for the two context sources: image (LR mirrors HR) vs. location
# (SatCLIP embedding), each shown alongside its "Trained" counterpart (decision
# heads learned jointly from scratch instead of reusing the frozen single-branch
# heads). The bar chart renders frozen and trained as two separate figures (kept
# a plain 2-color context comparison each) rather than a hatch, since hatch on
# these bars already means something else (domain on/off) in the per-family
# mode-ablation figures. The trade-off scatter instead encodes frozen/trained as
# a filled/hollow marker, since it's a different mark type (dots, not bars) and
# already carries color (rule) and shape (context).
WRA_METRIC = "test/test-od-worst-group-task-acc"
OOD_METRIC = "test/test-od-task-acc"

CONTEXT_ORDER = ["Image", "Location"]
CONTEXT_COLORS = {"Image": "#cf440a", "Location": "#115fb0"}
CONTEXT_MARKERS = {"Image": "o", "Location": "^"}
RULE_COLORS = {
    "Sum": "#e8701a",
    "Max": "#cf440a",
    "Min": "#3f93c9",
    "GeoPrior": "#115fb0",
}


def parse_group_name(name: str) -> tuple[str, str] | None:
    """Split a group name like "Decision Fusion DenseNet121 Trained Sum" into
    ``(family, decision_rule)``, or ``None`` if it doesn't end in a known rule."""
    for rule in DECISION_RULES:
        if name.endswith(f" {rule}"):
            return name[: -len(rule) - 1], rule
    return None


def context_and_trained(family: str) -> tuple[str, bool] | None:
    """Map a fusion family to its ``(context, trained)`` pair: context is
    "Image" (LR mirrors HR) or "Location" (SatCLIP suffix); ``trained`` marks
    the jointly-trained decision-head variant. ``None`` excludes families
    matching neither pattern."""
    body = family.removeprefix("Decision Fusion ").strip()
    _, _, suffix = body.partition(" ")
    if suffix == "":
        return "Image", False
    if suffix == "Trained":
        return "Image", True
    if suffix == "SatCLIP":
        return "Location", False
    if suffix == "SatCLIP Trained":
        return "Location", True
    return None


def backbone_of(family: str) -> str:
    """Display name of the HR backbone for a fusion family."""
    body = family.removeprefix("Decision Fusion ").strip()
    hr_key, _, _ = body.partition(" ")
    return HR_MODEL_DISPLAY.get(hr_key, hr_key)


def _mean_std(ref: str, run_name: str, metric: str) -> tuple[float, float] | None:
    """Mean and std (in %) of `metric` across seeds for a run ref, or None."""
    _, exp_key = parse_run_ref(ref, run_name)
    run_dir = find_run_dir(exp_key)
    vals = load_test_metrics(run_dir, [metric])[metric] if run_dir else []
    if not vals:
        return None
    return float(np.mean(vals)) * 100.0, float(np.std(vals)) * 100.0


def collect_backbones(families: dict[str, dict[str, dict]]) -> dict[str, dict]:
    """Group fusion families by backbone, context, and trained flag for the
    summary plots.

    Returns ``{backbone: {"baseline": ref, "contexts": {context: {trained: {rule: group}}}}}``.
    """
    backbones: dict[str, dict] = {}
    for family, rule_groups in families.items():
        parsed = context_and_trained(family)
        if parsed is None:
            continue
        context, trained = parsed
        bb = backbones.setdefault(backbone_of(family), {"baseline": None, "contexts": {}})
        bb["contexts"].setdefault(context, {})[trained] = rule_groups
        if bb["baseline"] is None and rule_groups:
            bb["baseline"] = next(iter(rule_groups.values()))["runs"][0]
    return backbones


def plot_context_bars(backbone, info, run_name, translations, figures_dir,
                      label_size, tick_size) -> None:
    """WRA grouped bars: one group per decision rule, one bar per context
    (image vs. location), using the Prior + Domain variant of each rule.
    Rendered as two separate figures, one for frozen and one for jointly
    trained decision heads, so each stays a plain 2-color context comparison
    -- keeping color+hatch meaning unambiguous relative to the per-family
    mode-ablation bars, which already use hatch for a different axis (domain
    on/off). The thesis places the two figures in a grid alongside the DINOv3
    pair, all four sharing the one legend from ``plot_context_legend``."""
    contexts = info["contexts"]
    for trained in (False, True):
        _plot_context_bars_variant(
            backbone, contexts, trained, info["baseline"], run_name, translations,
            figures_dir, label_size, tick_size,
        )


def _plot_context_bars_variant(backbone, contexts, trained, baseline_ref, run_name,
                               translations, figures_dir, label_size, tick_size) -> None:
    contexts_present = [c for c in CONTEXT_ORDER if c in contexts and trained in contexts[c]]
    rules_present = [r for r in DECISION_RULES if any(r in contexts[c][trained] for c in contexts_present)]
    if not rules_present or not contexts_present:
        return

    fig, ax = plt.subplots(figsize=(2.2 * len(rules_present) + 1.5, 5.5))
    x_base = np.arange(len(rules_present))
    n_ctx = len(contexts_present)
    bar_width = 0.6 / n_ctx

    # Track each bar's whisker extents so the y-axis can be cropped to frame
    # the spread between contexts and the baseline rather than starting at zero.
    extents: list[float] = []

    for ci, ctx in enumerate(contexts_present):
        for gi, rule in enumerate(rules_present):
            group = contexts[ctx][trained].get(rule)
            if group is None:
                continue
            ms = _mean_std(group["runs"][1], run_name, WRA_METRIC)
            if ms is None:
                continue
            mean, std = ms
            extents.extend((mean - std, mean + std))
            xpos = x_base[gi] + (ci - (n_ctx - 1) / 2) * bar_width
            ax.bar(
                xpos, mean, width=bar_width,
                yerr=std, capsize=6, error_kw=dict(elinewidth=1.5, capthick=1.5, ecolor="black"),
                color=CONTEXT_COLORS[ctx], edgecolor="black", linewidth=0.6,
            )

    base_ms = _mean_std(baseline_ref, run_name, WRA_METRIC) if baseline_ref else None
    if base_ms:
        base_mean, base_std = base_ms
        extents.extend((base_mean - base_std, base_mean + base_std))
        ax.axhspan(base_mean - base_std, base_mean + base_std, color="black", alpha=0.1, linewidth=0)
        ax.axhline(base_mean, color="black", linestyle="--", linewidth=1.2)

    # Crop the y-axis to a padded window around the data so small context/
    # baseline gaps are legible. The bars become truncated (they no longer
    # start at 0), matching the per-family mode-ablation bars.
    if extents:
        lo, hi = min(extents), max(extents)
        pad = max((hi - lo) * 0.125, 0.5)
        ax.set_ylim(max(0.0, lo - pad), hi + pad)

    ax.set_xticks(x_base)
    ax.set_xticklabels([RULE_DISPLAY.get(r, r) for r in rules_present])
    ax.set_ylabel(f"{format_metric_name(WRA_METRIC, translations)} (%)", fontsize=label_size)
    ax.tick_params(axis="both", labelsize=tick_size)
    ax.set_axisbelow(True)
    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.6)
    # No per-panel legend: all context-bar panels share one color encoding
    # (Image/Location/HR-only baseline), so a single legend covers every
    # panel -- see plot_context_legend.

    fig.tight_layout()
    safe = backbone.lower().replace(" ", "_").replace("-", "_")
    suffix = "_trained" if trained else ""
    save_figure(fig, figures_dir, f"context_bars_{safe}{suffix}_wra.svg", run_name)
    plt.close(fig)


def plot_context_legend(run_name, figures_dir, legend_size) -> None:
    """Standalone legend for the context-bar grid: Image context / Location
    context / HR only, laid out in a single horizontal row (``ncol`` equal to
    the handle count forces one row). Shared across all four context-bar
    panels (2 backbones x frozen/trained) since they all use this exact
    encoding -- see ``_plot_context_bars_variant``."""
    handles = [
        Patch(facecolor=CONTEXT_COLORS["Image"], edgecolor="black", linewidth=0.6, label="Image context"),
        Patch(facecolor=CONTEXT_COLORS["Location"], edgecolor="black", linewidth=0.6, label="Location context"),
        Line2D([0], [0], color="black", linestyle="--", label="HR only"),
    ]
    fig = plt.figure(figsize=(6.0, 0.5))
    fig.legend(handles=handles, loc="center", ncol=len(handles), fontsize=legend_size, frameon=False)
    save_figure(fig, figures_dir, "context_legend.svg", run_name, bbox_inches="tight")
    plt.close(fig)


def plot_tradeoff_scatter(backbone, info, run_name, translations, figures_dir,
                          label_size, tick_size) -> None:
    """Trade-off scatter: OOD overall accuracy (x) vs. WRA (y), one point per
    (decision rule, context), Prior + Domain variant, with the HR baseline as
    a star. Rendered as two separate figures, one for frozen and one for
    jointly-trained decision heads (mirroring ``plot_context_bars``), so each
    stays a plain rule-color x context-marker comparison. No per-panel
    legend: the encoding is identical across both backbones and both
    frozen/trained variants, so one legend (``plot_tradeoff_legend``) covers
    all of them."""
    for trained in (False, True):
        _plot_tradeoff_scatter_variant(backbone, info, trained, run_name, translations,
                                       figures_dir, label_size, tick_size)


def _plot_tradeoff_scatter_variant(backbone, info, trained, run_name, translations,
                                   figures_dir, label_size, tick_size) -> None:
    contexts = info["contexts"]
    contexts_present = [c for c in CONTEXT_ORDER if c in contexts and trained in contexts[c]]
    if not contexts_present:
        return

    fig, ax = plt.subplots(figsize=(6.0, 5.0))

    for ctx in contexts_present:
        for rule in DECISION_RULES:
            group = contexts[ctx][trained].get(rule)
            if group is None:
                continue
            ood = _mean_std(group["runs"][1], run_name, OOD_METRIC)
            wra = _mean_std(group["runs"][1], run_name, WRA_METRIC)
            if ood is None or wra is None:
                continue
            ax.errorbar(
                ood[0], wra[0], xerr=ood[1], yerr=wra[1],
                marker=CONTEXT_MARKERS[ctx], markersize=14,
                color=RULE_COLORS[rule], markeredgecolor="black", markeredgewidth=0.6,
                ecolor="gray", elinewidth=0.8, capsize=6, linestyle="none",
            )

    base_ood = _mean_std(info["baseline"], run_name, OOD_METRIC) if info["baseline"] else None
    base_wra = _mean_std(info["baseline"], run_name, WRA_METRIC) if info["baseline"] else None
    if base_ood and base_wra:
        ax.errorbar(
            base_ood[0], base_wra[0], xerr=base_ood[1], yerr=base_wra[1],
            marker="*", markersize=18, color="lightgray", markeredgecolor="black", markeredgewidth=0.6,
            ecolor="gray", elinewidth=0.8, capsize=2, linestyle="none",
        )

    ax.set_xlabel(f"{format_metric_name(OOD_METRIC, translations)} (%)", fontsize=label_size)
    ax.set_ylabel(f"{format_metric_name(WRA_METRIC, translations)} (%)", fontsize=label_size)
    ax.tick_params(axis="both", labelsize=tick_size)
    ax.set_axisbelow(True)
    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.6)

    fig.tight_layout()
    safe = backbone.lower().replace(" ", "_").replace("-", "_")
    suffix = "_trained" if trained else ""
    save_figure(fig, figures_dir, f"tradeoff_{safe}{suffix}_ood_wra.svg", run_name)
    plt.close(fig)


def plot_tradeoff_legend(run_name, figures_dir, legend_size) -> None:
    """Standalone legend for the trade-off scatter grid: decision-rule colors,
    context marker shapes, and the HR-only baseline star -- identical across
    both backbones, so one legend covers every panel. Rule colors and context
    markers are two different encodings, so they're laid out as two rows
    rather than forced into one."""
    rule_handles = [
        Line2D([0], [0], marker="s", linestyle="none", markerfacecolor=RULE_COLORS[r],
               markeredgecolor="black", markeredgewidth=0.7, markersize=8, label=RULE_DISPLAY.get(r, r))
        for r in DECISION_RULES
    ]
    ctx_handles = [
        Line2D([0], [0], marker=CONTEXT_MARKERS[c], linestyle="none", markerfacecolor="white",
               markeredgecolor="black", markeredgewidth=0.7, markersize=8, label=f"{c} context")
        for c in CONTEXT_ORDER
    ]
    baseline_handle = Line2D([0], [0], marker="*", linestyle="none", markerfacecolor="lightgray",
                             markeredgecolor="black", markeredgewidth=0.6, markersize=12, label="HR only")

    fig = plt.figure(figsize=(6.0, 1.0))
    fig.legend(handles=rule_handles + ctx_handles + [baseline_handle], loc="center",
              ncol=len(rule_handles), fontsize=legend_size, frameon=False)
    save_figure(fig, figures_dir, "tradeoff_legend.svg", run_name, bbox_inches="tight")
    plt.close(fig)


def plot_mode_bars(family, rule_groups, run_name, primary_metrics, translations, figures_dir,
                   label_size, tick_size) -> None:
    """Per-family grouped bars: one bar group per decision rule, with the four
    ablation modes (prior/domain on/off) as bars within it, one figure per metric.
    No per-panel legend: the mode encoding (shade + hatch) and the baseline are
    identical across every family and metric, so one legend (``plot_mode_legend``)
    covers all of them."""
    rules_present = [r for r in DECISION_RULES if r in rule_groups]
    if not rules_present:
        return

    n_modes = len(MODE_LABELS)
    bar_width = 0.6 / n_modes

    base_ref = rule_groups[rules_present[0]]["runs"][0]
    _, base_key = parse_run_ref(base_ref, run_name)
    base_dir = find_run_dir(base_key)

    for metric in primary_metrics:
        fig, ax = plt.subplots(figsize=(2.2 * len(rules_present) + 1.5, 3.5))
        x_base = np.arange(len(rules_present))

        # Track each bar's whisker extents so the y-axis can be cropped to frame
        # the spread between modes and the baseline rather than starting at zero.
        extents: list[float] = []

        for gi, rule in enumerate(rules_present):
            mode_refs = rule_groups[rule]["runs"][1:]
            for mi, ref in enumerate(mode_refs):
                _, exp_key = parse_run_ref(ref, run_name)
                run_dir = find_run_dir(exp_key)
                vals = load_test_metrics(run_dir, [metric])[metric] if run_dir else []
                if not vals:
                    continue
                mean, std = np.mean(vals) * 100.0, np.std(vals) * 100.0
                extents.extend((mean - std, mean + std))
                xpos = x_base[gi] + (mi - (n_modes - 1) / 2) * bar_width
                _, has_prior, hatch = MODES[mi]
                ax.bar(
                    xpos, mean, width=bar_width,
                    yerr=std, capsize=3, error_kw=dict(elinewidth=0.8, ecolor="black"),
                    color=mode_color(RULE_COLORS[rule], has_prior),
                    edgecolor="black", linewidth=0.6, hatch=hatch,
                )

        base_vals = load_test_metrics(base_dir, [metric])[metric] if base_dir else []
        if base_vals:
            base_mean, base_std = np.mean(base_vals) * 100.0, np.std(base_vals) * 100.0
            extents.extend((base_mean - base_std, base_mean + base_std))
            ax.axhspan(base_mean - base_std, base_mean + base_std, color="black", alpha=0.1, linewidth=0)
            ax.axhline(base_mean, color="black", linestyle="--", linewidth=1.2)

        # Crop the y-axis to a padded window around the data so small mode/baseline
        # gaps are legible. The bars become truncated (they no longer start at 0).
        if extents:
            lo, hi = min(extents), max(extents)
            pad = max((hi - lo) * 0.25, 0.5)
            ax.set_ylim(max(0.0, lo - pad), hi + pad)

        ax.set_xticks(x_base)
        ax.set_xticklabels([RULE_DISPLAY.get(r, r) for r in rules_present])
        ax.set_ylabel(f"{format_metric_name(metric, translations)} (%)", fontsize=label_size)
        ax.tick_params(axis="both", labelsize=tick_size)
        ax.set_axisbelow(True)
        ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.6)

        fig.tight_layout()
        safe_family = family.lower().replace(" ", "_").replace("-", "_")
        safe_metric = metric.rsplit("/", 1)[-1]
        save_figure(fig, figures_dir, f"{safe_family}_{safe_metric}.svg", run_name)
        plt.close(fig)


def plot_mode_legend(run_name, figures_dir, legend_size) -> None:
    """Standalone legend for the per-family mode-ablation bars: the four
    prior/domain modes (shade + hatch, neutral swatch color since hue is the
    rule in the actual chart) plus the HR-only baseline -- identical across
    every family and metric, so one legend covers all of them."""
    handles = [
        Patch(facecolor=mode_color(MODE_LEGEND_COLOR, has_prior),
              edgecolor="black", linewidth=0.6, hatch=hatch, label=lbl)
        for lbl, has_prior, hatch in MODES
    ]
    handles.append(Line2D([0], [0], color="black", linestyle="--", label="HR only"))

    fig = plt.figure(figsize=(11.0, 0.5))
    fig.legend(handles=handles, loc="center", ncol=len(handles), fontsize=legend_size, frameon=False)
    save_figure(fig, figures_dir, "mode_legend.svg", run_name, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Decision-fusion mode comparison plots")
    parser.add_argument(
        "eval_yaml",
        nargs="?",
        default=str(EVAL_CONFIG_DIR / "decision_fusion.yaml"),
        help="Path to eval YAML (default: src/train/configs/eval/decision_fusion.yaml)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Regenerate every per-family mode-ablation metric plot. By default, only "
             "the WRA mode-ablation plot plus the context and trade-off summary plots "
             "(the ones used in the thesis) are regenerated.",
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

    translations = load_translations()

    families: dict[str, dict[str, dict]] = {}
    for group in groups:
        parsed = parse_group_name(group["name"])
        if parsed is None:
            print(f"Warning: skipping group with unrecognized name: {group['name']}", file=sys.stderr)
            continue
        family, rule = parsed
        families.setdefault(family, {})[rule] = group

    figures_dir = REPO_ROOT / "figures" / run_name
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Font sizes per figure type — tune these to taste.
    mode_bars_sizes = dict(label_size=16, tick_size=14)
    context_bars_sizes = dict(label_size=28, tick_size=24, legend_size=22)
    tradeoff_sizes = dict(label_size=18, tick_size=16, legend_size=12)
    mode_legend_size = 16

    # Per-family ablation-mode comparison plots. Without --all, only the WRA
    # metric is redone -- that's the one shown in the thesis alongside the
    # context and trade-off summary plots below.
    mode_bars_metrics = primary_metrics if args.all else [WRA_METRIC]
    for family, rule_groups in families.items():
        plot_mode_bars(family, rule_groups, run_name, mode_bars_metrics, translations,
                       figures_dir, **mode_bars_sizes)
    plot_mode_legend(run_name, figures_dir, legend_size=mode_legend_size)

    # Per-backbone summary plots comparing image vs. location context, plus
    # the one legend shared by every context-bar panel and the one legend
    # shared by every trade-off scatter panel.
    for backbone, info in collect_backbones(families).items():
        plot_context_bars(backbone, info, run_name, translations, figures_dir,
                          label_size=context_bars_sizes["label_size"], tick_size=context_bars_sizes["tick_size"])
        plot_tradeoff_scatter(backbone, info, run_name, translations, figures_dir,
                             label_size=tradeoff_sizes["label_size"], tick_size=tradeoff_sizes["tick_size"])
    plot_context_legend(run_name, figures_dir, legend_size=context_bars_sizes["legend_size"])
    plot_tradeoff_legend(run_name, figures_dir, legend_size=tradeoff_sizes["legend_size"])


if __name__ == "__main__":
    main()
