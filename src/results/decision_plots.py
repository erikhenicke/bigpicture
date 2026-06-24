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


def save_figure(fig, figures_dir: Path, filename: str, run_name: str) -> None:
    """Write `fig` to the repo figures dir and mirror it into the thesis images
    dir under a ``<run_name>/`` subfolder when the thesis repo is present."""
    out_path = figures_dir / filename
    fig.savefig(out_path, format="svg")
    print(f"  wrote {out_path}")
    if THESIS_IMAGES_DIR.parent.exists():
        thesis_dir = THESIS_IMAGES_DIR / run_name
        thesis_dir.mkdir(parents=True, exist_ok=True)
        thesis_path = thesis_dir / filename
        fig.savefig(thesis_path, format="svg")
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
# (SatCLIP embedding). The family suffix selects the context; the "Trained"
# families are excluded.
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


def context_of(family: str) -> str | None:
    """Map a fusion family to its context source: "Image" (LR mirrors HR),
    "Location" (SatCLIP suffix), or ``None`` to exclude it (e.g. "Trained")."""
    body = family.removeprefix("Decision Fusion ").strip()
    _, _, suffix = body.partition(" ")
    if suffix == "":
        return "Image"
    if suffix == "SatCLIP":
        return "Location"
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
    """Group fusion families by backbone and context for the summary plots.

    Returns ``{backbone: {"baseline": ref, "contexts": {context: {rule: group}}}}``.
    """
    backbones: dict[str, dict] = {}
    for family, rule_groups in families.items():
        context = context_of(family)
        if context is None:
            continue
        bb = backbones.setdefault(backbone_of(family), {"baseline": None, "contexts": {}})
        bb["contexts"][context] = rule_groups
        if bb["baseline"] is None and rule_groups:
            bb["baseline"] = next(iter(rule_groups.values()))["runs"][0]
    return backbones


def plot_context_bars(backbone, info, run_name, translations, figures_dir,
                      label_size, tick_size, legend_size) -> None:
    """WRA grouped bars: one group per decision rule, one bar per context
    (image vs. location), using the Prior + Domain variant of each rule."""
    contexts = info["contexts"]
    contexts_present = [c for c in CONTEXT_ORDER if c in contexts]
    rules_present = [r for r in DECISION_RULES if any(r in contexts[c] for c in contexts_present)]
    if not rules_present or not contexts_present:
        return

    fig, ax = plt.subplots(figsize=(2.2 * len(rules_present) + 1.5, 4.5))
    x_base = np.arange(len(rules_present))
    n_ctx = len(contexts_present)
    bar_width = 0.4 / n_ctx

    for ci, ctx in enumerate(contexts_present):
        for gi, rule in enumerate(rules_present):
            group = contexts[ctx].get(rule)
            if group is None:
                continue
            ms = _mean_std(group["runs"][1], run_name, WRA_METRIC)
            if ms is None:
                continue
            mean, std = ms
            xpos = x_base[gi] + (ci - (n_ctx - 1) / 2) * bar_width
            ax.bar(
                xpos, mean, width=bar_width,
                yerr=std, capsize=3, error_kw=dict(elinewidth=0.8, ecolor="black"),
                color=CONTEXT_COLORS[ctx], edgecolor="black", linewidth=0.6,
            )

    handles = [
        Patch(facecolor=CONTEXT_COLORS[c], edgecolor="black", linewidth=0.6, label=f"{c} context")
        for c in contexts_present
    ]

    base_ms = _mean_std(info["baseline"], run_name, WRA_METRIC) if info["baseline"] else None
    if base_ms:
        base_mean, base_std = base_ms
        ax.axhspan(base_mean - base_std, base_mean + base_std, color="black", alpha=0.1, linewidth=0)
        ax.axhline(base_mean, color="black", linestyle="--", linewidth=1.2)
        handles.append(Line2D([0], [0], color="black", linestyle="--", label=f"{backbone}\n(HR only)"))

    ax.set_xticks(x_base)
    ax.set_xticklabels([RULE_DISPLAY.get(r, r) for r in rules_present])
    ax.set_ylabel(f"{format_metric_name(WRA_METRIC, translations)} (%)", fontsize=label_size)
    ax.tick_params(axis="both", labelsize=tick_size)
    ax.legend(handles=handles, fontsize=legend_size, loc="lower right")

    fig.tight_layout()
    safe = backbone.lower().replace(" ", "_").replace("-", "_")
    save_figure(fig, figures_dir, f"context_bars_{safe}_wra.svg", run_name)
    plt.close(fig)


def plot_tradeoff_scatter(backbone, info, run_name, translations, figures_dir,
                          label_size, tick_size, legend_size) -> None:
    """Trade-off scatter: OOD overall accuracy (x) vs. WRA (y), one point per
    (decision rule, context), Prior + Domain variant, with the HR baseline as a star."""
    contexts = info["contexts"]
    contexts_present = [c for c in CONTEXT_ORDER if c in contexts]
    if not contexts_present:
        return

    fig, ax = plt.subplots(figsize=(6.0, 5.0))

    for ctx in contexts_present:
        for rule in DECISION_RULES:
            group = contexts[ctx].get(rule)
            if group is None:
                continue
            ood = _mean_std(group["runs"][1], run_name, OOD_METRIC)
            wra = _mean_std(group["runs"][1], run_name, WRA_METRIC)
            if ood is None or wra is None:
                continue
            ax.errorbar(
                ood[0], wra[0], xerr=ood[1], yerr=wra[1],
                marker=CONTEXT_MARKERS[ctx], markersize=10,
                color=RULE_COLORS[rule], markeredgecolor="black", markeredgewidth=0.6,
                ecolor="gray", elinewidth=0.8, capsize=2, linestyle="none",
            )

    base_ood = _mean_std(info["baseline"], run_name, OOD_METRIC) if info["baseline"] else None
    base_wra = _mean_std(info["baseline"], run_name, WRA_METRIC) if info["baseline"] else None
    if base_ood and base_wra:
        ax.errorbar(
            base_ood[0], base_wra[0], xerr=base_ood[1], yerr=base_wra[1],
            marker="*", markersize=18, color="lightgray", markeredgecolor="black", markeredgewidth=0.6,
            ecolor="gray", elinewidth=0.8, capsize=2, linestyle="none",
        )

    rule_handles = [
        Line2D([0], [0], marker="s", linestyle="none", markerfacecolor=RULE_COLORS[r],
               markeredgecolor="black", markeredgewidth=0.7, markersize=8, label=RULE_DISPLAY.get(r, r))
        for r in DECISION_RULES
    ]
    ctx_handles = [
        Line2D([0], [0], marker=CONTEXT_MARKERS[c], linestyle="none", markerfacecolor="white",
               markeredgecolor="black", markeredgewidth=0.7, markersize=8, label=f"{c} context")
        for c in contexts_present
    ]
    if base_ood and base_wra:
        ctx_handles.append(
            Line2D([0], [0], marker="*", linestyle="none", markerfacecolor="lightgray",
                   markeredgecolor="black", markeredgewidth=0.6, markersize=12, label=f"{backbone}\n(HR only)")
        )

    # Single combined legend: decision-rule colors then context markers.
    ax.legend(handles=rule_handles + ctx_handles, fontsize=legend_size, loc="upper left")

    ax.set_xlabel(f"{format_metric_name(OOD_METRIC, translations)} (%)", fontsize=label_size)
    ax.set_ylabel(f"{format_metric_name(WRA_METRIC, translations)} (%)", fontsize=label_size)
    ax.tick_params(axis="both", labelsize=tick_size)

    fig.tight_layout()
    safe = backbone.lower().replace(" ", "_").replace("-", "_")
    save_figure(fig, figures_dir, f"tradeoff_{safe}_ood_wra.svg", run_name)
    plt.close(fig)


def plot_mode_bars(family, rule_groups, run_name, primary_metrics, translations, figures_dir,
                   label_size, tick_size, legend_size) -> None:
    """Per-family grouped bars: one bar group per decision rule, with the four
    ablation modes (prior/domain on/off) as bars within it, one figure per metric."""
    rules_present = [r for r in DECISION_RULES if r in rule_groups]
    if not rules_present:
        return

    n_modes = len(MODE_LABELS)
    bar_width = 0.6 / n_modes

    base_ref = rule_groups[rules_present[0]]["runs"][0]
    _, base_key = parse_run_ref(base_ref, run_name)
    base_dir = find_run_dir(base_key)
    base_label = "DenseNet-121\n(HR only)"

    for metric in primary_metrics:
        fig, ax = plt.subplots(figsize=(2.2 * len(rules_present) + 1.5, 4.5))
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

        handles = [
            Patch(facecolor=mode_color(MODE_LEGEND_COLOR, has_prior),
                  edgecolor="black", linewidth=0.6, hatch=hatch, label=lbl)
            for lbl, has_prior, hatch in MODES
        ]

        base_vals = load_test_metrics(base_dir, [metric])[metric] if base_dir else []
        if base_vals:
            base_mean, base_std = np.mean(base_vals) * 100.0, np.std(base_vals) * 100.0
            extents.extend((base_mean - base_std, base_mean + base_std))
            ax.axhspan(base_mean - base_std, base_mean + base_std, color="black", alpha=0.1, linewidth=0)
            ax.axhline(base_mean, color="black", linestyle="--", linewidth=1.2)
            handles.append(Line2D([0], [0], color="black", linestyle="--", label=base_label))

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
        ax.legend(handles=handles, fontsize=legend_size, loc="lower right")

        fig.tight_layout()
        safe_family = family.lower().replace(" ", "_").replace("-", "_")
        safe_metric = metric.rsplit("/", 1)[-1]
        save_figure(fig, figures_dir, f"{safe_family}_{safe_metric}.svg", run_name)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Decision-fusion mode comparison plots")
    parser.add_argument(
        "eval_yaml",
        nargs="?",
        default=str(EVAL_CONFIG_DIR / "decision_fusion.yaml"),
        help="Path to eval YAML (default: src/train/configs/eval/decision_fusion.yaml)",
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
    mode_bars_sizes = dict(label_size=16, tick_size=14, legend_size=12)
    context_bars_sizes = dict(label_size=16, tick_size=14, legend_size=12)
    tradeoff_sizes = dict(label_size=18, tick_size=16, legend_size=12)

    # Per-family ablation-mode comparison plots.
    for family, rule_groups in families.items():
        plot_mode_bars(family, rule_groups, run_name, primary_metrics, translations,
                       figures_dir, **mode_bars_sizes)

    # Per-backbone summary plots comparing image vs. location context.
    for backbone, info in collect_backbones(families).items():
        plot_context_bars(backbone, info, run_name, translations, figures_dir, **context_bars_sizes)
        plot_tradeoff_scatter(backbone, info, run_name, translations, figures_dir, **tradeoff_sizes)


if __name__ == "__main__":
    main()
