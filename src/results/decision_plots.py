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
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
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

DECISION_RULES = ["Sum", "Max", "Min", "GeoPrior"]
MODE_LABELS = ["Prior + Domain", "No Prior + Domain", "Prior + No Domain", "No Prior + No Domain"]
MODE_COLORS = ["#3f8fd6", "#f0993b", "#4caf50", "#e0524a"]


def parse_group_name(name: str) -> tuple[str, str] | None:
    """Split a group name like "Decision Fusion DenseNet121 Trained Sum" into
    ``(family, decision_rule)``, or ``None`` if it doesn't end in a known rule."""
    for rule in DECISION_RULES:
        if name.endswith(f" {rule}"):
            return name[: -len(rule) - 1], rule
    return None


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

    n_modes = len(MODE_LABELS)
    bar_width = 0.6 / n_modes

    for family, rule_groups in families.items():
        rules_present = [r for r in DECISION_RULES if r in rule_groups]
        if not rules_present:
            continue

        base_ref = rule_groups[rules_present[0]]["runs"][0]
        _, base_key = parse_run_ref(base_ref, run_name)
        base_dir = find_run_dir(base_key)
        base_label = "DenseNet-121 Baseline"

        for metric in primary_metrics:
            fig, ax = plt.subplots(figsize=(2.2 * len(rules_present) + 1.5, 4.5))
            x_base = np.arange(len(rules_present))

            for gi, rule in enumerate(rules_present):
                mode_refs = rule_groups[rule]["runs"][1:]
                for mi, ref in enumerate(mode_refs):
                    _, exp_key = parse_run_ref(ref, run_name)
                    run_dir = find_run_dir(exp_key)
                    vals = load_test_metrics(run_dir, [metric])[metric] if run_dir else []
                    if not vals:
                        continue
                    mean, std = np.mean(vals) * 100.0, np.std(vals) * 100.0
                    xpos = x_base[gi] + (mi - (n_modes - 1) / 2) * bar_width
                    ax.bar(
                        xpos, mean, width=bar_width,
                        yerr=std, capsize=3, error_kw=dict(elinewidth=0.8, ecolor="black"),
                        color=MODE_COLORS[mi], edgecolor="black", linewidth=0.6,
                    )

            handles = [Patch(facecolor=c, edgecolor="black", linewidth=0.6, label=lbl) for lbl, c in zip(MODE_LABELS, MODE_COLORS)]

            base_vals = load_test_metrics(base_dir, [metric])[metric] if base_dir else []
            if base_vals:
                base_mean, base_std = np.mean(base_vals) * 100.0, np.std(base_vals) * 100.0
                ax.axhspan(base_mean - base_std, base_mean + base_std, color="black", alpha=0.1, linewidth=0)
                ax.axhline(base_mean, color="black", linestyle="--", linewidth=1.2)
                handles.append(Line2D([0], [0], color="black", linestyle="--", label=base_label))

            ax.set_xticks(x_base)
            ax.set_xticklabels(rules_present)
            ax.set_ylabel(f"{format_metric_name(metric, translations)} (%)")
            ax.set_title(family)
            ax.legend(handles=handles, fontsize=7, loc="lower right")

            fig.tight_layout()
            safe_family = family.lower().replace(" ", "_").replace("-", "_")
            safe_metric = metric.rsplit("/", 1)[-1]
            out_path = figures_dir / f"decision_{safe_family}_{safe_metric}.svg"
            fig.savefig(out_path, format="svg")
            plt.close(fig)
            print(f"  wrote {out_path}")


if __name__ == "__main__":
    main()
