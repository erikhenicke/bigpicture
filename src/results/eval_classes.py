#!/usr/bin/env python3
"""Plot per-class accuracy gains/losses of runs relative to a baseline.

Reads the ``class-eval`` section of an eval YAML (e.g. ``feature_fusion.yaml``):

    class-eval:
      baseline: baselines@densenet_baseline
      runs:
        - concat
        - film

For each comparison run and each *setting* -- overall Test-OOD and the baseline's
worst OOD region (lowest ``test-od-region-<r>-task-acc``, kept identical across all
runs so the plots are mutually comparable) -- three diverging bar charts are written
to ``figures/<eval_name>/``:

  * ``..._top5.svg``     -- the five largest per-class accuracy gains and losses.
  * ``..._all.svg``      -- every class, sorted by accuracy delta (run - baseline).
  * ``..._weighted.svg`` -- every class, the delta weighted by class occurrence,
    i.e. ``delta_c * n_c / N``. These bars are each class's contribution (in pp) to
    the overall accuracy change and sum to the headline OOD / region accuracy delta.

Per-class accuracies live in ``metrics_rerun.csv`` (written by ``eval_reproduce.py``),
falling back to ``metrics.csv``; they are averaged across all seeds first. A run whose
loaded metrics lack the per-class keys is skipped with a warning. Class occurrence
counts come from the FMoW metadata CSV (``--metadata``) and are read from the OOD test
split (``split == 'test'``); if the file is absent the weighted plots are skipped.
"""

import argparse
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import pandas as pd
import yaml

from results.utils import (
    EVAL_CONFIG_DIR,
    REPO_ROOT,
    find_run_dir,
    format_experiment_name,
    load_run_configs,
    load_run_metrics,
    load_translations,
    parse_run_ref,
    resolve_experiments,
)

from models.utils import DOMAIN_NAMES, TASK_CLASSES

# FMoW class name -> integer class id (index in the canonical class list).
CLASS_IDS = {name: i for i, name in enumerate(TASK_CLASSES)}

OOD_PREFIX = "test/test-od"
TOP_N = 5
COVERAGE = 0.8  # fraction of gains / losses the filtered weighted plot retains
GAIN_COLOR = "#2ca02c"
LOSS_COLOR = "#d62728"
ACC_COLOR = "#1f77b4"
# Sequential colormap for occurrence: low fraction -> light, high -> dark.
OCC_CMAP = "viridis_r"
OCC_ZERO_COLOR = (0.85, 0.85, 0.85, 1.0)  # light grey for zero-count classes
DEFAULT_METADATA = REPO_ROOT / "data" / "rgb_metadata_extended.csv"

# test/test-od-class-<ClassName>-task-acc
_OOD_CLASS_RE = re.compile(r"^test/test-od-class-(.+)-task-acc$")


# --------------------------------------------------------------------------- #
# Metric extraction
# --------------------------------------------------------------------------- #
def worst_region(metrics: dict[str, float]) -> str | None:
    """Region with the lowest OOD top-1 task accuracy, or ``None`` if unavailable."""
    region_accs = {
        r: metrics[f"{OOD_PREFIX}-region-{r.lower()}-task-acc"]
        for r in DOMAIN_NAMES
        if f"{OOD_PREFIX}-region-{r.lower()}-task-acc" in metrics
    }
    if not region_accs:
        return None
    return min(region_accs, key=region_accs.get)


def ood_class_accs(metrics: dict[str, float]) -> dict[str, float]:
    """Per-class overall OOD top-1 accuracy keyed by FMoW class name."""
    out: dict[str, float] = {}
    for k, v in metrics.items():
        m = _OOD_CLASS_RE.match(k)
        if m:
            out[m.group(1)] = v
    return out


def region_class_accs(metrics: dict[str, float], region: str) -> dict[str, float]:
    """Per-class OOD top-1 accuracy within ``region`` keyed by FMoW class name."""
    prefix = f"{OOD_PREFIX}-region-{region.lower()}-class-"
    suffix = "-task-acc"
    out: dict[str, float] = {}
    for k, v in metrics.items():
        if k.startswith(prefix) and k.endswith(suffix):
            out[k[len(prefix):-len(suffix)]] = v
    return out


# --------------------------------------------------------------------------- #
# Class occurrence (from metadata)
# --------------------------------------------------------------------------- #
def load_test_class_counts(
    metadata_path: Path,
) -> tuple[dict[str, int], dict[str, dict[str, int]]]:
    """Per-class sample counts on the OOD test split, overall and per region.

    Returns ``(overall, per_region)`` where ``overall`` maps class name -> count
    and ``per_region`` maps region name -> {class name -> count} for the five named
    regions. Counts are taken from rows with ``split == 'test'`` (the OOD test split
    that the ``test-od`` loader evaluates). Returns empty dicts if the file is absent.
    """
    if not metadata_path.exists():
        print(f"Warning: metadata not found at {metadata_path}; skipping weighted plots.", file=sys.stderr)
        return {}, {}

    df = pd.read_csv(metadata_path, usecols=["split", "region", "category"])
    test = df[df["split"] == "test"]
    overall = {str(k): int(v) for k, v in test["category"].value_counts().items()}
    per_region: dict[str, dict[str, int]] = {}
    for rid, name in enumerate(DOMAIN_NAMES):
        sub = test[test["region"] == rid]
        per_region[name] = {str(k): int(v) for k, v in sub["category"].value_counts().items()}
    return overall, per_region


# --------------------------------------------------------------------------- #
# Deltas
# --------------------------------------------------------------------------- #
def all_deltas(baseline: dict[str, float], run: dict[str, float]) -> list[tuple[str, float]]:
    """Per-class deltas (run - baseline) for every class in both, sorted descending."""
    common = set(baseline) & set(run)
    return sorted(((c, run[c] - baseline[c]) for c in common), key=lambda kv: kv[1], reverse=True)


def top_deltas(
    baseline: dict[str, float], run: dict[str, float], n: int = TOP_N
) -> tuple[list[tuple[str, float]], list[tuple[str, float]]]:
    """Return (gains, losses): the n largest positive and n largest negative
    per-class deltas (run - baseline) over classes present in both."""
    ordered = all_deltas(baseline, run)  # descending
    gains = [cd for cd in ordered if cd[1] > 0][:n]
    losses = [cd for cd in reversed(ordered) if cd[1] < 0][:n]
    return gains, losses


def weighted_deltas(
    baseline: dict[str, float], run: dict[str, float], counts: dict[str, int]
) -> list[tuple[str, float, float]]:
    """Per-class occurrence-weighted accuracy deltas with standard errors.

    Returns ``(class, w, se)`` triples sorted by ``w`` descending, over classes
    present in baseline, run, and counts. ``w = (run_c - base_c) * n_c / N`` is
    the class's contribution to the overall accuracy delta (the values sum to it).

    ``se`` is the standard error of ``w``. The per-class delta is a difference of
    two binomial proportions on the same ``n_c`` samples; lacking per-image
    predictions we use the independent approximation
    ``Var = [p_r(1-p_r) + p_b(1-p_b)] / n_c`` (an upper bound -- it ignores the
    positive correlation between the two models, which a paired/McNemar estimate
    would exploit) and scale by ``n_c / N``.
    """
    common = set(baseline) & set(run) & set(counts)
    total = sum(counts[c] for c in common)
    if total <= 0:
        return []
    out: list[tuple[str, float, float]] = []
    for c in common:
        n = counts[c]
        frac = n / total
        pr, pb = run[c], baseline[c]
        w = (pr - pb) * frac
        var_delta = (pr * (1 - pr) + pb * (1 - pb)) / n if n > 0 else 0.0
        se = (var_delta**0.5) * frac
        out.append((c, w, se))
    return sorted(out, key=lambda t: t[1], reverse=True)


def pareto_weighted(
    items: list[tuple[str, float, float]], coverage: float = COVERAGE
) -> list[tuple[str, float, float]]:
    """Keep the classes covering ``coverage`` of the gains and of the losses.

    From ``(class, w, se)`` triples, selects the largest positive contributors
    whose cumulative ``w`` reaches ``coverage`` of the total positive
    contribution, plus the largest negative contributors covering ``coverage`` of
    the total negative magnitude. The negligible middle is dropped. Returned
    sorted by ``w`` descending.
    """
    def _cover(side: list[tuple[str, float, float]]) -> list[tuple[str, float, float]]:
        tot = sum(abs(t[1]) for t in side)
        if tot <= 0:
            return []
        acc, kept = 0.0, []
        for t in side:
            kept.append(t)
            acc += abs(t[1])
            if acc >= coverage * tot:
                break
        return kept

    pos = sorted((t for t in items if t[1] > 0), key=lambda t: t[1], reverse=True)
    neg = sorted((t for t in items if t[1] < 0), key=lambda t: t[1])  # most negative first
    selected = _cover(pos) + _cover(neg)
    return sorted(selected, key=lambda t: t[1], reverse=True)


def weighted_accs(
    accs: dict[str, float], counts: dict[str, int]
) -> list[tuple[str, float]]:
    """Per-class accuracies weighted by class frequency, sorted descending.

    Each value is ``acc_c * n_c / N`` over classes present in both ``accs`` and
    ``counts`` -- the class's contribution to the overall accuracy. The values
    therefore sum to that overall (or region) accuracy.
    """
    common = set(accs) & set(counts)
    total = sum(counts[c] for c in common)
    if total <= 0:
        return []
    items = [(c, accs[c] * counts[c] / total) for c in common]
    return sorted(items, key=lambda kv: kv[1], reverse=True)


def occurrence_colors(
    items: list[tuple[str, float]], counts: dict[str, int]
) -> tuple[list, mcolors.LogNorm | None]:
    """Per-bar colors keyed by each class's occurrence fraction (log scale).

    Darker = larger ``n_c / N`` (more trustworthy contribution), lighter = rarer;
    zero-count classes are light grey. Returns ``(colors, norm)`` aligned with
    ``items``; ``norm`` (for a colorbar) is ``None`` if no class has a positive
    count. The color scale spans the *full* class-count distribution (not just
    the shown ``items``), so a class keeps the same color across full, filtered,
    and top-5 plots.
    """
    total = sum(counts.values())
    if total <= 0:
        return [OCC_ZERO_COLOR] * len(items), None
    all_fracs = [v / total for v in counts.values() if v > 0]
    if not all_fracs:
        return [OCC_ZERO_COLOR] * len(items), None
    norm = mcolors.LogNorm(vmin=min(all_fracs), vmax=max(all_fracs))
    cmap = plt.get_cmap(OCC_CMAP)
    colors = []
    for c, _ in items:
        f = counts.get(c, 0) / total
        colors.append(cmap(norm(f)) if f > 0 else OCC_ZERO_COLOR)
    return colors, norm


def prettify_class(name: str) -> str:
    pretty = name.replace("_", " ").title()
    cid = CLASS_IDS.get(name)
    return f"{pretty} ({cid})" if cid is not None else pretty


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #
def plot_bars(
    items: list[tuple[str, float]],
    title: str,
    subtitle: str,
    xlabel: str,
    out_path: Path,
    annotate: bool,
    bar_colors: list | None = None,
    cbar: tuple[mcolors.LogNorm, str] | None = None,
    errors: list[float] | None = None,
) -> bool:
    """Diverging horizontal bar chart of per-class values (already in fractions).

    Values are rendered in percentage points. ``items`` must be pre-sorted; the
    first item is drawn at the top. By default bars are colored by sign (green
    gain / red loss); pass ``bar_colors`` (aligned with ``items``) to override,
    and ``cbar=(norm, label)`` to add a matching colorbar. ``errors`` (aligned
    with ``items``, same fraction units) draws symmetric horizontal error bars.
    Returns False (and writes nothing) if empty.
    """
    if not items:
        return False
    labels = [prettify_class(c) for c, _ in items]
    values = [v * 100.0 for _, v in items]  # percentage points
    colors = bar_colors if bar_colors is not None else [
        GAIN_COLOR if v >= 0 else LOSS_COLOR for v in values
    ]
    xerr = [e * 100.0 for e in errors] if errors is not None else None

    per_row = 0.32 if annotate else 0.20
    fig, ax = plt.subplots(figsize=(8.0, max(2.0, per_row * len(items) + 1.6)))
    y = list(range(len(items)))
    ax.barh(
        y, values, color=colors, xerr=xerr,
        error_kw={"elinewidth": 0.6, "capsize": 2, "ecolor": "#333333"},
    )
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8 if annotate else 6)
    ax.invert_yaxis()  # largest value on top
    ax.axvline(0.0, color="black", linewidth=0.8)
    ax.set_xlabel(xlabel)
    ax.set_title(f"{title}\n{subtitle}", fontsize=10)

    if annotate:
        pad = max((abs(v) for v in values), default=1.0) * 0.02 + 0.05
        for yi, v in zip(y, values):
            ax.text(
                v + (pad if v >= 0 else -pad), yi, f"{v:+.1f}",
                va="center", ha="left" if v >= 0 else "right", fontsize=8,
            )
    ax.margins(x=0.15)
    if cbar is not None:
        norm, label = cbar
        sm = plt.cm.ScalarMappable(norm=norm, cmap=plt.get_cmap(OCC_CMAP))
        sm.set_array([])
        fig.colorbar(sm, ax=ax, pad=0.02, label=label)
    fig.tight_layout()
    fig.savefig(out_path, format="svg")
    plt.close(fig)
    return True


def sorted_accs(accs: dict[str, float]) -> list[tuple[str, float]]:
    """Class accuracies as ``(class, acc)`` pairs sorted by accuracy descending."""
    return sorted(accs.items(), key=lambda kv: kv[1], reverse=True)


def plot_abs_bars(
    items: list[tuple[str, float]],
    title: str,
    subtitle: str,
    xlabel: str,
    out_path: Path,
    fixed_xlim: bool = True,
) -> bool:
    """Horizontal bar chart of absolute per-class accuracies (fractions -> pp).

    ``items`` must be pre-sorted; the first item is drawn at the top. With
    ``fixed_xlim`` the x-axis spans 0-100% (for plain accuracies); pass False to
    autoscale (for the small occurrence-weighted contributions). Returns False
    (and writes nothing) if empty.
    """
    if not items:
        return False
    labels = [prettify_class(c) for c, _ in items]
    values = [v * 100.0 for _, v in items]  # percent
    fig, ax = plt.subplots(figsize=(8.0, max(2.0, 0.20 * len(items) + 1.6)))
    y = list(range(len(items)))
    ax.barh(y, values, color=ACC_COLOR)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=6)
    ax.invert_yaxis()  # largest value on top
    ax.set_xlabel(xlabel)
    if fixed_xlim:
        ax.set_xlim(0.0, 100.0)
    else:
        ax.margins(x=0.05)
    ax.set_title(f"{title}\n{subtitle}", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, format="svg")
    plt.close(fig)
    return True


def plot_acc_vs_count(
    accs: dict[str, float],
    counts: dict[str, int],
    title: str,
    subtitle: str,
    out_path: Path,
) -> bool:
    """Scatter of per-class accuracy (y, %) against sample count (x, log).

    One point per class present in both ``accs`` and ``counts``, annotated with
    its integer class id. Sets each class's accuracy in perspective of how many
    samples back it (low-count classes have noisy accuracy). Returns False (and
    writes nothing) if there is nothing to plot.
    """
    common = [c for c in (set(accs) & set(counts)) if counts[c] > 0]
    if not common:
        return False
    xs = [counts[c] for c in common]
    ys = [accs[c] * 100.0 for c in common]
    fig, ax = plt.subplots(figsize=(8.0, 6.0))
    ax.scatter(xs, ys, color=ACC_COLOR, s=20, zorder=3)
    for c, x, y in zip(common, xs, ys):
        cid = CLASS_IDS.get(c)
        if cid is not None:
            ax.annotate(
                str(cid), (x, y), textcoords="offset points", xytext=(3, 3),
                fontsize=6, color="#333333",
            )
    ax.set_xscale("log")
    ax.set_xlabel("Samples per class (n, log scale)")
    ax.set_ylabel("Top-1 accuracy (%)")
    ax.set_ylim(0.0, 100.0)
    ax.grid(True, which="both", linewidth=0.3, alpha=0.5)
    ax.set_title(f"{title}\n{subtitle}", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, format="svg")
    plt.close(fig)
    return True


def emit_abs(
    figures_dir: Path,
    exp_key: str,
    label: str,
    scope_key: str,
    scope_label: str,
    accs: dict[str, float],
    counts: dict[str, int],
) -> None:
    """Write the absolute, occurrence-weighted, and accuracy-vs-count plots."""
    stem = f"classacc_{exp_key}_{scope_key}"
    if plot_abs_bars(
        sorted_accs(accs), label,
        f"{scope_label}: per-class top-1 accuracy (all classes)",
        "Top-1 accuracy (%)", figures_dir / f"{stem}.svg",
    ):
        print(f"  wrote {stem}.svg")

    if counts:
        if plot_abs_bars(
            weighted_accs(accs, counts), label,
            f"{scope_label}: occurrence-weighted contribution to accuracy",
            "Contribution to overall accuracy (pp)",
            figures_dir / f"{stem}_weighted.svg", fixed_xlim=False,
        ):
            print(f"  wrote {stem}_weighted.svg")

        if plot_acc_vs_count(
            accs, counts, label,
            f"{scope_label}: per-class accuracy vs. sample count (labels = class id)",
            figures_dir / f"{stem}_scatter.svg",
        ):
            print(f"  wrote {stem}_scatter.svg")


def emit_setting(
    figures_dir: Path,
    exp_key: str,
    run_label: str,
    base_name: str,
    scope_key: str,
    scope_label: str,
    base_accs: dict[str, float],
    run_accs: dict[str, float],
    counts: dict[str, int],
) -> None:
    """Write the top-5, all-class, and occurrence-weighted plots for one setting."""
    head = f"{run_label} vs. {base_name}"
    delta_xlabel = "Top-1 accuracy delta vs. baseline (pp)"
    stem = f"classdiff_{exp_key}_{scope_key}"

    gains, losses = top_deltas(base_accs, run_accs)
    if plot_bars(
        sorted(gains + losses, key=lambda kv: kv[1], reverse=True),
        head, f"{scope_label}: largest per-class accuracy gains and losses",
        delta_xlabel, figures_dir / f"{stem}_top5.svg", annotate=True,
    ):
        print(f"  wrote {stem}_top5.svg")

    if plot_bars(
        all_deltas(base_accs, run_accs),
        head, f"{scope_label}: per-class accuracy gains and losses (all classes)",
        delta_xlabel, figures_dir / f"{stem}_all.svg", annotate=False,
    ):
        print(f"  wrote {stem}_all.svg")

    if counts:
        triples = weighted_deltas(base_accs, run_accs, counts)
        weighted_xlabel = "Contribution to accuracy delta vs. baseline (pp)"
        cbar_label = "Class occurrence fraction (log)"

        def _weighted_plot(rows, subtitle, out_name):
            items = [(c, w) for c, w, _ in rows]
            errs = [se for _, _, se in rows]
            colors, norm = occurrence_colors(items, counts)
            if plot_bars(
                items, head, subtitle, weighted_xlabel,
                figures_dir / out_name, annotate=False, bar_colors=colors,
                cbar=(norm, cbar_label) if norm is not None else None, errors=errs,
            ):
                print(f"  wrote {out_name}")

        # All classes.
        _weighted_plot(
            triples, f"{scope_label}: occurrence-weighted contribution to accuracy delta",
            f"{stem}_weighted.svg",
        )
        # Filtered: classes covering the top 80% of gains and bottom 80% of losses.
        _weighted_plot(
            pareto_weighted(triples),
            f"{scope_label}: top-{COVERAGE:.0%} gains & bottom-{COVERAGE:.0%} losses "
            "(occurrence-weighted)",
            f"{stem}_weighted_filtered.svg",
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Per-class accuracy gain/loss plots")
    parser.add_argument(
        "eval_yaml",
        nargs="?",
        default=str(EVAL_CONFIG_DIR / "feature_fusion.yaml"),
        help="Path to eval YAML (default: src/train/configs/eval/feature_fusion.yaml)",
    )
    parser.add_argument(
        "--metadata",
        default=str(DEFAULT_METADATA),
        help="FMoW metadata CSV for class occurrence counts "
             "(default: data/rgb_metadata_extended.csv)",
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
    class_eval = cfg.get("class-eval")
    if not class_eval:
        print(f"No 'class-eval' section in {eval_yaml.name}", file=sys.stderr)
        sys.exit(1)

    baseline_ref: str = class_eval["baseline"]
    run_refs: list[str] = list(class_eval["runs"])

    all_refs = [baseline_ref] + run_refs
    config_names = {parse_run_ref(ref, run_name)[0] for ref in all_refs}
    run_configs = load_run_configs(config_names)
    run_experiments = resolve_experiments([{"runs": all_refs}], run_configs, run_name)
    translations = load_translations()

    # Class occurrence counts on the OOD test split (same for every run).
    counts_overall, counts_region = load_test_class_counts(Path(args.metadata))

    # Baseline reference metrics + worst region.
    _, base_key = parse_run_ref(baseline_ref, run_name)
    base_dir = find_run_dir(base_key)
    base_metrics = load_run_metrics(base_dir)
    base_ood = ood_class_accs(base_metrics)
    base_name = format_experiment_name(baseline_ref, run_experiments, translations)

    if not base_ood:
        print(
            f"Baseline '{baseline_ref}' has no per-class OOD metrics "
            f"(need metrics_rerun.csv at {base_dir}); aborting.",
            file=sys.stderr,
        )
        sys.exit(1)

    wr = worst_region(base_metrics)
    base_region = region_class_accs(base_metrics, wr) if wr else {}
    if wr is None:
        print("Warning: no per-region OOD accuracy for baseline; skipping worst-region plots.", file=sys.stderr)

    figures_dir = REPO_ROOT / "figures" / run_name
    figures_dir.mkdir(parents=True, exist_ok=True)
    print(f"Baseline: {base_name}  (worst OOD region: {wr})")
    print(f"Writing figures to {figures_dir}")

    # Baseline's own absolute per-class accuracies, plotted once.
    emit_abs(figures_dir, base_key, base_name, "test-od", "Test-OOD", base_ood, counts_overall)
    if wr is not None and base_region:
        emit_abs(
            figures_dir, base_key, base_name, wr.lower(), f"Worst region ({wr})",
            base_region, counts_region.get(wr, {}),
        )

    for ref in run_refs:
        _, exp_key = parse_run_ref(ref, run_name)
        run_dir = find_run_dir(exp_key)
        run_metrics = load_run_metrics(run_dir)
        run_label = format_experiment_name(ref, run_experiments, translations)
        run_ood = ood_class_accs(run_metrics)

        if not run_ood:
            print(f"  skip {ref}: no per-class OOD metrics (need metrics_rerun.csv).")
            continue

        # Overall Test-OOD setting.
        emit_abs(figures_dir, exp_key, run_label, "test-od", "Test-OOD", run_ood, counts_overall)
        emit_setting(
            figures_dir, exp_key, run_label, base_name,
            "test-od", "Test-OOD", base_ood, run_ood, counts_overall,
        )

        # Worst-region setting (baseline's worst region).
        if wr is None or not base_region:
            continue
        run_region = region_class_accs(run_metrics, wr)
        if not run_region:
            print(f"  skip {ref} worst-region plots: no per-class metrics for region {wr}.")
            continue
        emit_abs(
            figures_dir, exp_key, run_label, wr.lower(), f"Worst region ({wr})",
            run_region, counts_region.get(wr, {}),
        )
        emit_setting(
            figures_dir, exp_key, run_label, base_name,
            wr.lower(), f"Worst region ({wr})", base_region, run_region,
            counts_region.get(wr, {}),
        )


if __name__ == "__main__":
    main()
