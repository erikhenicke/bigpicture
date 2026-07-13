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

The module is organized in four groups: metric extraction (``worst_region``,
``ood_class_accs``, ``region_class_accs``) pulls per-class accuracy dicts out of a
run's flat metrics dict; occurrence/delta helpers (``load_test_class_counts``,
``all_deltas``, ``top_deltas``, ``weighted_deltas``, ``pareto_weighted``,
``weighted_accs``, ``occurrence_norm``, ``occurrence_colors``) turn those accuracy
dicts plus class counts into the values/colors the charts render; low-level plotting
(``plot_colorbar``, ``prettify_class``, ``save_figure``, ``plot_bars``, ``sorted_accs``,
``plot_abs_bars``, ``plot_acc_vs_count``) draws and saves individual figures; and
``emit_abs``/``emit_setting`` combine the above into the full set of figures for one
run/scope, driven by ``main``, the CLI entry point.
"""

import argparse
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
    OOD_CLASS_RE,
    find_run_dir,
    format_experiment_name,
    load_run_configs,
    load_run_metrics,
    load_translations,
    parse_run_ref,
    resolve_experiments,
    class_accs,
    get_africa_class_acc
)

from models.utils import DOMAIN_NAMES, TASK_CLASSES

CLASS_COMPARISON_RUNS = [
    "densenet_baseline",
    "densenet_baseline_pe_freq_hr",
    "film_om_gauss_pe_freq",
    "d3g_detach_hr_om_gauss_pe_freq_hr",
    "densenet_le_film_l_40"
]

SINGLE_UNIT_RESIDENTIAL = "single-unit_residential"

# FMoW class name -> integer class id (index in the canonical class list).
CLASS_IDS = {name: i for i, name in enumerate(TASK_CLASSES)}

OOD_PREFIX = "test/test-od"
TOP_N = 5
COVERAGE = 0.7  # fraction of gains / losses the filtered weighted plot retains
# The filtered thesis plots are all rendered at the height of this many rows (the
# max element count across them), so every filtered figure is the same physical
# size; plots with fewer classes stretch their bars to fill it, and the thesis
# style parameters (row/bar height, etc.) describe this reference 13-row plot.
FILTERED_ROWS = 13
# Extra vertical breathing room above the first and below the last bar (in row
# units) for the fixed-height plots.
Y_MARGIN_ROWS = 0.5
# Cap how much a sparse fixed-height plot's bars may thicken relative to the
# reference FILTERED_ROWS plot; past this the surplus becomes wider gaps, not
# fatter bars.
MAX_BAR_STRETCH = 1.2
GAIN_COLOR = "#2ca02c"
LOSS_COLOR = "#d62728"
ACC_COLOR = "#1f77b4"
# Sequential colormap for occurrence: low fraction -> light, high -> dark.
OCC_CMAP = "viridis_r"
OCC_ZERO_COLOR = (0.85, 0.85, 0.85, 1.0)  # light grey for zero-count classes
DEFAULT_METADATA = REPO_ROOT / "data" / "rgb_metadata_extended.csv"
# Mirror every figure into the thesis repo's images dir, under a per-run subfolder
# matching the repo's ``figures/<run_name>/`` layout (see decision_plots.py).
THESIS_IMAGES_DIR = Path.home() / "git" / "thesis" / "images"


# --------------------------------------------------------------------------- #
# Print class accuracies for thesis 
# --------------------------------------------------------------------------- #
def print_class_accs_thesis():
    """Print class single-unit residentials in africa of best model settings
    """
    for run in CLASS_COMPARISON_RUNS:
        accs = get_africa_class_acc(run, SINGLE_UNIT_RESIDENTIAL)
        splits = accs.keys()
        print(f"{run}: ")
        for split in splits:
            print(f"\t{split}: {accs[split][0]:.3} ({accs[split][1]:.3f})")


# --------------------------------------------------------------------------- #
# Metric extraction
# --------------------------------------------------------------------------- #
def worst_region(metrics: dict[str, float]) -> str | None:
    """Region with the lowest OOD top-1 task accuracy, or ``None`` if unavailable.

    Args:
        metrics (dict[str, float]): Flat run metrics dict (as returned by
            `results.utils.load_run_metrics`), keyed by metric name.

    Returns:
        str | None: The `models.utils.DOMAIN_NAMES` entry with the lowest
            ``test/test-od-region-<region>-task-acc`` value, or None if none
            of those keys are present in `metrics`.
    """
    region_accs = {
        r: metrics[f"{OOD_PREFIX}-region-{r.lower()}-task-acc"]
        for r in DOMAIN_NAMES
        if f"{OOD_PREFIX}-region-{r.lower()}-task-acc" in metrics
    }
    if not region_accs:
        return None
    return min(region_accs, key=region_accs.get)


def region_class_accs(metrics: dict[str, float], region: str) -> dict[str, float]:
    """Per-class OOD top-1 accuracy within ``region`` keyed by FMoW class name.

    Args:
        metrics (dict[str, float]): Flat run metrics dict, keyed by metric
            name.
        region (str): Region name (one of `models.utils.DOMAIN_NAMES`,
            case-insensitive).

    Returns:
        dict[str, float]: Mapping of FMoW class name to top-1 accuracy
            (fraction in [0, 1]) within `region`, extracted from every
            ``test/test-od-region-<region>-class-<ClassName>-task-acc`` key
            in `metrics`.
    """
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

    Args:
        metadata_path (Path): Path to the FMoW metadata CSV (must have
            ``split``, ``region``, and ``category`` columns).

    Returns:
        tuple[dict[str, int], dict[str, dict[str, int]]]: ``(overall,
            per_region)``. `overall` maps FMoW class name to its sample
            count on the OOD test split. `per_region` maps each name in
            `models.utils.DOMAIN_NAMES` to a ``{class name: count}`` dict
            for that region's OOD test rows. Both are empty dicts if
            `metadata_path` does not exist.
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
    """Per-class deltas (run - baseline) for every class in both, sorted descending.

    Args:
        baseline (dict[str, float]): Baseline per-class accuracy, keyed by
            FMoW class name.
        run (dict[str, float]): Comparison run's per-class accuracy, keyed
            by FMoW class name.

    Returns:
        list[tuple[str, float]]: ``(class, run[class] - baseline[class])``
            pairs for every class present in both, sorted by delta
            descending.
    """
    common = set(baseline) & set(run)
    return sorted(((c, run[c] - baseline[c]) for c in common), key=lambda kv: kv[1], reverse=True)


def top_deltas(
    baseline: dict[str, float], run: dict[str, float], n: int = TOP_N
) -> tuple[list[tuple[str, float]], list[tuple[str, float]]]:
    """Return (gains, losses): the n largest positive and n largest negative
    per-class deltas (run - baseline) over classes present in both.

    Args:
        baseline (dict[str, float]): Baseline per-class accuracy, keyed by
            FMoW class name.
        run (dict[str, float]): Comparison run's per-class accuracy, keyed
            by FMoW class name.
        n (int): Maximum number of gains/losses to return each. Defaults to
            `TOP_N`.

    Returns:
        tuple[list[tuple[str, float]], list[tuple[str, float]]]: ``(gains,
            losses)`` -- up to `n` ``(class, delta)`` pairs with the largest
            positive deltas (`gains`, descending) and up to `n` pairs with
            the largest negative deltas (`losses`, most negative first).
    """
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

    Args:
        baseline (dict[str, float]): Baseline per-class accuracy (fraction
            in [0, 1]), keyed by FMoW class name.
        run (dict[str, float]): Comparison run's per-class accuracy
            (fraction in [0, 1]), keyed by FMoW class name.
        counts (dict[str, int]): Per-class sample counts, keyed by FMoW
            class name (e.g. the ``overall`` or one region's dict from
            `load_test_class_counts`).

    Returns:
        list[tuple[str, float, float]]: ``(class, w, se)`` triples, one per
            class present in `baseline`, `run`, and `counts`, sorted by `w`
            descending.
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

    Args:
        items (list[tuple[str, float, float]]): ``(class, w, se)`` triples,
            e.g. from `weighted_deltas`.
        coverage (float): Fraction (in [0, 1]) of the total positive (resp.
            negative) contribution magnitude to retain on each side.
            Defaults to `COVERAGE`.

    Returns:
        list[tuple[str, float, float]]: The selected ``(class, w, se)``
            triples (positive-`w` and negative-`w` sides each independently
            filtered to `coverage`), sorted by `w` descending.
    """
    def _cover(side: list[tuple[str, float, float]]) -> list[tuple[str, float, float]]:
        """Keep the largest-magnitude prefix of one side (all-positive or
        all-negative `w`) whose cumulative ``|w|`` reaches `coverage` of that
        side's total.

        Args:
            side (list[tuple[str, float, float]]): ``(class, w, se)``
                triples already sorted by descending contribution magnitude,
                all with `w` of the same sign.

        Returns:
            list[tuple[str, float, float]]: The retained prefix of `side`
                (empty if `side`'s total magnitude is 0).
        """
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

    Args:
        accs (dict[str, float]): Per-class accuracy (fraction in [0, 1]),
            keyed by FMoW class name.
        counts (dict[str, int]): Per-class sample counts, keyed by FMoW
            class name.

    Returns:
        list[tuple[str, float]]: ``(class, acc_c * n_c / N)`` pairs for
            classes present in both `accs` and `counts`, sorted by value
            descending. Empty if the total count is 0.
    """
    common = set(accs) & set(counts)
    total = sum(counts[c] for c in common)
    if total <= 0:
        return []
    items = [(c, accs[c] * counts[c] / total) for c in common]
    return sorted(items, key=lambda kv: kv[1], reverse=True)


def occurrence_norm(counts: dict[str, int]) -> mcolors.LogNorm | None:
    """LogNorm spanning the *full* positive class-fraction range of ``counts``
    (``n_c / N`` over every class, not just a plotted subset), or ``None`` if no
    class has a positive count. Shared by the bar colors and the standalone
    colorbar so a class keeps the same color across every plot of one scope.

    Args:
        counts (dict[str, int]): Per-class sample counts, keyed by FMoW
            class name.

    Returns:
        matplotlib.colors.LogNorm | None: Log-scale normalizer spanning
            ``[min, max]`` of ``n_c / N`` over classes with a positive
            count, or None if the total count is 0 or no class has a
            positive count.
    """
    total = sum(counts.values())
    if total <= 0:
        return None
    all_fracs = [v / total for v in counts.values() if v > 0]
    if not all_fracs:
        return None
    return mcolors.LogNorm(vmin=min(all_fracs), vmax=max(all_fracs))


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

    Args:
        items (list[tuple[str, float]]): ``(class, value)`` pairs to color,
            e.g. from `weighted_deltas` (with the `se` dropped) or
            `weighted_accs`.
        counts (dict[str, int]): Per-class sample counts (the full scope's
            distribution, not just the classes in `items`), keyed by FMoW
            class name.

    Returns:
        tuple[list, matplotlib.colors.LogNorm | None]: ``(colors, norm)`` --
            one RGBA color per entry of `items` (aligned, light grey for
            zero-count classes), and the `occurrence_norm` used to compute
            them (None if no class in `counts` has a positive count, in
            which case every color is the zero-count grey).
    """
    norm = occurrence_norm(counts)
    if norm is None:
        return [OCC_ZERO_COLOR] * len(items), None
    total = sum(counts.values())
    cmap = plt.get_cmap(OCC_CMAP)
    colors = []
    for c, _ in items:
        f = counts.get(c, 0) / total
        colors.append(cmap(norm(f)) if f > 0 else OCC_ZERO_COLOR)
    return colors, norm


def plot_colorbar(counts: dict[str, int], out_path: Path, label: str,
                  label_size: int = 8, tick_size: int = 7) -> bool:
    """Standalone horizontal colorbar for one scope's occurrence color scale.

    Uses the same :func:`occurrence_norm` as the bars, so it is the shared key for
    every weighted plot of that scope (which no longer carry their own colorbar).
    Returns False (and writes nothing) if the scope has no positive counts.

    Args:
        counts (dict[str, int]): Per-class sample counts for the scope,
            keyed by FMoW class name.
        out_path (Path): Output SVG path.
        label (str): Colorbar axis label.
        label_size (int): Font size for `label`. Defaults to 8.
        tick_size (int): Font size for the colorbar tick labels. Defaults
            to 7.

    Returns:
        bool: True if the colorbar was written, False if `counts` has no
            positive entries (nothing written).
    """
    norm = occurrence_norm(counts)
    if norm is None:
        return False
    fig = plt.figure(figsize=(6.0, 0.4))
    # Explicit cax so the bar's height (its thickness) is controlled directly --
    # a thin strip near the top, leaving room below for the ticks and label.
    cax = fig.add_axes([0.06, 0.4, 0.88, 0.12])
    sm = plt.cm.ScalarMappable(norm=norm, cmap=plt.get_cmap(OCC_CMAP))
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cb.set_label(label, fontsize=label_size, labelpad=7)
    cb.ax.xaxis.set_label_position("top")  # label above the bar; ticks stay below
    cb.ax.tick_params(labelsize=tick_size)
    # Crop to content so the label/ticks below the short bar aren't clipped
    # (the axes fills the tiny figure), mirroring the standalone legends.
    save_figure(fig, out_path, bbox_inches="tight")
    plt.close(fig)
    return True


# Wrap a class label onto two lines at its first space past this character index,
# so long multi-word labels (e.g. "Recreational Facility (46)") don't run wide.
# A short first token (e.g. "Airport (0)", space at index 7) stays on one line.
LABEL_WRAP_AFTER = 7


def prettify_class(name: str, include_id: bool = True) -> str:
    """Format an FMoW class name for display, optionally with its class id.

    Long multi-word labels are wrapped onto a second line after the first
    space past `LABEL_WRAP_AFTER` characters (see that constant).

    Args:
        name (str): Raw FMoW class name (e.g. ``"crop_field"``).
        include_id (bool): Whether to append the class's integer id (from
            `CLASS_IDS`) in parentheses, if known. Defaults to True.

    Returns:
        str: Title-cased, space-separated class name, optionally
            id-suffixed and wrapped onto two lines with an embedded newline.
    """
    pretty = name.replace("_", " ").title()
    cid = CLASS_IDS.get(name)
    label = f"{pretty} ({cid})" if (include_id and cid is not None) else pretty
    sp = label.find(" ", LABEL_WRAP_AFTER + 1)
    if sp != -1:
        label = f"{label[:sp]}\n{label[sp + 1:]}"
    return label


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #
def save_figure(fig, out_path: Path, **savefig_kwargs) -> None:
    """Write `fig` to `out_path` in the repo figures dir and mirror it into the
    thesis images dir under the same ``<run_name>/`` subfolder (``out_path``'s
    parent dir name) when the thesis repo is present -- matching decision_plots.py.

    Args:
        fig (matplotlib.figure.Figure): Figure to save.
        out_path (Path): Destination SVG path in this repo (its parent
            directory name is reused as the thesis-side subfolder).
        **savefig_kwargs: Extra keyword arguments forwarded to
            ``fig.savefig`` (in addition to ``format="svg"``).

    Returns:
        None
    """
    fig.savefig(out_path, format="svg", **savefig_kwargs)
    if THESIS_IMAGES_DIR.parent.exists():
        thesis_dir = THESIS_IMAGES_DIR / out_path.parent.name
        thesis_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(thesis_dir / out_path.name, format="svg", **savefig_kwargs)


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
    show_title: bool = True,
    ylabel_size: int | None = None,
    xlabel_size: int | None = None,
    xtick_size: int | None = None,
    label_bold: bool = False,
    show_class_id: bool = True,
    row_height: float | None = None,
    bar_height: float | None = None,
    fig_width: float = 8.0,
    fixed_rows: int | None = None,
) -> bool:
    """Diverging horizontal bar chart of per-class values (already in fractions).

    Values are rendered in percentage points. ``items`` must be pre-sorted; the
    first item is drawn at the top. By default bars are colored by sign (green
    gain / red loss); pass ``bar_colors`` (aligned with ``items``) to override,
    and ``cbar=(norm, label)`` to add a matching colorbar. ``errors`` (aligned
    with ``items``, same fraction units) draws symmetric horizontal error bars.
    Returns False (and writes nothing) if empty.

    The thesis-facing knobs override the defaults for a cleaner figure: pass
    ``show_title=False`` to drop the title/subtitle block, ``ylabel_size`` to
    enlarge the class labels, ``xlabel_size`` / ``xtick_size`` for the x-axis label
    and its tick numbers, ``row_height`` to space the classes further apart,
    ``bar_height`` to thicken the bars within each row, and a smaller ``fig_width``
    to narrow the bar column (giving the labels relatively more room).
    ``fixed_rows`` fixes the figure height to that many rows while the y-axis
    still fits exactly the bars present, so plots with fewer bars come out the
    same physical size but stretch their bars to fill it (rather than padding with
    blank slots). Plots with more bars than ``fixed_rows`` grow taller at the
    reference bar thickness.

    Args:
        items (list[tuple[str, float]]): Pre-sorted ``(class, value)``
            pairs, values as fractions (multiplied by 100 for display); the
            first item is drawn at the top.
        title (str): Figure title (shown only if `show_title`).
        subtitle (str): Figure subtitle, shown below `title`.
        xlabel (str): X-axis label.
        out_path (Path): Output SVG path.
        annotate (bool): Whether to print each bar's signed value next to
            it, and to use the wider default row spacing/label font size
            associated with annotated plots.
        bar_colors (list | None): Per-bar colors aligned with `items`; if
            None, bars are colored green (>= 0) / red (< 0).
        cbar (tuple[matplotlib.colors.LogNorm, str] | None): Optional
            ``(norm, label)`` to draw a colorbar matching `bar_colors`.
        errors (list[float] | None): Per-bar symmetric error magnitudes
            (fractions, aligned with `items`) for horizontal error bars, or
            None to omit them.
        show_title (bool): Whether to draw the title/subtitle block.
            Defaults to True.
        ylabel_size (int | None): Font size for the class y-tick labels;
            falls back to 8 (annotate) / 6 (not annotate) if None.
        xlabel_size (int | None): Font size for `xlabel`; falls back to 8
            (annotate) / 6 (not annotate) if None.
        xtick_size (int | None): Font size for the x-axis tick numbers; left
            at the matplotlib default if None.
        label_bold (bool): Whether the y-tick and x-axis labels are bold.
            Defaults to False.
        show_class_id (bool): Whether `prettify_class` appends the class id.
            Defaults to True.
        row_height (float | None): Vertical space per class row (figure
            inches); falls back to 0.32 (annotate) / 0.20 (not annotate) if
            None.
        bar_height (float | None): Bar thickness as a fraction of a row
            (``barh``'s ``height``); uses matplotlib's default if None.
        fig_width (float): Figure width in inches. Defaults to 8.0.
        fixed_rows (int | None): If set, the figure height is sized for at
            least this many rows (never fewer than ``len(items)``), and bars
            with fewer than `fixed_rows` items stretch to fill it (capped by
            `MAX_BAR_STRETCH`) instead of leaving blank space. If None, the
            height is sized exactly to ``len(items)`` rows.

    Returns:
        bool: True if the figure was written, False if `items` is empty
            (nothing written).
    """
    if not items:
        return False
    labels = [prettify_class(c, include_id=show_class_id) for c, _ in items]
    values = [v * 100.0 for _, v in items]  # percentage points
    colors = bar_colors if bar_colors is not None else [
        GAIN_COLOR if v >= 0 else LOSS_COLOR for v in values
    ]
    xerr = [e * 100.0 for e in errors] if errors is not None else None

    per_row = row_height if row_height is not None else (0.32 if annotate else 0.20)
    # The figure height is sized for a fixed row count when requested (never fewer
    # than the actual bars), so every plot with <= fixed_rows bars comes out the
    # same height regardless of how many classes it shows.
    n_rows = max(fixed_rows, len(items)) if fixed_rows is not None else len(items)
    fig, ax = plt.subplots(figsize=(fig_width, max(2.0, per_row * n_rows + 1.6)))
    y = list(range(len(items)))
    # Bar thickness (fraction of a row). With a fixed figure height (fixed_rows) a
    # sparse plot's rows stretch; cap that so bars grow to at most MAX_BAR_STRETCH x
    # their thickness in the reference fixed_rows plot, the surplus becoming wider
    # gaps rather than fatter bars.
    bh = bar_height
    if bar_height is not None and fixed_rows is not None and items:
        stretch = (fixed_rows + 2 * Y_MARGIN_ROWS) / (len(items) + 2 * Y_MARGIN_ROWS)
        if stretch > MAX_BAR_STRETCH:
            bh = bar_height * MAX_BAR_STRETCH / stretch
    barh_kw = {} if bh is None else {"height": bh}
    ax.barh(
        y, values, color=colors, xerr=xerr, **barh_kw,
        error_kw={"elinewidth": 1.2, "capsize": 4, "ecolor": "#333333"},
    )
    ax.set_yticks(y)
    ax.set_yticklabels(
        labels,
        fontsize=ylabel_size if ylabel_size is not None else (8 if annotate else 6),
        fontweight="bold" if label_bold else "normal",
    )
    if xtick_size is not None:
        ax.tick_params(axis="x", labelsize=xtick_size)
    ax.invert_yaxis()  # largest value on top
    if fixed_rows is not None:
        # Figure height is fixed (n_rows) but the y-axis fits the bars present plus
        # Y_MARGIN_ROWS of breathing room top and bottom, so a plot with fewer than
        # fixed_rows bars stretches them to fill the height (up to the bar-thickness
        # cap above) instead of leaving blank slots at the bottom.
        ax.set_ylim(bottom=len(items) - 0.5 + Y_MARGIN_ROWS, top=-0.5 - Y_MARGIN_ROWS)
    ax.axvline(0.0, color="black", linewidth=0.8)
    ax.set_xlabel(
        xlabel,
        fontsize=xlabel_size if xlabel_size is not None else (8 if annotate else 6),
        fontweight="bold" if label_bold else "normal",
    )
    if show_title:
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
    save_figure(fig, out_path)
    plt.close(fig)
    return True


def sorted_accs(accs: dict[str, float]) -> list[tuple[str, float]]:
    """Class accuracies as ``(class, acc)`` pairs sorted by accuracy descending.

    Args:
        accs (dict[str, float]): Per-class accuracy (fraction in [0, 1]),
            keyed by FMoW class name.

    Returns:
        list[tuple[str, float]]: ``(class, acc)`` pairs sorted by `acc`
            descending.
    """
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

    Args:
        items (list[tuple[str, float]]): Pre-sorted ``(class, value)``
            pairs, values as fractions (multiplied by 100 for display); the
            first item is drawn at the top.
        title (str): Figure title.
        subtitle (str): Figure subtitle, shown below `title`.
        xlabel (str): X-axis label.
        out_path (Path): Output SVG path.
        fixed_xlim (bool): If True, fix the x-axis to ``[0, 100]``; if
            False, autoscale to the data with a small margin. Defaults to
            True.

    Returns:
        bool: True if the figure was written, False if `items` is empty
            (nothing written).
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
    save_figure(fig, out_path)
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

    Args:
        accs (dict[str, float]): Per-class accuracy (fraction in [0, 1]),
            keyed by FMoW class name.
        counts (dict[str, int]): Per-class sample counts, keyed by FMoW
            class name.
        title (str): Figure title.
        subtitle (str): Figure subtitle, shown below `title`.
        out_path (Path): Output SVG path.

    Returns:
        bool: True if the figure was written, False if no class has a
            positive count in both `accs` and `counts` (nothing written).
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
    save_figure(fig, out_path)
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
    """Write the absolute, occurrence-weighted, and accuracy-vs-count plots.

    Args:
        figures_dir (Path): Directory to write the output SVGs to.
        exp_key (str): Experiment key, used in output file names.
        label (str): Run display name, used as the figure title.
        scope_key (str): Scope identifier (e.g. ``"test-od"`` or a
            lowercased region name), used in output file names.
        scope_label (str): Human-readable scope label, used in figure
            subtitles.
        accs (dict[str, float]): Per-class accuracy (fraction in [0, 1]) for
            this run and scope, keyed by FMoW class name.
        counts (dict[str, int]): Per-class sample counts for this scope,
            keyed by FMoW class name. If empty, only the absolute-accuracy
            plot is written.

    Returns:
        None
    """
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
    all_figures: bool = False,
) -> None:
    """Write per-class delta plots for one setting.

    By default only the thesis-facing occurrence-weighted *filtered* plot is
    written. With ``all_figures`` the top-5, all-class, and unfiltered weighted
    plots are written too.

    Args:
        figures_dir (Path): Directory to write the output SVGs to.
        exp_key (str): Comparison run's experiment key, used in output file
            names.
        run_label (str): Comparison run's display name.
        base_name (str): Baseline run's display name, used in the "vs."
            title.
        scope_key (str): Scope identifier (e.g. ``"test-od"`` or a
            lowercased region name), used in output file names.
        scope_label (str): Human-readable scope label, used in figure
            subtitles.
        base_accs (dict[str, float]): Baseline per-class accuracy (fraction
            in [0, 1]) for this scope, keyed by FMoW class name.
        run_accs (dict[str, float]): Comparison run's per-class accuracy
            (fraction in [0, 1]) for this scope, keyed by FMoW class name.
        counts (dict[str, int]): Per-class sample counts for this scope,
            keyed by FMoW class name. If empty, the weighted plots are
            skipped.
        all_figures (bool): If True, also write the top-5, all-class, and
            unfiltered weighted plots (not just the filtered one). Defaults
            to False.

    Returns:
        None
    """
    head = f"{run_label} vs. {base_name}"
    delta_xlabel = "Top-1 accuracy delta vs. baseline (pp)"
    stem = f"classdiff_{exp_key}_{scope_key}"

    if all_figures:
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

        # Thesis-facing axis label: the worst-region plots read "OOD WRA" (worst-
        # region accuracy), the overall Test-OOD plot "OOD Acc.".
        metric_label = "OOD Acc." if scope_key == "test-od" else "OOD WRA"
        thesis_xlabel = rf"{metric_label} (%) vs. Baseline"
        # Drop the titles and enlarge/space the class labels, thicken the bars,
        # and narrow the bar column so the labels breathe (see plot_bars knobs).
        thesis_style = dict(
            show_title=False, ylabel_size=12, xlabel_size=16, xtick_size=14,
            label_bold=False, show_class_id=False,
            row_height=0.4, bar_height=0.75, fig_width=6.0, fixed_rows=FILTERED_ROWS,
        )

        def _weighted_plot(rows, subtitle, out_name, thesis=False):
            """Render one occurrence-weighted delta bar chart from `weighted_deltas` rows.

            Args:
                rows (list[tuple[str, float, float]]): ``(class, w, se)``
                    triples to plot, e.g. from `weighted_deltas` or
                    `pareto_weighted`.
                subtitle (str): Figure subtitle.
                out_name (str): Output SVG file name (within `figures_dir`).
                thesis (bool): If True, use the thesis-facing style
                    (`thesis_style`, `thesis_xlabel`, no inline colorbar); if
                    False, use the default style with an inline colorbar.
                    Defaults to False.

            Returns:
                None
            """
            items = [(c, w) for c, w, _ in rows]
            errs = [se for _, _, se in rows]
            colors, norm = occurrence_colors(items, counts)
            style = thesis_style if thesis else {}
            # Thesis filtered plots drop their inline colorbar in favor of the
            # standalone per-scope colorbar (see plot_colorbar); the --all
            # weighted plot keeps its own.
            cbar = None if thesis else ((norm, cbar_label) if norm is not None else None)
            if plot_bars(
                items, head, subtitle,
                thesis_xlabel if thesis else weighted_xlabel,
                figures_dir / out_name, annotate=False, bar_colors=colors,
                cbar=cbar, errors=errs,
                **style,
            ):
                print(f"  wrote {out_name}")

        # All classes (only with --all).
        if all_figures:
            _weighted_plot(
                triples, f"{scope_label}: occurrence-weighted contribution to accuracy delta",
                f"{stem}_weighted.svg",
            )
        # Filtered: classes covering the top 80% of gains and bottom 80% of losses.
        _weighted_plot(
            pareto_weighted(triples),
            f"{scope_label}: top-{COVERAGE:.0%} gains & bottom-{COVERAGE:.0%} losses "
            "(occurrence-weighted)",
            f"{stem}_weighted_filtered.svg", thesis=True,
        )


def main() -> None:
    """CLI entry point: load the eval YAML and write every per-class figure.

    Reads the ``class-eval`` section of the eval YAML given as the
    positional argument (default ``feature_fusion.yaml``), resolves the
    baseline and comparison runs' per-class metrics, loads class occurrence
    counts from ``--metadata``, and writes (to ``figures/<eval_name>/``) the
    baseline's worst-region lookup, per-scope colorbars, and -- for each
    comparison run -- the delta plots for the overall Test-OOD setting and
    the baseline's worst OOD region (`emit_setting`, plus `emit_abs` for
    absolute accuracies with ``--all``). Exits with status 1 if the eval
    YAML is missing, has no ``class-eval`` section, or the baseline has no
    per-class OOD metrics.

    Returns:
        None
    """
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
    parser.add_argument(
        "--all",
        action="store_true",
        help="Generate every per-class figure (absolute accuracy, scatter, top-5, "
             "all-class, unfiltered weighted). By default, only the thesis-facing "
             "occurrence-weighted filtered delta plots are written.",
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
    base_ood = class_accs(base_metrics, OOD_CLASS_RE)
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

    # One standalone horizontal colorbar per scope, shared by that scope's
    # weighted_filtered plots (which no longer carry their own colorbar).
    if plot_colorbar(counts_overall, figures_dir / "classdiff_test-od_colorbar.svg", "Log Class Fraction"):
        print("  wrote classdiff_test-od_colorbar.svg")
    if wr is not None and plot_colorbar(
        counts_region.get(wr, {}), figures_dir / f"classdiff_{wr.lower()}_colorbar.svg", "Log Class Fraction"
    ):
        print(f"  wrote classdiff_{wr.lower()}_colorbar.svg")

    # Baseline's own absolute per-class accuracies (only with --all).
    if args.all:
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
        run_ood = class_accs(run_metrics, OOD_CLASS_RE)

        if not run_ood:
            print(f"  skip {ref}: no per-class OOD metrics (need metrics_rerun.csv).")
            continue

        # Overall Test-OOD setting.
        if args.all:
            emit_abs(figures_dir, exp_key, run_label, "test-od", "Test-OOD", run_ood, counts_overall)
        emit_setting(
            figures_dir, exp_key, run_label, base_name,
            "test-od", "Test-OOD", base_ood, run_ood, counts_overall,
            all_figures=args.all,
        )

        # Worst-region setting (baseline's worst region).
        if wr is None or not base_region:
            continue
        run_region = region_class_accs(run_metrics, wr)
        if not run_region:
            print(f"  skip {ref} worst-region plots: no per-class metrics for region {wr}.")
            continue
        if args.all:
            emit_abs(
                figures_dir, exp_key, run_label, wr.lower(), f"Worst region ({wr})",
                run_region, counts_region.get(wr, {}),
            )
        emit_setting(
            figures_dir, exp_key, run_label, base_name,
            wr.lower(), f"Worst region ({wr})", base_region, run_region,
            counts_region.get(wr, {}),
            all_figures=args.all,
        )


if __name__ == "__main__":
    main()
