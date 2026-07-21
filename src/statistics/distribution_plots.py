#!/usr/bin/env python3
"""Grouped bar plot of how each region's share of samples changes across splits.

One bar group per region, one bar per WILDS split, height = that region's
percentage of the split's samples. This makes the FMoW-WILDS domain shift
visible: the three in-distribution splits share a composition (Europe ~45%,
Africa ~2%), while in OOD-Test Europe shrinks to 27% and Africa -- the WRA
region -- grows to 12%, so OOD-Test is not just a later time window but a
differently composed one.

Splits are reconstructed from the raw CSV split column plus the timestamp window,
the same way ``average_class_extent.py`` and ``fmow_wilds_statistics.ipynb`` do:
    train    = CSV 'train', 2002-2012
    id_val   = CSV 'val',   2002-2012
    id_test  = CSV 'test',  2002-2012
    ood_val  = CSV 'val',   2013-2015
    ood_test = CSV 'test',  2016-2017
The FMoW 'seq' split and rows outside these windows are not shown.

Functions:
    `region_shares`: Per-region percentage of each split, as a table.
    `plot_region_shares`: Render and save the grouped bar figure.
    `plot_split_legend`: Render and save the split legend as its own figure.
    `main`: CLI entrypoint.

Usage:
    uv run python distribution_plots.py                 # write the SVG
    uv run python distribution_plots.py --counts        # also print raw counts
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

# rgb_metadata_extended.csv lives at <repo>/data/fmow_landsat/ ; this file is at
# <repo>/src/statistics/ , so go up two levels to reach the repo root.
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_METADATA = REPO_ROOT / "data" / "fmow_landsat" / "rgb_metadata_extended.csv"
# Mirror the figure into the thesis repo's images dir, matching the per-figure
# subfolder layout the results plots use.
THESIS_IMAGES_DIR = Path.home() / "git" / "thesis" / "images"
FIGURE_SUBDIR = "dataset_statistics"

# WILDS region index -> name (verified against country_code_mapping.csv). Region 5
# ("Other") is left out of the plot: it is 0.05% of Train and 0.02% of the OOD
# splits, so it only adds an empty bar group.
REGION_NAMES = {0: "Asia", 1: "Europe", 2: "Africa", 3: "Americas", 4: "Oceania", 5: "Other"}
PLOT_REGIONS = ["Asia", "Europe", "Africa", "Americas", "Oceania"]

# (WILDS split key, display label) in dataset order; the CSV split column and the
# timestamp window below together define each one.
SPLITS = [
    ("train", "Train"),
    ("id_val", "ID Val"),
    ("id_test", "ID Test"),
    ("ood_val", "OOD Val"),
    ("ood_test", "OOD Test"),
]
# Cool for the three in-distribution splits, warm for the two OOD ones, so the
# ID/OOD break is visible before the labels are read; within each family the
# shade darkens with the split's size -- same palette as the decision-fusion
# figures.
SPLIT_COLORS = {
    "train": "#115fb0",
    "id_val": "#3f93c9",
    "id_test": "#a9cee7",
    "ood_val": "#e8701a",
    "ood_test": "#cf440a",
}

# Time-window boundaries (UTC) that partition FMoW into the ID / OOD-Val / OOD-Test
# periods, matching the WILDS split construction in fmow_wilds_statistics.ipynb.
_ID_START = pd.Timestamp("2002-01-01", tz="UTC")
_ID_END = pd.Timestamp("2013-01-01", tz="UTC")        # ID period is [2002, 2013)
_OOD_VAL_END = pd.Timestamp("2016-01-01", tz="UTC")   # OOD-Val is [2013, 2016)
_OOD_TEST_END = pd.Timestamp("2018-01-01", tz="UTC")  # OOD-Test is [2016, 2018)


def save_figure(fig, out_dir: Path, filename: str, **savefig_kwargs) -> None:
    """Write `fig` to `out_dir` and mirror it into the thesis images dir when
    the thesis repo is present.

    Args:
        fig (matplotlib.figure.Figure): Figure to save.
        out_dir (Path): Destination directory in this repo; created if missing.
        filename (str): Output file name, including extension.
        **savefig_kwargs: Extra keyword arguments forwarded to ``fig.savefig``
            (in addition to ``format="svg"``).

    Returns:
        None
    """
    targets = [out_dir]
    if THESIS_IMAGES_DIR.parent.exists():
        targets.append(THESIS_IMAGES_DIR / FIGURE_SUBDIR)
    for directory in targets:
        directory.mkdir(parents=True, exist_ok=True)
        out_path = directory / filename
        fig.savefig(out_path, format="svg", **savefig_kwargs)
        print(f"  wrote {out_path}")


def region_shares(metadata_path: Path = DEFAULT_METADATA) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-region share (%) and raw sample count for the train / OOD-Val / OOD-Test splits.

    Args:
        metadata_path (Path): Path to `rgb_metadata_extended.csv`. Defaults to
            `DEFAULT_METADATA`.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: `(shares, counts)`, both indexed by
        region name (`PLOT_REGIONS` order) with one column per split key in
        `SPLITS` order. `shares` holds each region's percentage of its split's
        samples (columns sum to ~100 minus the omitted "Other" region), `counts`
        the underlying sample counts.
    """
    df = pd.read_csv(metadata_path, usecols=["split", "timestamp", "region"])
    ts = pd.to_datetime(df["timestamp"], utc=True, format="%Y-%m-%dT%H:%M:%SZ")
    split = df["split"]

    id_period = (ts >= _ID_START) & (ts < _ID_END)
    wilds_split = pd.Series(pd.NA, index=df.index, dtype="object")
    wilds_split[(split == "train") & id_period] = "train"
    wilds_split[(split == "val") & id_period] = "id_val"
    wilds_split[(split == "test") & id_period] = "id_test"
    wilds_split[(split == "val") & (ts >= _ID_END) & (ts < _OOD_VAL_END)] = "ood_val"
    wilds_split[(split == "test") & (ts >= _OOD_VAL_END) & (ts < _OOD_TEST_END)] = "ood_test"
    df["wilds_split"] = wilds_split

    counts = pd.crosstab(df["region"], df["wilds_split"])
    counts.index = [REGION_NAMES[i] for i in counts.index]
    # Percentages are of the whole split, so normalize before dropping "Other".
    shares = counts / counts.sum() * 100.0

    keys = [key for key, _ in SPLITS]
    return shares.loc[PLOT_REGIONS, keys], counts.loc[PLOT_REGIONS, keys]


def plot_region_shares(shares: pd.DataFrame, out_dir: Path, label_size: int = 16,
                       tick_size: int = 14) -> None:
    """Draw the grouped region-share bars and save them as an SVG.

    No in-figure legend: the split colors are the same encoding the standalone
    `plot_split_legend` figure renders, so the thesis places that legend
    alongside the plot instead.

    Args:
        shares (pd.DataFrame): Share table from `region_shares`, indexed by
            region with one column per split key.
        out_dir (Path): Directory to write `region_distribution.svg` to; created
            if missing. The figure is also mirrored into
            `THESIS_IMAGES_DIR/FIGURE_SUBDIR` when the thesis repo is present.
        label_size (int): Font size for the y-axis label.
        tick_size (int): Font size for the axis tick labels.

    Returns:
        None
    """
    regions = list(shares.index)
    bar_width = 0.8 / len(SPLITS)
    x_base = np.arange(len(regions))

    fig, ax = plt.subplots(figsize=(2.2 * len(regions) + 1.5, 5.5))
    for si, (key, label) in enumerate(SPLITS):
        values = shares[key].to_numpy()
        xpos = x_base + (si - (len(SPLITS) - 1) / 2) * bar_width
        ax.bar(xpos, values, width=bar_width, label=label,
               color=SPLIT_COLORS[key], edgecolor="black", linewidth=0.6)
        # Regions differ by more than an order of magnitude (Africa ~2% vs. Europe
        # ~45%), so annotate the short bars rather than leave them unreadable.
        for x, v in zip(xpos, values):
            ax.text(x, v + 0.6, f"{v:.1f}", ha="center", va="bottom", fontsize=tick_size - 5)

    ax.set_xticks(x_base)
    ax.set_xticklabels(regions)
    ax.set_ylabel("Share of samples (%)", fontsize=label_size)
    ax.set_ylim(0, shares.to_numpy().max() * 1.15)
    ax.tick_params(axis="both", labelsize=tick_size)
    ax.set_axisbelow(True)

    fig.tight_layout()
    save_figure(fig, out_dir, "region_distribution.svg")
    plt.close(fig)


def plot_split_legend(out_dir: Path, legend_size: int = 14) -> None:
    """Standalone legend for the region-share bars: one swatch per WILDS split,
    laid out in a single horizontal row (``ncol`` equal to the handle count
    forces one row).

    Args:
        out_dir (Path): Directory to write `region_distribution_legend.svg` to;
            created if missing. Also mirrored into the thesis images dir.
        legend_size (int): Font size for the legend labels.

    Returns:
        None
    """
    handles = [
        Patch(facecolor=SPLIT_COLORS[key], edgecolor="black", linewidth=0.6, label=label)
        for key, label in SPLITS
    ]
    fig = plt.figure(figsize=(9.0, 0.5))
    fig.legend(handles=handles, loc="center", ncol=len(handles), fontsize=legend_size, frameon=False)
    save_figure(fig, out_dir, "region_distribution_legend.svg", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """CLI entrypoint: print the region-share table and write the grouped bar
    figure plus its standalone split legend.

    Returns:
        None
    """
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA,
                        help="path to rgb_metadata_extended.csv")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "figures" / FIGURE_SUBDIR,
                        help="directory for the output SVG")
    parser.add_argument("--counts", action="store_true",
                        help="also print the raw sample counts behind the shares")
    args = parser.parse_args()

    shares, counts = region_shares(args.metadata)
    print(shares.round(2).to_string())
    if args.counts:
        print()
        print(counts.to_string())
        print(f"\nsplit totals (plotted regions): {counts.sum().to_dict()}")
    plot_region_shares(shares, args.out_dir)
    plot_split_legend(args.out_dir)


if __name__ == "__main__":
    main()
