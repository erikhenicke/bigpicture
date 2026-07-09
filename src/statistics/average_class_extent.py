"""Compute the spatial extent statistics of an FMoW-WILDS class.

The spatial *extent* of a sample is the ground distance covered by the HR image
(`img_span_km` in the extended metadata). Reported per class: mean, median, min,
max and standard deviation (km), alongside the same statistics for all classes
pooled (within the selected split / region) for comparison.

Splits are the WILDS canonical splits, reconstructed from timestamp windows plus
the raw CSV split column exactly as ``fmow_wilds_statistics.ipynb`` does:
    train    = CSV 'train', 2002-2012
    id_val   = CSV 'val',   2002-2012
    id_test  = CSV 'test',  2002-2012
    ood_val  = CSV 'val',   2013-2015
    ood_test = CSV 'test',  2016-2017
``all`` (default) is the union of those five (the WILDS dataset); the FMoW 'seq'
split and any samples outside these windows are excluded.

Regions (WILDS index -> name): 0 Asia, 1 Europe, 2 Africa, 3 Americas,
4 Oceania, 5 Other. WRA (the thesis's primary metric) is OOD-Test on Africa,
i.e. ``--split ood_test --region africa``.

Functions:
    `assign_wilds_split`: Map metadata rows to their WILDS split from the
        raw split column + timestamp.
    `resolve_region`: Resolve a region argument (index or name) to
        `(index, label)`.
    `load_metadata`: Load, split-tag, filter and clean the extended metadata.
    `class_extent_stats`: Extent statistics for one class (and all classes
        pooled).
    `all_class_stats`: Extent statistics for every class, as a table.
    `_print_stats` / `main`: CLI presentation and entrypoint.

Usage:
    uv run python average_class_extent.py 57                          # one class, WILDS union
    uv run python average_class_extent.py 57 --split ood_test         # one class, OOD-Test
    uv run python average_class_extent.py 0 --region africa           # one class, Africa only
    uv run python average_class_extent.py --all --split train         # ranked table, train split
    uv run python average_class_extent.py --all --sort-n              # table sorted by sample count
"""

import argparse
from pathlib import Path

import pandas as pd

# rgb_metadata_extended.csv lives at <repo>/data/fmow_landsat/ ; this file is at
# <repo>/src/statistics/ , so go up two levels to reach the repo root.
DEFAULT_METADATA = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "fmow_landsat"
    / "rgb_metadata_extended.csv"
)

NUM_CLASSES = 62

# WILDS canonical splits (union of these = the WILDS dataset).
WILDS_SPLITS = ["train", "id_val", "id_test", "ood_val", "ood_test"]

# WILDS region index -> name (verified against country_code_mapping.csv).
REGION_NAMES = {0: "Asia", 1: "Europe", 2: "Africa", 3: "Americas", 4: "Oceania", 5: "Other"}
_NAME_TO_REGION = {name.lower(): idx for idx, name in REGION_NAMES.items()}

# Time-window boundaries (UTC) that partition FMoW into the ID / OOD-Val / OOD-Test
# periods, matching the WILDS split construction in fmow_wilds_statistics.ipynb.
_ID_START = pd.Timestamp("2002-01-01", tz="UTC")
_ID_END = pd.Timestamp("2013-01-01", tz="UTC")       # ID period is [2002, 2013)
_OOD_VAL_END = pd.Timestamp("2016-01-01", tz="UTC")  # OOD-Val is [2013, 2016)
_OOD_TEST_END = pd.Timestamp("2018-01-01", tz="UTC")  # OOD-Test is [2016, 2018)


def assign_wilds_split(df: pd.DataFrame) -> pd.Series:
    """Map each row to its WILDS split (or <NA> if outside the WILDS dataset).

    Args:
        df (pd.DataFrame): Metadata with `split` and `timestamp` columns.

    Returns:
        pd.Series: WILDS split name per row, `<NA>` for rows not in WILDS
            (i.e. `split == "seq"` or a train/val/test row outside the
            timestamp windows that define the WILDS dataset).
    """
    ts = pd.to_datetime(df["timestamp"], utc=True, format="%Y-%m-%dT%H:%M:%SZ")
    id_period = (ts >= _ID_START) & (ts < _ID_END)
    ood_val_period = (ts >= _ID_END) & (ts < _OOD_VAL_END)
    ood_test_period = (ts >= _OOD_VAL_END) & (ts < _OOD_TEST_END)
    split = df["split"]

    out = pd.Series(pd.NA, index=df.index, dtype="object")
    out[(split == "train") & id_period] = "train"
    out[(split == "val") & id_period] = "id_val"
    out[(split == "test") & id_period] = "id_test"
    out[(split == "val") & ood_val_period] = "ood_val"
    out[(split == "test") & ood_test_period] = "ood_test"
    return out


def resolve_region(region) -> tuple[int | None, str]:
    """Resolve a region given as an int (0-5), a numeric string, or a name to (index, label).

    Args:
        region: Region filter, one of: `None` (no filter), an `int`/numeric
            `str` in `[0, 5]`, or a case-insensitive region name (see
            `REGION_NAMES`).

    Returns:
        tuple[int | None, str]: `(region_index, region_label)`. `(None,
        "all")` when `region` is `None`; otherwise the resolved index (0-5)
        and its display name from `REGION_NAMES`.

    Raises:
        ValueError: If `region` is a non-numeric string not found in
            `REGION_NAMES`, or resolves to an index outside `[0, 5]`.
    """
    if region is None:
        return None, "all"
    try:
        idx = int(region)
    except (TypeError, ValueError):
        key = str(region).lower()
        if key not in _NAME_TO_REGION:
            raise ValueError(
                f"unknown region {region!r}; choose 0-5 or one of {list(REGION_NAMES.values())}"
            )
        idx = _NAME_TO_REGION[key]
    if idx not in REGION_NAMES:
        raise ValueError(f"region index must be in 0-5, got {idx}")
    return idx, REGION_NAMES[idx]


def load_metadata(
    metadata_path: Path = DEFAULT_METADATA, split: str = "all", region=None
) -> pd.DataFrame:
    """Load the extended metadata, assign WILDS splits, filter, and drop rows without a span.

    Reads only the columns needed for extent statistics, tags each row with
    its WILDS split via `assign_wilds_split`, restricts to `split` (or the
    full WILDS union if `split="all"`), optionally restricts to one
    `region`, and drops rows with no matched Landsat pair (and therefore no
    `img_span_km`).

    Args:
        metadata_path (Path): Path to `rgb_metadata_extended.csv`. Defaults
            to `DEFAULT_METADATA`.
        split (str): `"all"` (WILDS union, default) or one of
            `WILDS_SPLITS`.
        region (int | str | None): `None` (all regions, default), a region
            index 0-5, or a region name (see `REGION_NAMES`).

    Returns:
        pd.DataFrame: Filtered rows with columns `split`, `category`, `y`,
        `img_span_km`, `timestamp`, `region`, `wilds_split`.

    Raises:
        ValueError: If `split` is not `"all"` or a member of
            `WILDS_SPLITS`, or if `region` cannot be resolved (see
            `resolve_region`).
    """
    if split != "all" and split not in WILDS_SPLITS:
        raise ValueError(f"split must be 'all' or one of {WILDS_SPLITS}, got {split!r}")
    region_idx, _ = resolve_region(region)

    df = pd.read_csv(
        metadata_path,
        usecols=["split", "category", "y", "img_span_km", "timestamp", "region"],
    )
    df["wilds_split"] = assign_wilds_split(df)

    if split == "all":
        df = df[df["wilds_split"].notna()]  # WILDS dataset: drop 'seq' / out-of-window
    else:
        df = df[df["wilds_split"] == split]

    if region_idx is not None:
        df = df[df["region"] == region_idx]

    # ~54k rows have no matched Landsat pair and therefore no span; drop them.
    df = df.dropna(subset=["img_span_km"])
    return df


def class_extent_stats(
    label: int,
    metadata_path: Path = DEFAULT_METADATA,
    split: str = "all",
    region=None,
    df: pd.DataFrame | None = None,
) -> dict:
    """Compute spatial-extent statistics (km) for one class and for all classes pooled.

    Args:
        label (int): Integer class label in `[0, NUM_CLASSES - 1]`.
        metadata_path (Path): Path to the extended metadata CSV; ignored if
            `df` is given. Defaults to `DEFAULT_METADATA`.
        split (str): `"all"` (WILDS union, default) or one of
            `WILDS_SPLITS`; ignored if `df` is given.
        region (int | str | None): `None` (all regions, default), a region
            index 0-5, or a region name; ignored if `df` is given.
        df (pd.DataFrame | None): Optional pre-loaded, pre-filtered frame
            from `load_metadata` (avoids re-reading/re-filtering the CSV).
            If `None`, loaded via
            `load_metadata(metadata_path, split=split, region=region)`.

    Returns:
        dict: Statistics with keys `label`, `category`, `split`, `region`,
        `n`, `extent_{mean,median,min,max,std}_km` for the requested class,
        plus `n_total` and the corresponding
        `overall_extent_{mean,median,min,max,std}_km` keys for all classes
        pooled (within the same split/region).

    Raises:
        ValueError: If `label` is outside `[0, NUM_CLASSES - 1]`, or if no
            rows match `label` (with the given `split`/`region`).
    """
    if not (0 <= label < NUM_CLASSES):
        raise ValueError(f"class label must be in [0, {NUM_CLASSES - 1}], got {label}")
    _, region_label = resolve_region(region)

    if df is None:
        df = load_metadata(metadata_path, split=split, region=region)

    sub = df[df["y"] == label]
    if len(sub) == 0:
        raise ValueError(
            f"no samples with span found for class {label} "
            f"(split={split}, region={region_label})"
        )

    category = sub["category"].iloc[0]
    return {
        "label": label,
        "category": category,
        "split": split,
        "region": region_label,
        "n": len(sub),
        "extent_mean_km": float(sub["img_span_km"].mean()),
        "extent_median_km": float(sub["img_span_km"].median()),
        "extent_min_km": float(sub["img_span_km"].min()),
        "extent_max_km": float(sub["img_span_km"].max()),
        "extent_std_km": float(sub["img_span_km"].std()),
        # All classes pooled (same split / region) for comparison.
        "n_total": len(df),
        "overall_extent_mean_km": float(df["img_span_km"].mean()),
        "overall_extent_median_km": float(df["img_span_km"].median()),
        "overall_extent_min_km": float(df["img_span_km"].min()),
        "overall_extent_max_km": float(df["img_span_km"].max()),
        "overall_extent_std_km": float(df["img_span_km"].std()),
    }


def all_class_stats(
    metadata_path: Path = DEFAULT_METADATA, split: str = "all", region=None
) -> pd.DataFrame:
    """Compute per-class spatial-extent statistics (km) for every class.

    Args:
        metadata_path (Path): Path to the extended metadata CSV. Defaults
            to `DEFAULT_METADATA`.
        split (str): `"all"` (WILDS union, default) or one of
            `WILDS_SPLITS`.
        region (int | str | None): `None` (all regions, default), a region
            index 0-5, or a region name.

    Returns:
        pd.DataFrame: One row per `(y, category)` class present in the
        filtered metadata, with columns `y`, `category`, `n`,
        `extent_mean_km`, `extent_median_km`, `extent_min_km`,
        `extent_max_km`, `extent_std_km`.
    """
    df = load_metadata(metadata_path, split=split, region=region)
    g = df.groupby(["y", "category"])["img_span_km"]
    out = pd.DataFrame(
        {
            "n": g.size(),
            "extent_mean_km": g.mean(),
            "extent_median_km": g.median(),
            "extent_min_km": g.min(),
            "extent_max_km": g.max(),
            "extent_std_km": g.std(),
        }
    )
    return out.reset_index()


def _print_stats(s: dict) -> None:
    """Print a `class_extent_stats` result as a human-readable summary.

    Args:
        s (dict): Statistics dict as returned by `class_extent_stats`.
    """
    scope = f"split={s['split']}, region={s['region']}"
    print(f"Class {s['label']:>2}  ({s['category']})   {scope}   n={s['n']}")
    print(f"  spatial extent (km)   mean = {s['extent_mean_km']:.4f}   "
          f"median = {s['extent_median_km']:.4f}   "
          f"min = {s['extent_min_km']:.4f}   "
          f"max = {s['extent_max_km']:.4f}   "
          f"std = {s['extent_std_km']:.4f}")
    print(f"All classes pooled ({scope})   n={s['n_total']}")
    print(f"  spatial extent (km)   mean = {s['overall_extent_mean_km']:.4f}   "
          f"median = {s['overall_extent_median_km']:.4f}   "
          f"min = {s['overall_extent_min_km']:.4f}   "
          f"max = {s['overall_extent_max_km']:.4f}   "
          f"std = {s['overall_extent_std_km']:.4f}")


def main() -> None:
    """CLI entrypoint: print spatial-extent statistics for one class or a ranked table for all classes.

    Parses `--split`/`--region`/`--metadata` filters plus either a
    positional class `label` or `--all`. With `--all`, prints a table of
    per-class extent statistics (via `all_class_stats`), sorted by sample
    count (`--sort-n`) or by mean extent (default), followed by the pooled
    statistics across all classes. Otherwise prints the single-class
    statistics (via `class_extent_stats` / `_print_stats`) for `label`.

    Raises:
        SystemExit: If neither a `label` nor `--all` is given.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("label", type=int, nargs="?", help="class label in [0, 61]")
    parser.add_argument("--all", action="store_true", help="print a ranked table for all 62 classes")
    parser.add_argument("--split", choices=["all"] + WILDS_SPLITS, default="all",
                        help="WILDS split (default: all = WILDS union)")
    parser.add_argument("--region", default=None,
                        help="region filter: index 0-5 or name (Asia/Europe/Africa/Americas/Oceania/Other)")
    parser.add_argument("--sort-n", action="store_true",
                        help="with --all, sort the table by sample count (n) instead of mean extent")
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA,
                        help="path to rgb_metadata_extended.csv")
    args = parser.parse_args()

    _, region_label = resolve_region(args.region)  # validate early

    if args.all:
        df = load_metadata(args.metadata, split=args.split, region=args.region)
        table = all_class_stats(args.metadata, split=args.split, region=args.region)
        if args.sort_n:
            table = table.sort_values("n", ascending=False).reset_index(drop=True)
        else:
            table = table.sort_values("extent_mean_km").reset_index(drop=True)
        with pd.option_context("display.max_rows", None, "display.width", 120,
                               "display.float_format", lambda x: f"{x:.4f}"):
            print(table.to_string(index=False))
        span = df["img_span_km"]
        print(f"\nAll classes pooled (split={args.split}, region={region_label}, n={len(df)}) "
              f"extent (km):   mean = {span.mean():.4f}   median = {span.median():.4f}   "
              f"min = {span.min():.4f}   max = {span.max():.4f}   std = {span.std():.4f}")
        print(f"Mean of per-class mean extents: {table['extent_mean_km'].mean():.4f} km "
              f"(spread {table['extent_mean_km'].min():.3f}–{table['extent_mean_km'].max():.3f} km)")
        return

    if args.label is None:
        parser.error("provide a class label in [0, 61], or use --all")

    _print_stats(class_extent_stats(args.label, args.metadata, split=args.split, region=args.region))


if __name__ == "__main__":
    main()
