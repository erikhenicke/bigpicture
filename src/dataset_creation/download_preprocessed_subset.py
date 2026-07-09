"""
download_preprocessed_subset.py

Sample a random, not-yet-downloaded subset of preprocessed FMoW/Landsat pairs
matching a class + region specification from `rgb_metadata_extended.csv`,
then rsync them down from the gaia4 preprocessing cache.

The `.pt` files in `fmow_preprocessed/{fmow_rgb,landsat}/` are named by the
row index of `rgb_metadata_extended.csv` (`rgb_img_<id>.pt` / `image_<id>.pt`),
so the CSV index doubles as the sample id used both locally and on the remote.

Only rows inside the WILDS FMoW dataset were ever preprocessed - the raw CSV's
470k train/val/test rows are shrunk to 141,696 by the WILDS timestamp windows
(train/val/test split column + [2002,2013)/[2013,2016)/[2016,2018) timestamp
windows -> train/{id,ood}_val/{id,ood}_test), exactly as reconstructed in
average_class_extent.py / fmow_wilds_statistics.ipynb. `split == "seq"` rows
and any train/val/test row outside those windows were never turned into .pt
files and must be masked out before sampling ids.

Example:
    uv run python src/dataset_creation/download_preprocessed_subset.py \
        --category crop_field --region africa -n 50
"""
import argparse
import difflib
import pathlib
import random
import re
import subprocess
import sys

import pandas as pd

PROJECT_ROOT = pathlib.Path(__file__).parent.parent.parent.resolve()
METADATA_CSV = PROJECT_ROOT / "data" / "fmow_landsat" / "rgb_metadata_extended.csv"
LOCAL_PREPROCESSED_DIR = PROJECT_ROOT / "data" / "fmow_preprocessed"
REMOTE_PREPROCESSED_DIR = "gaia4:/data/henicke/FMoW_LandSat_Norm/fmow_preprocessed/"

RGB_SUBDIR = "fmow_rgb"
LANDSAT_SUBDIR = "landsat"
RGB_ID_RE = re.compile(r"rgb_img_(\d+)\.pt$")
LANDSAT_ID_RE = re.compile(r"image_(\d+)\.pt$")

REGION_NAME_TO_CODE = {
    "asia": 0,
    "europe": 1,
    "africa": 2,
    "americas": 3,
    "oceania": 4,
    "other": 5,
}

# WILDS canonical splits (union of these = the WILDS dataset), matching
# average_class_extent.py.
WILDS_SPLITS = ["train", "id_val", "id_test", "ood_val", "ood_test"]

# Time-window boundaries (UTC) that partition FMoW into the ID / OOD-Val /
# OOD-Test periods, matching the WILDS split construction in
# fmow_wilds_statistics.ipynb / average_class_extent.py.
_ID_START = pd.Timestamp("2002-01-01", tz="UTC")
_ID_END = pd.Timestamp("2013-01-01", tz="UTC")        # ID period is [2002, 2013)
_OOD_VAL_END = pd.Timestamp("2016-01-01", tz="UTC")   # OOD-Val is [2013, 2016)
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


def parse_regions(values: list[str]) -> list[int]:
    """Resolve region specifiers (names or numeric codes) to region codes.

    Args:
        values (list[str]): Region names (case-insensitive) or numeric codes (0-5).

    Returns:
        list[int]: Resolved region codes.
    """
    codes = []
    for value in values:
        if value.isdigit():
            codes.append(int(value))
        elif value.lower() in REGION_NAME_TO_CODE:
            codes.append(REGION_NAME_TO_CODE[value.lower()])
        else:
            valid = ", ".join(REGION_NAME_TO_CODE)
            raise SystemExit(f"Unknown region '{value}'. Use a code 0-5 or one of: {valid}.")
    return codes


def validate_categories(categories: list[str], known_categories: pd.Series) -> None:
    """Raise a helpful error if a requested category does not exist in the metadata.

    Args:
        categories (list[str]): Requested class names (case-insensitive).
        known_categories (pd.Series): The metadata's `category` column.
    """
    known_lower = {c.lower() for c in known_categories.unique()}
    for category in categories:
        if category.lower() not in known_lower:
            close = difflib.get_close_matches(category.lower(), known_lower, n=3)
            hint = f" Did you mean: {', '.join(close)}?" if close else ""
            raise SystemExit(f"Unknown category '{category}'.{hint}")


def existing_ids(directory: pathlib.Path, pattern: re.Pattern) -> set[int]:
    """Collect sample ids already present in a local directory of `.pt` files.

    Args:
        directory (pathlib.Path): Directory to scan.
        pattern (re.Pattern): Regex with one capture group extracting the id.

    Returns:
        set[int]: Ids already downloaded.
    """
    if not directory.exists():
        return set()
    ids = set()
    for path in directory.iterdir():
        match = pattern.match(path.name)
        if match:
            ids.add(int(match.group(1)))
    return ids


def select_sample_ids(metadata: pd.DataFrame, categories: list[str], region_codes: list[int],
                       splits: list[str] | None, n: int, seed: int | None,
                       already_downloaded: set[int]) -> list[int]:
    """Filter metadata by class/region/WILDS-split and randomly pick `n` new sample ids.

    Args:
        metadata (pd.DataFrame): `rgb_metadata_extended.csv` already restricted to rows
            inside the WILDS dataset (non-null `wilds_split` column), indexed by sample id.
        categories (list[str]): Class names to match against the `category` column.
        region_codes (list[int]): Region codes to match against the `region` column.
        splits (list[str] | None): Optional WILDS split filter (see `WILDS_SPLITS`).
        n (int): Number of samples requested.
        seed (int | None): Random seed for reproducible sampling.
        already_downloaded (set[int]): Ids to exclude because they exist locally already.

    Returns:
        list[int]: Sampled ids matching the specification and not yet downloaded.
    """
    categories_lower = {c.lower() for c in categories}
    mask = metadata["category"].str.lower().isin(categories_lower) & metadata["region"].isin(region_codes)
    if splits:
        mask &= metadata["wilds_split"].isin(splits)
    candidates = set(metadata.index[mask]) - already_downloaded

    if not candidates:
        raise SystemExit("No matching samples left to download - all candidates are already local.")

    if n > len(candidates):
        print(f"Warning: only {len(candidates)} new matching samples available, "
              f"requested {n}. Downloading all of them.", file=sys.stderr)
        n = len(candidates)

    rng = random.Random(seed)
    return rng.sample(sorted(candidates), n)


def build_file_list(sample_ids: list[int]) -> str:
    """Build the relative-path file list rsync should transfer.

    Args:
        sample_ids (list[int]): Sample ids to include.

    Returns:
        str: Newline-separated relative paths for `rsync --files-from`.
    """
    lines = []
    for sample_id in sample_ids:
        lines.append(f"{RGB_SUBDIR}/rgb_img_{sample_id}.pt")
        lines.append(f"{LANDSAT_SUBDIR}/image_{sample_id}.pt")
    return "\n".join(lines) + "\n"


def run_rsync(file_list: str, remote: str, local: pathlib.Path) -> None:
    """Invoke rsync to pull the selected files from the remote preprocessing cache.

    Args:
        file_list (str): Newline-separated relative paths, fed via `--files-from=-`.
        remote (str): Remote rsync source, e.g. `gaia4:/path/to/fmow_preprocessed/`.
        local (pathlib.Path): Local destination directory.
    """
    local.mkdir(parents=True, exist_ok=True)
    cmd = ["rsync", "-avzP", "--files-from=-", remote, str(local) + "/"]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, input=file_list, text=True, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sample new class/region-matching preprocessed FMoW+Landsat pairs and rsync them from gaia4.")
    parser.add_argument("--category", "-c", nargs="+", required=True,
                        help="One or more FMoW class names (e.g. crop_field), case-insensitive.")
    parser.add_argument("--region", "-r", nargs="+", required=True,
                        help="One or more regions: name (asia/europe/africa/americas/oceania/other) or code 0-5.")
    parser.add_argument("--split", nargs="+", default=None, choices=WILDS_SPLITS,
                        help=f"Optional WILDS split filter, from {WILDS_SPLITS}. Default: all (WILDS union).")
    parser.add_argument("-n", "--num-samples", type=int, required=True,
                        help="Number of new samples to download.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for sample selection.")
    parser.add_argument("--metadata-csv", type=pathlib.Path, default=METADATA_CSV)
    parser.add_argument("--local-dir", type=pathlib.Path, default=LOCAL_PREPROCESSED_DIR)
    parser.add_argument("--remote", default=REMOTE_PREPROCESSED_DIR)
    parser.add_argument("--dry-run", action="store_true",
                        help="Select samples and print the rsync command/file list without connecting to the remote.")
    args = parser.parse_args()

    region_codes = parse_regions(args.region)

    metadata = pd.read_csv(args.metadata_csv)
    metadata["wilds_split"] = assign_wilds_split(metadata)
    metadata = metadata[metadata["wilds_split"].notna()]
    validate_categories(args.category, metadata["category"])

    local_ids = existing_ids(args.local_dir / RGB_SUBDIR, RGB_ID_RE)
    local_ids |= existing_ids(args.local_dir / LANDSAT_SUBDIR, LANDSAT_ID_RE)
    print(f"Found {len(local_ids)} sample ids already present locally in {args.local_dir}.")

    sample_ids = select_sample_ids(
        metadata, args.category, region_codes, args.split,
        args.num_samples, args.seed, local_ids)

    print(f"Selected {len(sample_ids)} new samples "
          f"(category={args.category}, region={args.region}, split={args.split or 'all'}).")

    file_list = build_file_list(sample_ids)
    if args.dry_run:
        print(f"--dry-run: would rsync {len(sample_ids)} pairs from {args.remote} to {args.local_dir}/")
        print(file_list, end="")
        return

    run_rsync(file_list, args.remote, args.local_dir)


if __name__ == "__main__":
    main()
