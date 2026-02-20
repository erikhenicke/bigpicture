from pathlib import Path

import pandas as pd
from wilds.datasets.wilds_dataset import WILDSSubset

from dataset.fmow_multiscale_dataset import FMoWMultiScaleDataset
import platform


FIVE_REGIONS = ["Europe", "Americas", "Asia", "Africa", "Oceania"]


def get_data(frac: float = 0.1):
    fmow_dir = '/home/henicke/git/bigpicture/data'
    landsat_dir = '/home/datasets4/FMoW_LandSat'
    preprocessed_dir = None

    if platform.node() == 'gaia4' or platform.node() == 'gaia5':
        preprocessed_dir = '/data/henicke/FMoW_LandSat'

    dataset = FMoWMultiScaleDataset(
        fmow_dir=fmow_dir,
        landsat_dir=landsat_dir,
        preprocessed_dir=preprocessed_dir
    )
    return dataset.get_subset(split='train', frac=frac), dataset.get_subset(split='val', frac=frac)


def load_country_to_region_map(mapping_csv: str):
    mapping_df = pd.read_csv(mapping_csv, dtype={"alpha-3": str})
    mapping_df["region"] = mapping_df["region"].fillna("")

    country_to_region = {}
    for _, row in mapping_df.iterrows():
        country_code = row["alpha-3"]
        region = row["region"]
        if region in FIVE_REGIONS:
            country_to_region[country_code] = region

    return country_to_region


def split_subset_by_region(subset, country_to_region):
    """
    Split a WILDS subset into 5 geographic bins by country code.

    Index alignment detail:
    - subset.indices indexes into non-sequestered dataset arrays
    - dataset.metadata is the full metadata table
    - dataset.full_idxs maps dataset indices -> original metadata row indices
    """
    dataset = subset.dataset
    metadata_df = dataset.metadata

    region_indices = {region: [] for region in FIVE_REGIONS}

    for dataset_idx in subset.indices:
        original_metadata_idx = int(dataset.full_idxs[dataset_idx])
        country_code = str(metadata_df.iloc[original_metadata_idx]["country_code"]).strip().upper()
        region = country_to_region.get(country_code)
        if region in region_indices:
            region_indices[region].append(int(dataset_idx))

    return {
        region: WILDSSubset(
            dataset,
            indices,
            getattr(subset, "transform", None),
            getattr(subset, "do_transform_y", False),
        )
        for region, indices in region_indices.items()
    }


def split_train_val_by_region(train_subset, val_subset, mapping_csv: str):
    country_to_region = load_country_to_region_map(mapping_csv)
    train_regions = split_subset_by_region(train_subset, country_to_region)
    val_regions = split_subset_by_region(val_subset, country_to_region)
    return train_regions, val_regions


if __name__ == "__main__":
    train, val = get_data(frac=1.0)

    mapping_csv = Path('/home/henicke/git/bigpicture/data/fmow_v1.1/country_code_mapping.csv')
    train_regions, val_regions = split_train_val_by_region(train, val, str(mapping_csv))

    print("Train split sizes by region:")
    for region in FIVE_REGIONS:
        print(f"  {region}: {len(train_regions[region])}")

    print("\nVal split sizes by region:")
    for region in FIVE_REGIONS:
        print(f"  {region}: {len(val_regions[region])}")

    train_regions_total = sum(len(subset) for subset in train_regions.values())
    val_regions_total = sum(len(subset) for subset in val_regions.values())
    print(f"Total train size: {len(train)}, Total val size: {len(val)}")
    print(f"\nTotal train size from regions: {train_regions_total}, Total val size from regions: {val_regions_total}")  

    def _print_missing(subset, split_name):
        dataset = subset.dataset
        metadata_df = dataset.metadata

        original_indices = set(int(idx) for idx in subset.indices)
        region_indices = set()
        for region in FIVE_REGIONS:
            region_indices.update(int(idx) for idx in train_regions[region].indices if split_name == "train")
            region_indices.update(int(idx) for idx in val_regions[region].indices if split_name == "val")

        missing_indices = sorted(original_indices - region_indices)
        print(f"\nMissing {split_name} indices: {len(missing_indices)}")
        if not missing_indices:
            return

        missing_country_codes = []
        for dataset_idx in missing_indices:
            original_metadata_idx = int(dataset.full_idxs[dataset_idx])
            country_code = str(metadata_df.iloc[original_metadata_idx]["country_code"]).strip().upper()
            missing_country_codes.append(country_code)

        print(f"Missing {split_name} country codes (unique): {sorted(set(missing_country_codes))}")

    _print_missing(train, "train")
    _print_missing(val, "val")