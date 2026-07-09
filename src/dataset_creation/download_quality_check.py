"""
download_quality_check.py

Quality-checks downloaded Landsat8 GeoTIFFs in `IMAGES_DIR`: verifies that all FMoW
WILDS train/val/test/id_val/id_test indices have a corresponding downloaded image, then
runs a sample (or all) of the GeoTIFFs through `check_image_quality` to flag images with
too many zero, masked, or NaN/Inf pixels. Results are written to `quality_check.log`.

Main functions:
    - check_image_quality: Computes zero/mask/NaN-Inf pixel fractions for one GeoTIFF and
      checks them against thresholds.
    - get_downloaded_indices: Extracts and sorts the FMoW sample indices present in a list
      of downloaded GeoTIFF filenames.
    - main: Sets up logging, checks download completeness against the FMoW WILDS dataset,
      and runs the quality check over a (sub)sample of downloaded images.
"""
import os
import pathlib
import random
import logging
from glob import glob
import re

import numpy as np
import rasterio
from wilds import *


PROJECT_ROOT = pathlib.Path(__file__).parent.parent.parent.resolve()
DATA_DIR = (PROJECT_ROOT.parent.parent.parent /
            "datasets4" / "FMoW_LandSat" / "fmow_landsat")
IMAGES_DIR = DATA_DIR / "images"
LOG_DIR = PROJECT_ROOT / "log"
LOG_FILE = LOG_DIR / 'quality_check.log'


def check_image_quality(image_path, zero_threshold=0.01, mask_threshold=0.01, nan_inf_threshold=0.0):
    """Check a Landsat8 GeoTIFF for zero, masked, and NaN/Inf pixels against thresholds.

    Reads every band and computes the fraction of masked pixels, the fraction of
    exact-zero pixels (masked or not), and the fraction of non-finite (NaN/Inf) pixels
    across the whole image. The image fails the check if any of these fractions exceeds
    its threshold.

    Args:
        image_path (str): Path to the GeoTIFF file.
        zero_threshold (float): Maximum allowed fraction of zero-valued pixels. Defaults
            to 0.01.
        mask_threshold (float): Maximum allowed fraction of masked pixels. Defaults to
            0.01.
        nan_inf_threshold (float): Maximum allowed fraction of NaN/Inf pixels. Defaults
            to 0.0.

    Returns:
        tuple[bool, dict]: Whether the image passes all thresholds, and a dict with keys
            `zero_frac`, `mask_frac`, `nan_inf_frac` (float fractions) and `total_pixels`
            (int, summed over bands). If the image has zero total pixels, returns
            `(False, {...})` with all fractions set to 1.0.
    """
    with rasterio.open(image_path) as src:
        zero_counts = []
        mask_counts = []
        nan_inf_counts = []
        total_pixels = 0

        for band_idx in range(1, src.count + 1):
            band = src.read(band_idx, masked=True)
            total_pixels_per_band = band.size

            # Masked pixels
            masked_pixels = np.sum(band.mask)

            # Total zeros: masked or non-masked where value == 0
            # Works on full array (NaN/inf != 0)
            total_zeros = np.sum(band.data == 0)

            # NaN + Inf (finite=False), anywhere
            nan_inf_pixels = np.sum(~np.isfinite(band.data))

            zero_counts.append(total_zeros)
            mask_counts.append(masked_pixels)
            nan_inf_counts.append(nan_inf_pixels)
            total_pixels += total_pixels_per_band

        if total_pixels == 0:
            return False, {'zero_frac': 1.0, 'mask_frac': 1.0, 'nan_inf_frac': 1.0, 'total_pixels': 0}

        metrics = {
            'zero_frac': sum(zero_counts) / total_pixels,
            'mask_frac': sum(mask_counts) / total_pixels,
            'nan_inf_frac': sum(nan_inf_counts) / total_pixels,
            'total_pixels': total_pixels
        }

        # FAIL if ANY metric >= threshold (zero tolerance for nan_inf)
        passes = (
            metrics['zero_frac'] <= zero_threshold and
            metrics['mask_frac'] <= mask_threshold and
            metrics['nan_inf_frac'] <= nan_inf_threshold
        )
        return passes, metrics


def get_downloaded_indices(tiff_files, logger):
    """Extract and sort the FMoW sample indices encoded in downloaded GeoTIFF filenames.

    Args:
        tiff_files (list[str]): Paths/filenames of downloaded GeoTIFFs, expected to match
            `.*image_(\\d+)\\.tif$`.
        logger (logging.Logger): Logger used to report filenames that don't match the
            pattern.

    Returns:
        np.ndarray: Sorted array of shape (len(tiff_files),), dtype float64, containing
            the extracted sample indices. Entries for filenames that fail to match the
            pattern are left uninitialized (`np.empty`) rather than skipped.
    """
    size_indices = len(tiff_files)
    downloaded_indices = np.empty(shape=(size_indices,))
    pattern = r'.*image_(\d+)\.tif$'
    for i, filename in enumerate(tiff_files):
        match = re.search(pattern, filename)
        if match:
            downloaded_indices[i] = int(match.group(1))
        else:
            logger.error(f"Image {filename} does not match pattern {pattern}.")
    return np.sort(downloaded_indices)


def main(num_samples=None, zero_threshold=0.01, mask_threshold=0.01, nan_inf_threshold=0.0):
    """Check download completeness and run a quality check over downloaded Landsat8 GeoTIFFs.

    Sets up file logging, confirms every FMoW WILDS train/val/test/id_val/id_test index
    has a corresponding GeoTIFF in `IMAGES_DIR`, then runs `check_image_quality` over
    either a random `num_samples` GeoTIFFs or all of them, logging PASS/FAIL per image and
    a summary of failures.

    Args:
        num_samples (int | None): Number of GeoTIFFs to randomly sample for the quality
            check. If None, not an int, or larger than the number of files found, all
            downloaded GeoTIFFs are checked.
        zero_threshold (float): Passed through to `check_image_quality`.
        mask_threshold (float): Passed through to `check_image_quality`.
        nan_inf_threshold (float): Passed through to `check_image_quality`.
    """

    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)

    # Initialize logging
    logger = logging.getLogger('quality_check')
    logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(LOG_FILE, mode='w')
    file_handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)

    if not IMAGES_DIR.exists():
        logger.error(f"Images directory not found: {IMAGES_DIR}")
        return

    tif_paths = glob(str(IMAGES_DIR / "*.tif"))
    if not tif_paths:
        logger.warning("No TIFF files found in images directory.")
        return

    # Check if download is complete
    dataset = get_dataset(dataset="fmow", download=False, root_dir="/home/henicke/data")
    split_names = ['train', 'val', 'test', 'id_val', 'id_test']
    subsets_indices = [dataset.get_subset(
        split_name).indices for split_name in split_names]
    fmow_wilds_indices = np.sort(np.concatenate(subsets_indices))
    downloaded_indices = get_downloaded_indices(tif_paths, logger)
    if np.array_equiv(fmow_wilds_indices, downloaded_indices):
        logger.info("All images downloaded!")
    else:
        logger.info("Download incomplete!")

    if num_samples and isinstance(num_samples, int) and num_samples <= len(tif_paths):
        sample_paths = random.sample(
            tif_paths, min(num_samples, len(tif_paths)))
    else:
        sample_paths = tif_paths

    # Quality check
    logger.info(
        f"Found {len(tif_paths)} TIFF files. Quality checking {len(sample_paths)} images.")

    failures = []
    for path in sample_paths:
        passes, metrics = check_image_quality(
            path, zero_threshold, mask_threshold, nan_inf_threshold)
        filename = os.path.basename(path)
        status = "PASS" if passes else "FAIL"
        logger.info(
            f"{status}: {filename} | zero={metrics['zero_frac']:.1%} | mask={metrics['mask_frac']:.1%} | nan_inf={metrics['nan_inf_frac']:.1%} | pixels={metrics['total_pixels']:,}")
        if not passes:
            failures.append((path, metrics))

    if failures:
        logger.warning(f"{len(failures)}/{len(sample_paths)} images failed.")
        for path, metrics in failures:
            reason = []
            if metrics['zero_frac'] > zero_threshold:
                reason.append(f"zeros>{zero_threshold:.0%}")
            if metrics['mask_frac'] > mask_threshold:
                reason.append(f"mask>{mask_threshold:.0%}")
            if metrics['nan_inf_frac'] > nan_inf_threshold:
                reason.append("nan_inf>0%")
            logger.warning(
                f"{os.path.basename(path)}: {', '.join(reason)} | zero={metrics['zero_frac']:.1%}, mask={metrics['mask_frac']:.1%}, nan_inf={metrics['nan_inf_frac']:.1%}")
    else:
        logger.info("All sampled images passed all checks!")


if __name__ == "__main__":
    main()
