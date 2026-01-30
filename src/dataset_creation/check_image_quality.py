import os
import pathlib
import random
import logging
from glob import glob

import numpy as np
import rasterio


PROJECT_ROOT = pathlib.Path(__file__).parent.parent.parent.resolve()
# TODO
DATA_DIR = PROJECT_ROOT / "data" / "fmow_landsat"
# DATA_DIR = (PROJECT_ROOT.parent.parent.parent /
#             "datasets4" / "FMoW_LandSat" / "fmow_landsat")
IMAGES_DIR = DATA_DIR / "images"
LOG_FILE = PROJECT_ROOT / 'quality_check.log'


def check_image_quality(image_path, zero_threshold=0.01, mask_threshold=0.01, nan_inf_threshold=0.0):
    """
    Comprehensive check: zeros (masked+non-masked), masked pixels, NaN/Inf.
    FAILS if ANY threshold exceeded (zero tolerance for NaN/Inf).

    Args:
        image_path (str): Path to the GeoTIFF file.
        zero_threshold (float): Max fraction zeros (default 5%).
        mask_threshold (float): Max fraction masked (default 95%).
        nan_inf_threshold (float): Max fraction NaN/Inf (default 0%).

    Returns:
        tuple: (bool: passes, dict: metrics)
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


def main(num_samples=None, zero_threshold=0.01, mask_threshold=0.01, nan_inf_threshold=0.0):

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

    if num_samples and isinstance(num_samples, int) and num_samples <= len(tif_paths):
        sample_paths = random.sample(
            tif_paths, min(num_samples, len(tif_paths)))
    else:
        sample_paths = tif_paths

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
