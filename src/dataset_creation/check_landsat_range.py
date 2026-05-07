"""Check whether fmow-statistics normalized Landsat values fall outside [-1, 1].

Loads raw GeoTIFFs (no preprocessing cache, no resize), applies the
fmow-statistics normalization, and reports per-band min/max statistics.
"""

import argparse
from pathlib import Path

import numpy as np
import rasterio
import torch
from torchvision.transforms import v2 as transforms
from tqdm import tqdm

LANDSAT_NORM = transforms.Normalize(
    mean=[0.06259285658597946, 0.0880340114235878, 0.09441816806793213,
          0.2327403724193573, 0.19073842465877533, 0.12976829707622528],
    std=[0.039894334971904755, 0.049978554248809814, 0.0687960833311081,
         0.092967689037323, 0.09390033036470413, 0.0819208025932312],
)

NUM_BANDS = 6


def check_range(landsat_dir: str, max_images: int = 500):
    images_dir = Path(landsat_dir) / "fmow_landsat" / "images"
    all_tif_files = list(images_dir.glob("image_*.tif"))
    rng = np.random.default_rng()
    tif_files = list(rng.permutation(all_tif_files))[:max_images]
    print(f"Checking {len(tif_files)} raw Landsat GeoTIFFs from {images_dir}\n")

    global_min = torch.full((NUM_BANDS,), float("inf"))
    global_max = torch.full((NUM_BANDS,), float("-inf"))
    band_sum = torch.zeros(NUM_BANDS, dtype=torch.float64)
    band_sum_sq = torch.zeros(NUM_BANDS, dtype=torch.float64)
    total_pixels = 0
    outside_count = 0

    for tif_path in tqdm(tif_files, desc="Scanning"):
        with rasterio.open(tif_path) as src:
            data = src.read().astype(np.float32)

        tensor = torch.from_numpy(data)
        tensor = LANDSAT_NORM(tensor)

        n_pixels = tensor.shape[1] * tensor.shape[2]
        total_pixels += n_pixels
        band_sum += tensor.to(torch.float64).sum(dim=(1, 2))
        band_sum_sq += (tensor.to(torch.float64) ** 2).sum(dim=(1, 2))

        band_min = tensor.amin(dim=(1, 2))
        band_max = tensor.amax(dim=(1, 2))
        global_min = torch.minimum(global_min, band_min)
        global_max = torch.maximum(global_max, band_max)

        if tensor.min() < -1 or tensor.max() > 1:
            outside_count += 1

    band_mean = band_sum / total_pixels
    band_std = ((band_sum_sq / total_pixels) - band_mean ** 2).sqrt()

    print(f"\n{'Band':<6} {'Min':>10} {'Max':>10} {'Mean':>10} {'Std':>10}  Outside [-1,1]?")
    print("-" * 65)
    for b in range(NUM_BANDS):
        flag = "YES" if global_min[b] < -1 or global_max[b] > 1 else "no"
        print(f"  {b:<4} {global_min[b]:>10.4f} {global_max[b]:>10.4f} {band_mean[b]:>10.4f} {band_std[b]:>10.4f}  {flag}")

    print(f"\n{outside_count}/{len(tif_files)} images have at least one value outside [-1, 1]")


def check_preprocessed(preprocessed_dir: str, max_images: int = 500):
    """Load cached .pt files and check if all values are clipped to [-1, 1]."""
    landsat_dir = Path(preprocessed_dir) / "fmow_preprocessed_norm" / "landsat"
    all_pt_files = list(landsat_dir.glob("image_*.pt"))
    rng = np.random.default_rng()
    pt_files = list(rng.permutation(all_pt_files))[:max_images]
    print(f"Checking {len(pt_files)} preprocessed .pt files from {landsat_dir}\n")

    global_min = torch.full((NUM_BANDS,), float("inf"))
    global_max = torch.full((NUM_BANDS,), float("-inf"))
    band_sum = torch.zeros(NUM_BANDS, dtype=torch.float64)
    band_sum_sq = torch.zeros(NUM_BANDS, dtype=torch.float64)
    total_pixels = 0
    outside_count = 0

    for pt_path in tqdm(pt_files, desc="Scanning cached"):
        tensor = torch.load(pt_path, weights_only=False)

        n_pixels = tensor.shape[1] * tensor.shape[2]
        total_pixels += n_pixels
        band_sum += tensor.to(torch.float64).sum(dim=(1, 2))
        band_sum_sq += (tensor.to(torch.float64) ** 2).sum(dim=(1, 2))

        band_min = tensor.amin(dim=(1, 2))
        band_max = tensor.amax(dim=(1, 2))
        global_min = torch.minimum(global_min, band_min)
        global_max = torch.maximum(global_max, band_max)

        if tensor.min() < -1 or tensor.max() > 1:
            outside_count += 1

    band_mean = band_sum / total_pixels
    band_std = ((band_sum_sq / total_pixels) - band_mean ** 2).sqrt()

    print(f"\n{'Band':<6} {'Min':>10} {'Max':>10} {'Mean':>10} {'Std':>10}  Outside [-1,1]?")
    print("-" * 65)
    for b in range(NUM_BANDS):
        flag = "YES" if global_min[b] < -1 or global_max[b] > 1 else "no"
        print(f"  {b:<4} {global_min[b]:>10.4f} {global_max[b]:>10.4f} {band_mean[b]:>10.4f} {band_std[b]:>10.4f}  {flag}")

    print(f"\n{outside_count}/{len(pt_files)} images have at least one value outside [-1, 1]")
    if outside_count == 0:
        print("=> All values within [-1, 1] — confirms PIL uint8 clipping during preprocessing.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--landsat-dir", type=str, default="/home/datasets4/FMoW_LandSat")
    parser.add_argument("--preprocessed-dir", type=str, 
                        help="Path to preprocessed cache (e.g. /data/henicke/FMoW_LandSat). "
                             "If set, checks cached .pt files instead of raw GeoTIFFs.")
    parser.add_argument("--max-images", type=int, default=500)
    args = parser.parse_args()

    if args.preprocessed_dir:
        check_preprocessed(args.preprocessed_dir, args.max_images)
    else:
        check_range(args.landsat_dir, args.max_images)
