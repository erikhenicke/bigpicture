"""
check_image_stats.py

Check whether fmow-statistics normalized image values fall outside [-1, 1].

Supports both Landsat GeoTIFFs and FMoW RGB PNGs, as well as their
preprocessed .pt counterparts.

Functions:
    check_landsat_range / check_rgb_range: Sample raw GeoTIFFs/PNGs from disk,
        apply LANDSAT_NORM/RGB_NORM, and print per-channel min/max/mean/std.
    check_landsat_preprocessed / check_rgb_preprocessed: Same, but for
        already-normalized cached `.pt` tensors (no renormalization applied).
    _gather_stats: Shared per-channel statistics accumulator used by all four
        `check_*` functions above.
    _sample_files / _load_pt_tensors / _print_band_table: Helpers for
        randomly sampling files, lazily loading cached tensors, and printing
        the resulting stats table.

Run as a script with --landsat-dir/--fmow-dir/--preprocessed-dir/--modality
to check one or both modalities, either from raw source images or from a
preprocessed cache.
"""

import argparse
from pathlib import Path

import numpy as np
import rasterio
import torch
from PIL import Image
from torchvision.transforms import v2 as transforms
from tqdm import tqdm

LANDSAT_NORM = transforms.Normalize(
    mean=[0.06259285658597946, 0.0880340114235878, 0.09441816806793213,
          0.2327403724193573, 0.19073842465877533, 0.12976829707622528],
    std=[0.039894334971904755, 0.049978554248809814, 0.0687960833311081,
         0.092967689037323, 0.09390033036470413, 0.0819208025932312],
)

RGB_NORM = transforms.Compose([
    transforms.ToImage(),
    transforms.ToDtype(torch.float32, scale=True),
    transforms.Normalize(
        mean=[0.4155880808830261, 0.41815927624702454, 0.3903605341911316],
        std=[0.24812281131744385, 0.24405813217163086, 0.2482403963804245],
    ),
])


def _print_band_table(num_channels, global_min, global_max, band_mean, band_std, total_files, outside_count):
    """Print a per-channel min/max/mean/std table plus an outside-[-1,1] summary.

    Args:
        num_channels (int): Number of channels/bands to print a row for.
        global_min (torch.Tensor): Per-channel minimum, shape (num_channels,).
        global_max (torch.Tensor): Per-channel maximum, shape (num_channels,).
        band_mean (torch.Tensor): Per-channel mean, shape (num_channels,).
        band_std (torch.Tensor): Per-channel standard deviation, shape (num_channels,).
        total_files (int): Total number of images the stats were computed over.
        outside_count (int): Number of images with at least one value outside [-1, 1].
    """
    print(f"\n{'Chan':<6} {'Min':>10} {'Max':>10} {'Mean':>10} {'Std':>10}  Outside [-1,1]?")
    print("-" * 65)
    for b in range(num_channels):
        flag = "YES" if global_min[b] < -1 or global_max[b] > 1 else "no"
        print(f"  {b:<4} {global_min[b]:>10.4f} {global_max[b]:>10.4f} {band_mean[b]:>10.4f} {band_std[b]:>10.4f}  {flag}")
    print(f"\n{outside_count}/{total_files} images have at least one value outside [-1, 1]")


def _gather_stats(tensors, num_channels, desc):
    """Accumulate per-channel min/max/mean/std across a stream of image tensors.

    Iterates once over `tensors`, tracking global per-channel min/max, running
    sums for mean/std (in float64 for numerical stability), and how many
    images have at least one value outside [-1, 1].

    Args:
        tensors (Iterable[torch.Tensor]): Image tensors to accumulate over, each
            shaped (num_channels, H, W).
        num_channels (int): Number of channels per tensor.
        desc (str): Description label shown on the tqdm progress bar.

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
            `(global_min, global_max, band_mean, band_std, outside_count)`,
            where the first four are 1D tensors of shape (num_channels,) and
            `outside_count` is the number of images with a value outside
            [-1, 1].
    """
    global_min = torch.full((num_channels,), float("inf"))
    global_max = torch.full((num_channels,), float("-inf"))
    band_sum = torch.zeros(num_channels, dtype=torch.float64)
    band_sum_sq = torch.zeros(num_channels, dtype=torch.float64)
    total_pixels = 0
    outside_count = 0

    for tensor in tqdm(tensors, desc=desc):
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
    return global_min, global_max, band_mean, band_std, outside_count


def _sample_files(file_list, max_images):
    """Randomly permute and truncate a list of files to at most `max_images` entries.

    Args:
        file_list (list): Files (or paths) to sample from.
        max_images (int): Maximum number of files to keep.

    Returns:
        list: Randomly ordered subset of `file_list`, of length
            `min(len(file_list), max_images)`.
    """
    rng = np.random.default_rng()
    return list(rng.permutation(file_list))[:max_images]


def _load_pt_tensors(pt_files):
    """Lazily load a sequence of preprocessed `.pt` tensors from disk.

    Args:
        pt_files (Iterable[pathlib.Path]): Paths to `.pt` files to load.

    Yields:
        torch.Tensor: The tensor stored in each `.pt` file, in order.
    """
    for pt_path in pt_files:
        yield torch.load(pt_path, weights_only=False)


def check_landsat_range(landsat_dir: str, max_images: int = 500):
    """Sample raw Landsat GeoTIFFs, apply LANDSAT_NORM, and print per-channel value-range stats.

    Args:
        landsat_dir (str): Root directory containing a `fmow_landsat/images/`
            subdirectory with `image_*.tif` GeoTIFFs.
        max_images (int): Maximum number of files to sample and check. Defaults to 500.
    """
    images_dir = Path(landsat_dir) / "fmow_landsat" / "images"
    tif_files = _sample_files(list(images_dir.glob("image_*.tif")), max_images)
    print(f"Checking {len(tif_files)} raw Landsat GeoTIFFs from {images_dir}\n")

    def load():
        for tif_path in tif_files:
            with rasterio.open(tif_path) as src:
                data = src.read().astype(np.float32)
            yield LANDSAT_NORM(torch.from_numpy(data))

    stats = _gather_stats(load(), 6, "Scanning Landsat")
    _print_band_table(6, *stats, len(tif_files))


def check_landsat_preprocessed(preprocessed_dir: str, max_images: int = 500):
    """Sample cached, already-normalized Landsat `.pt` tensors and print per-channel value-range stats.

    Args:
        preprocessed_dir (str): Root directory containing a `landsat/`
            subdirectory with `image_*.pt` files.
        max_images (int): Maximum number of files to sample and check. Defaults to 500.
    """
    landsat_dir = Path(preprocessed_dir) / "landsat"
    pt_files = _sample_files(list(landsat_dir.glob("image_*.pt")), max_images)
    print(f"Checking {len(pt_files)} preprocessed Landsat .pt files from {landsat_dir}\n")

    stats = _gather_stats(_load_pt_tensors(pt_files), 6, "Scanning Landsat cached")
    _print_band_table(6, *stats, len(pt_files))


def check_rgb_range(fmow_dir: str, max_images: int = 500):
    """Sample raw FMoW RGB PNGs, apply RGB_NORM, and print per-channel value-range stats.

    Args:
        fmow_dir (str): Root directory containing a `fmow_v1.1/images/`
            subdirectory with `rgb_img_*.png` files.
        max_images (int): Maximum number of files to sample and check. Defaults to 500.
    """
    images_dir = Path(fmow_dir) / "fmow_v1.1" / "images"
    png_files = _sample_files(list(images_dir.glob("rgb_img_*.png")), max_images)
    print(f"Checking {len(png_files)} raw FMoW RGB PNGs from {images_dir}\n")

    def load():
        for png_path in png_files:
            img = Image.open(png_path).convert("RGB")
            yield RGB_NORM(img)

    stats = _gather_stats(load(), 3, "Scanning RGB")
    _print_band_table(3, *stats, len(png_files))


def check_rgb_preprocessed(preprocessed_dir: str, max_images: int = 500):
    """Sample cached, already-normalized RGB `.pt` tensors and print per-channel value-range stats.

    Args:
        preprocessed_dir (str): Root directory containing a `fmow_rgb/`
            subdirectory with `rgb_img_*.pt` files.
        max_images (int): Maximum number of files to sample and check. Defaults to 500.
    """
    rgb_dir = Path(preprocessed_dir) / "fmow_rgb"
    pt_files = _sample_files(list(rgb_dir.glob("rgb_img_*.pt")), max_images)
    print(f"Checking {len(pt_files)} preprocessed RGB .pt files from {rgb_dir}\n")

    stats = _gather_stats(_load_pt_tensors(pt_files), 3, "Scanning RGB cached")
    _print_band_table(3, *stats, len(pt_files))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Check normalized image value ranges for Landsat and/or RGB"
    )
    parser.add_argument("--landsat-dir", type=str, default="/home/datasets4/FMoW_LandSat")
    parser.add_argument("--fmow-dir", type=str, default="/home/henicke/data")
    parser.add_argument("--preprocessed-dir", type=str,
                        help="Path to preprocessed cache (e.g. /data/henicke/FMoW_LandSat/fmow_preprocessed). "
                             "If set, checks cached .pt files instead of raw images.")
    parser.add_argument("--modality", choices=["landsat", "rgb", "both"], default="both",
                        help="Which modality to check (default: both)")
    parser.add_argument("--max-images", type=int, default=1000)
    args = parser.parse_args()

    check_landsat = args.modality in ("landsat", "both")
    check_rgb = args.modality in ("rgb", "both")

    if args.preprocessed_dir:
        if check_landsat:
            check_landsat_preprocessed(args.preprocessed_dir, args.max_images)
        if check_rgb:
            if check_landsat:
                print("\n" + "=" * 65 + "\n")
            check_rgb_preprocessed(args.preprocessed_dir, args.max_images)
    else:
        if check_landsat:
            check_landsat_range(args.landsat_dir, args.max_images)
        if check_rgb:
            if check_landsat:
                print("\n" + "=" * 65 + "\n")
            check_rgb_range(args.fmow_dir, args.max_images)
