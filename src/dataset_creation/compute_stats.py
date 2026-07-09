"""
compute_stats.py

Compute channel-wise mean and standard deviation for raw (untransformed)
FMoW RGB and Landsat images, using Welford's online algorithm for numerical
stability over large datasets.

Functions:
    compute_stats: Main entry point; discovers RGB/Landsat sample pairs on
        disk, accumulates per-channel statistics, prints/returns them, and
        optionally writes them to a JSON file.
    _load_raw_rgb / _load_raw_landsat: Helpers that load a single raw RGB PNG
        / Landsat GeoTIFF into a (C, H, W) tensor for `compute_stats`.

Run as a script with --fmow-dir/--landsat-dir/--output-json.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import rasterio
import torch
from PIL import Image
from tqdm import tqdm
from welford_torch import Welford


def _load_raw_rgb(rgb_path):
    """Load a raw FMoW RGB PNG as a normalized float tensor.

    Args:
        rgb_path (str | pathlib.Path): Path to the RGB PNG file.

    Returns:
        torch.Tensor: Image tensor of shape (3, H, W), dtype float32, scaled to [0, 1].
    """
    rgb_np = np.asarray(Image.open(rgb_path).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(rgb_np).permute(2, 0, 1)  # Permute: (H, W, C) -> (C, H, W)


def _load_raw_landsat(landsat_path):
    """Load a raw Landsat GeoTIFF as a float tensor of its raw band values.

    Args:
        landsat_path (str | pathlib.Path): Path to the Landsat GeoTIFF file.

    Returns:
        torch.Tensor: Image tensor of shape (6, H, W), dtype float32, containing
            the raw (unscaled) GeoTIFF band values.
    """
    with rasterio.open(landsat_path) as src:
        landsat_np = src.read().astype(np.float32)
    return torch.from_numpy(landsat_np)


def compute_stats(
    fmow_dir="data",
    landsat_dir="data",
    output_json=None,
):
    """Compute per-channel mean/std for untransformed RGB and Landsat samples.

    Discovers sample indices from `landsat_dir/fmow_landsat/images/image_*.tif`,
    then for each index loads the matching raw RGB PNG
    (`fmow_dir/fmow_v1.1/images/rgb_img_<idx>.png`) and Landsat GeoTIFF,
    skipping any index where a file is missing or has an unexpected channel
    count (RGB must have 3 channels, Landsat must have 6). Statistics are
    accumulated online via `Welford` (numerically stable, single pass) over
    all pixels, printed to stdout, and optionally written to a JSON file.

    Args:
        fmow_dir (str): Directory containing the FMoW dataset (expects a
            `fmow_v1.1/images/` subdirectory). Defaults to "data".
        landsat_dir (str): Directory containing the Landsat dataset (expects
            a `fmow_landsat/images/` subdirectory). Defaults to "data".
        output_json (str | None): Optional path to write the computed stats
            as JSON. If None, stats are not written to disk.

    Returns:
        dict: Statistics with keys `value_space`, `counts` (`processed` and
            `skipped` sample counts), `rgb` (`mean`/`std` lists of length 3),
            and `landsat` (`mean`/`std` lists of length 6).

    Raises:
        ValueError: If no Landsat `.tif` files are found in
            `landsat_dir/fmow_landsat/images/`.
        RuntimeError: If no sample was processed successfully (e.g. all
            indices were skipped due to missing files or unexpected channel
            counts).
    """
    landsat_images_dir = Path(landsat_dir) / "fmow_landsat" / "images"
    idxs = []

    if landsat_images_dir.exists():
        for img_file in landsat_images_dir.glob("image_*.tif"):
            idx_str = img_file.stem.replace("image_", "")
            try:
                idxs.append(int(idx_str))
            except ValueError:
                continue

    if not idxs:
        raise ValueError(f"No Landsat files found in {landsat_images_dir}")

    rgb_images_dir = Path(fmow_dir) / "fmow_v1.1" / "images"

    rgb_stats = Welford()
    landsat_stats = Welford()

    processed = 0
    skipped = 0

    print(f"Using raw RGB directory: {rgb_images_dir}")
    print(f"Using raw Landsat directory: {landsat_images_dir}")
    print(f"Found {len(idxs)} indices in landsat directory")

    with tqdm(total=len(idxs), desc="Computing stats") as pbar:
        for idx in idxs:
            try:
                rgb_path = rgb_images_dir / f"rgb_img_{idx}.png"
                landsat_path = landsat_images_dir / f"image_{idx}.tif"

                if not rgb_path.exists() or not landsat_path.exists():
                    skipped += 1
                    pbar.update(1)
                    continue

                rgb = _load_raw_rgb(rgb_path)
                landsat = _load_raw_landsat(landsat_path)

                if rgb.shape[0] != 3 or landsat.shape[0] != 6:
                    skipped += 1
                    print(
                        f"Skipping index {idx}: unexpected channels "
                        f"(rgb={rgb.shape[0]}, landsat={landsat.shape[0]})"
                    )
                    pbar.update(1)
                    continue
  
                # Reshape (C, H, W) -> (H*W, C) for Welford
                # Welford uses first dimension as batch dimension
                rgb_stats.add_all(rgb.permute(1, 2, 0).reshape(-1, 3))
                landsat_stats.add_all(landsat.permute(1, 2, 0).reshape(-1, 6))

                processed += 1
            except Exception as e:
                skipped += 1
                print(f"Error processing index {idx}: {e}")

            pbar.update(1)

    if processed == 0:
        raise RuntimeError("No samples were processed successfully.")

    rgb_mean, rgb_std = rgb_stats.mean, np.sqrt(rgb_stats.var_p)
    landsat_mean, landsat_std = landsat_stats.mean, np.sqrt(landsat_stats.var_p)

    stats = {
        "value_space": {
            "rgb": "raw uint8-like pixel values (0-255)",
            "landsat": "raw GeoTIFF band values",
        },
        "counts": {
            "processed": processed,
            "skipped": skipped,
        },
        "rgb": {
            "mean": [float(v) for v in rgb_mean.tolist()],
            "std": [float(v) for v in rgb_std.tolist()],
        },
        "landsat": {
            "mean": [float(v) for v in landsat_mean.tolist()],
            "std": [float(v) for v in landsat_std.tolist()],
        },
    }

    print("\nComputed untransformed dataset statistics:")
    print("RGB")
    print(f"  mean: {stats['rgb']['mean']}")
    print(f"  std:  {stats['rgb']['std']}")
    print("Landsat")
    print(f"  mean: {stats['landsat']['mean']}")
    print(f"  std:  {stats['landsat']['std']}")
    print(
        f"Processed: {processed}, Skipped: {skipped}, "
    )

    if output_json is not None:
        output_path = Path(output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)
        print(f"\nSaved stats JSON to: {output_path}")

    return stats



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute channel-wise mean/std for untransformed RGB and Landsat images"
    )
    parser.add_argument(
        "--fmow-dir",
        type=str,
        default="/home/henicke/data",
        help="Directory containing FMoW dataset",
    )
    parser.add_argument(
        "--landsat-dir",
        type=str,
        default="/home/datasets4/FMoW_LandSat",
        help="Directory containing Landsat dataset",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Optional path to save computed stats as JSON",
    )

    args = parser.parse_args()

    compute_stats(
        fmow_dir=args.fmow_dir,
        landsat_dir=args.landsat_dir,
        output_json=args.output_json,
    )