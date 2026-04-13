"""
Script to compute channel-wise mean and standard deviation for
FMoW RGB and Landsat images.
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
    rgb_np = np.asarray(Image.open(rgb_path).convert("RGB"), dtype=np.float32)
    return torch.from_numpy(rgb_np).permute(2, 0, 1)  # Permute: (H, W, C) -> (C, H, W)


def _load_raw_landsat(landsat_path):
    with rasterio.open(landsat_path) as src:
        landsat_np = src.read().astype(np.float32)
    return torch.from_numpy(landsat_np)


def _collect_indices(landsat_images_dir):
    idxs = []

    if landsat_images_dir.exists():
        for img_file in landsat_images_dir.glob("image_*.tif"):
            idx_str = img_file.stem.replace("image_", "")
            try:
                idxs.append(int(idx_str))
            except ValueError:
                continue

    return sorted(idxs)


def _add_channelwise_samples(stats, tensor):
    channel_count = tensor.shape[0]
    stats.add_all(tensor.movedim(0, -1).reshape(-1, channel_count))


def _summarize_stats(rgb_stats, landsat_stats, processed, skipped):
    rgb_mean, rgb_std = rgb_stats.mean, np.sqrt(rgb_stats.var_p)
    landsat_mean, landsat_std = landsat_stats.mean, np.sqrt(landsat_stats.var_p)

    return {
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


def _write_stats_json(stats, output_json):
    if output_json is None:
        return

    output_path = Path(output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(f"\nSaved stats JSON to: {output_path}")


def _compute_stats(idxs, load_rgb, load_landsat, description, value_space, output_json=None):
    rgb_stats = Welford()
    landsat_stats = Welford()

    processed = 0
    skipped = 0

    with tqdm(total=len(idxs), desc=description) as pbar:
        for idx in idxs:
            try:
                rgb = load_rgb(idx)
                landsat = load_landsat(idx)

                if rgb.shape[0] != 3 or landsat.shape[0] != 6:
                    skipped += 1
                    print(
                        f"Skipping index {idx}: unexpected channels "
                        f"(rgb={rgb.shape[0]}, landsat={landsat.shape[0]})"
                    )
                    pbar.update(1)
                    continue

                _add_channelwise_samples(rgb_stats, rgb)
                _add_channelwise_samples(landsat_stats, landsat)
                processed += 1
            except Exception as e:
                skipped += 1
                print(f"Error processing index {idx}: {e}")

            pbar.update(1)

    if processed == 0:
        raise RuntimeError("No samples were processed successfully.")

    stats = {
        "value_space": value_space,
        **_summarize_stats(rgb_stats, landsat_stats, processed, skipped),
    }

    print(f"\nComputed {description} dataset statistics:")
    print("RGB")
    print(f"  mean: {stats['rgb']['mean']}")
    print(f"  std:  {stats['rgb']['std']}")
    print("Landsat")
    print(f"  mean: {stats['landsat']['mean']}")
    print(f"  std:  {stats['landsat']['std']}")
    print(f"Processed: {processed}, Skipped: {skipped}")

    _write_stats_json(stats, output_json)
    return stats


def compute_untransformed_stats(
    fmow_dir="data",
    landsat_dir="data",
    output_json=None,
):
    """
    Compute per-channel mean/std for untransformed RGB and Landsat samples.

    Args:
        fmow_dir: Directory containing FMoW dataset.
        landsat_dir: Directory containing Landsat dataset.
        output_json: Optional path to write stats as JSON.
    """
    landsat_images_dir = Path(landsat_dir) / "fmow_landsat" / "images"
    idxs = _collect_indices(landsat_images_dir)

    if not idxs:
        raise ValueError(f"No Landsat files found in {landsat_images_dir}")

    rgb_images_dir = Path(fmow_dir) / "fmow_v1.1" / "images"

    print(f"Using raw RGB directory: {rgb_images_dir}")
    print(f"Using raw Landsat directory: {landsat_images_dir}")
    print(f"Found {len(idxs)} indices in landsat directory")
    return _compute_stats(
        idxs=idxs,
        load_rgb=lambda idx: _load_raw_rgb(rgb_images_dir / f"rgb_img_{idx}.png"),
        load_landsat=lambda idx: _load_raw_landsat(landsat_images_dir / f"image_{idx}.tif"),
        description="raw",
        value_space={
            "rgb": "raw uint8 pixel values (0-255)",
            "landsat": "raw GeoTIFF band values",
        },
        output_json=output_json,
    )


def compute_transformed_stats(
    fmow_dir="data",
    landsat_dir="data",
    output_json=None,
):
    from dataset.fmow_multiscale_dataset import FMoWMultiScaleDataset

    landsat_images_dir = Path(landsat_dir) / "fmow_landsat" / "images"
    idxs = _collect_indices(landsat_images_dir)

    if not idxs:
        raise ValueError(f"No Landsat files found in {landsat_images_dir}")

    dataset = FMoWMultiScaleDataset(
        fmow_dir=fmow_dir,
        landsat_dir=landsat_dir,
        augment=False,
    )

    print(f"Using transformed RGB/Landsat dataset from: {fmow_dir}, {landsat_dir}")
    print(f"Found {len(idxs)} indices in landsat directory")

    return _compute_stats(
        idxs=idxs,
        load_rgb=lambda idx: dataset.get_rgb_input(idx),
        load_landsat=lambda idx: dataset.get_landsat_input(idx),
        description="transformed",
        value_space={
            "rgb": "dataset-transformed RGB tensors",
            "landsat": "dataset-transformed Landsat tensors",
        },
        output_json=output_json,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute channel-wise mean/std for raw RGB and Landsat images"
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

    compute_untransformed_stats(
        fmow_dir=args.fmow_dir,
        landsat_dir=args.landsat_dir,
        output_json=args.output_json,
    )