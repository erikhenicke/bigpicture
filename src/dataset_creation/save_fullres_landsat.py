"""
Preprocess and save Landsat images at their *native* resolution (498x498),
normalized but NOT downscaled to 224x224.

This is the storage step for the spatial-extent ablation: keeping the full-res
images lets us later crop an arbitrary centered extent and resize *that* to 224
at load time (see the dataset's scale_to_img_size flag). The standard 224 set
has already thrown away this detail, so it cannot be re-cropped — hence a
separate full-res set.

Only the Landsat (LR) branch is saved; the HR/RGB branch stays at 224 and is
unaffected by the extent experiment. Files are written as fp16 tensors named
``image_{file_idx}.pt`` — keyed by the original FMoW index (the tif filename
number), matching how source="preprocessed" loads them.

Run (PYTHONPATH=src is required, see project memory):
    PYTHONPATH=src uv run --env-file .env \
        src/dataset_creation/save_fullres_landsat.py
"""

import argparse
from pathlib import Path

import torch
from tqdm import tqdm

from dataset.fmow_multiscale_dataset import FMoWMultiScaleDataset


def save_fullres_landsat(
    fmow_dir="/home/henicke/data",
    landsat_dir="/home/datasets4/FMoW_LandSat",
    output_dir="/home/datasets4/FMoW_LandSat/fmow_preprocessed",
    image_norm="fmow-statistics",
    limit=None,
):
    """Save every Landsat GeoTIFF as a normalized, full-resolution fp16 tensor.

    Args:
        fmow_dir: Directory containing the FMoW dataset (needed to build the
            dataset's metadata; only the LR branch is read from disk).
        landsat_dir: Directory containing the Landsat GeoTIFFs.
        output_dir: Base dir; tensors are written under ``<output_dir>/landsat``.
        image_norm: Normalization scheme; must match the runs that will consume
            this set (default "fmow-statistics").
        limit: If set, only process the first ``limit`` images (for quick tests).
    """
    output_landsat_dir = Path(output_dir) / "landsat"
    output_landsat_dir.mkdir(parents=True, exist_ok=True)

    # The tif filenames are keyed by the original FMoW index (file_idx); that is
    # exactly the index get_landsat_input() expects and the key the preprocessed
    # loader reads back, so we iterate them directly.
    landsat_images_dir = Path(landsat_dir) / "fmow_landsat" / "images"
    file_idxs = sorted(
        int(f.stem.replace("image_", ""))
        for f in landsat_images_dir.glob("image_*.tif")
        if f.stem.replace("image_", "").isdigit()
    )
    if limit is not None:
        file_idxs = file_idxs[:limit]

    print(f"Loading dataset metadata from {fmow_dir}...")
    # source="raw" + scale_to_img_size=False -> native-res, normalized LR tensors.
    dataset = FMoWMultiScaleDataset(
        fmow_dir=fmow_dir,
        landsat_dir=landsat_dir,
        source="raw",
        scale_to_img_size=False,
        image_norm=image_norm,
    )

    print(f"Found {len(file_idxs)} Landsat images")
    print(f"Saving full-res Landsat tensors (fp16) to: {output_landsat_dir}\n")

    saved = 0
    skipped = 0
    failed = 0
    logged_shape = False
    for file_idx in tqdm(file_idxs, desc="Saving full-res Landsat"):
        out_path = output_landsat_dir / f"image_{file_idx}.pt"
        if out_path.exists():
            skipped += 1
            continue

        try:
            landsat = dataset.get_landsat_input(file_idx)  # (6, H, W), fp32, normalized
            if not logged_shape:
                print(f"First tensor: shape={tuple(landsat.shape)}, dtype={landsat.dtype}")
                logged_shape = True
            torch.save(landsat.half().contiguous(), out_path)
            saved += 1
        except Exception as e:  # noqa: BLE001 - log and continue over a large batch
            print(f"Error processing index {file_idx}: {e}")
            failed += 1

    print(f"\nDone. saved={saved}, skipped(existing)={skipped}, failed={failed}")
    print(f"Full-res Landsat images: {output_landsat_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Save Landsat images at native resolution as normalized fp16 tensors."
    )
    parser.add_argument("--fmow-dir", type=str, default="/home/henicke/data")
    parser.add_argument("--landsat-dir", type=str, default="/home/datasets4/FMoW_LandSat")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/home/datasets4/FMoW_LandSat/fmow_preprocessed",
        help="Base dir; tensors are written under <output-dir>/landsat",
    )
    parser.add_argument(
        "--image-norm",
        type=str,
        default="fmow-statistics",
        choices=["fmow-statistics", "const", "imagenet-statistics"],
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N images (for a quick test run).",
    )

    args = parser.parse_args()

    save_fullres_landsat(
        fmow_dir=args.fmow_dir,
        landsat_dir=args.landsat_dir,
        output_dir=args.output_dir,
        image_norm=args.image_norm,
        limit=args.limit,
    )
