"""
save_transformed_images.py

Preprocesses and saves both RGB (HR) and Landsat (LR) images from
`FMoWMultiScaleDataset` as normalized PyTorch tensors (`.pt` files), so training can
load pre-transformed tensors instead of raw images/GeoTIFFs. Unlike
`save_fullres_landsat.py`, which stores only the Landsat branch at native resolution,
this script saves both branches at the dataset's standard (resized) resolution.

Main function:
    - save_transformed_images: Iterates over all Landsat-indexed samples, applies the
      dataset's transforms/normalization, and writes each sample's RGB and Landsat
      tensors to `<output_dir>/fmow_preprocessed/{fmow_rgb,landsat}/`.
"""

import argparse
from pathlib import Path
import torch
from tqdm import tqdm
from dataset.fmow_multiscale_dataset import FMoWMultiScaleDataset


def save_transformed_images(
    fmow_dir="data",
    landsat_dir="data",
    output_dir="data",
    batch_size=1,
    image_norm="fmow-statistics",
):
    """Load all images from the dataset, apply transforms, and save as .pt files.

    Iterates over every sample index found in `landsat_dir`'s GeoTIFF filenames, builds
    an `FMoWMultiScaleDataset`, and for each not-yet-preprocessed index runs it through
    the dataset (applying its RGB/Landsat transforms and normalization), buffering
    `batch_size` samples before writing them out as individual `.pt` tensor files.

    Args:
        fmow_dir (str): Directory containing the FMoW dataset.
        landsat_dir (str): Directory containing the Landsat dataset (GeoTIFFs read from
            `<landsat_dir>/fmow_landsat/images`).
        output_dir (str): Directory under which `fmow_preprocessed/{fmow_rgb,landsat}`
            are created and the `.pt` files are saved.
        batch_size (int): Number of samples to buffer in memory before writing them to
            disk.
        image_norm (str): Normalization scheme forwarded to `FMoWMultiScaleDataset`.
    """
    
    # Create output directories
    output_rgb_dir = Path(output_dir) / "fmow_preprocessed" / "fmow_rgb"
    output_landsat_dir = Path(output_dir) / "fmow_preprocessed" / "landsat"
    output_rgb_dir.mkdir(parents=True, exist_ok=True)
    output_landsat_dir.mkdir(parents=True, exist_ok=True)
    
    # Read indices from landsat image filenames
    landsat_images_dir = Path(landsat_dir) / "fmow_landsat" / "images"
    idxs = []
    if landsat_images_dir.exists():
        for img_file in landsat_images_dir.glob("image_*.tif"):
            idx_str = img_file.stem.replace("image_", "")
            try:
                idxs.append(int(idx_str))
            except ValueError:
                continue
    
    print(f"Loading dataset from {fmow_dir}...")
    dataset = FMoWMultiScaleDataset(
        fmow_dir=fmow_dir,
        landsat_dir=landsat_dir,
        image_norm=image_norm,
    )
    
    print(f"Found {len(idxs)} images in landsat directory")
    print(f"Saving RGB images to: {output_rgb_dir}")
    print(f"Saving Landsat images to: {output_landsat_dir}\n")
     
    # Process all images with progress bar
    rgb_list = []
    landsat_list = []
    indices = []
    
    with tqdm(total=len(idxs), desc="Processing images") as pbar:
        for idx in idxs:
            # Skip if already preprocessed
            if (output_rgb_dir / f"rgb_img_{idx}.pt").exists() and \
               (output_landsat_dir / f"image_{idx}.pt").exists():
                pbar.update(1)
                continue
            
            try:
                x, _, _ = dataset[idx]
                rgb_list.append(x["rgb"])
                landsat_list.append(x["landsat"])
                indices.append(idx)
                
                pbar.update(1)
                
                # Save in batches to avoid memory issues
                if len(rgb_list) >= batch_size or idx == idxs[-1]:
                    for i, orig_idx in enumerate(indices):
                        torch.save(rgb_list[i], output_rgb_dir / f"rgb_img_{orig_idx}.pt")
                        torch.save(landsat_list[i], output_landsat_dir / f"image_{orig_idx}.pt")
                    
                    rgb_list = []
                    landsat_list = []
                    indices = []
                    
            except Exception as e:
                print(f"Error processing index {idx}: {e}")
                pbar.update(1)
                continue
    
    print("\nSuccessfully saved all transformed images!")
    print(f" RGB images: {output_rgb_dir}")
    print(f" Landsat images: {output_landsat_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Save transformed RGB and Landsat images as PyTorch tensors"
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
        "--output-dir",
        type=str,
        default="/home/datasets4/FMoW_LandSat/fmow_preprocessed",
        help="Directory to save preprocessed .pt files",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Batch size for processing (higher = more memory usage)",
    )
    parser.add_argument(
        "--image-norm",
        type=str,
        default="fmow-statistics",
        choices=["fmow-statistics", "const", "imagenet-statistics"],
    )
    
    args = parser.parse_args()
    
    save_transformed_images(
        fmow_dir=args.fmow_dir,
        landsat_dir=args.landsat_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        image_norm=args.image_norm,
    )
