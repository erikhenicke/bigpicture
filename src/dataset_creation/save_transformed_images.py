"""
Script to preprocess and save all RGB and Landsat images as PyTorch tensors.
This converts images to normalized .pt files for faster loading during training.
"""

import argparse
from pathlib import Path
import torch
from tqdm import tqdm
from dataset.fmow_multiscale_dataset import FMoWMultiScaleDataset


def save_transformed_images(
    fmow_dir="data",
    landsat_dir="data",
    output_dir="data/preprocessed",
    batch_size=1,
):
    """
    Load all images from the dataset, apply transforms, and save as .pt files.
    
    Args:
        fmow_dir: Directory containing FMoW dataset
        landsat_dir: Directory containing Landsat dataset
        output_dir: Directory to save preprocessed .pt files
        batch_size: Number of images to process before saving
    """
    
    # Create output directories
    output_rgb_dir = Path(output_dir) / "fmow_rgb"
    output_landsat_dir = Path(output_dir) / "landsat"
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
    )
    
    print(f"Found {len(idxs)} images in landsat directory")
    print(f"Saving RGB images to: {output_rgb_dir}")
    print(f"Saving Landsat images to: {output_landsat_dir}")
    print()
    
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
    
    print("\n✓ Successfully saved all transformed images!")
    print(f"  RGB images: {output_rgb_dir}")
    print(f"  Landsat images: {output_landsat_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Save transformed RGB and Landsat images as PyTorch tensors"
    )
    parser.add_argument(
        "--fmow-dir",
        type=str,
        default="/home/henicke/git/bigpicture/data",
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
    
    args = parser.parse_args()
    
    save_transformed_images(
        fmow_dir=args.fmow_dir,
        landsat_dir=args.landsat_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
    )
