"""Display preprocessed .pt images by reversing normalization.

Uses the same directory layout and argument structure as check_image_stats.py.
Landsat reflectance is clipped to [0, 0.3] for display (matching
translate_geotiff_to_png.py); RGB is clamped to [0, 1].

Usage:
    uv run python src/dataset_creation/display_preprocessed_images.py --preprocessed-dir /path/to/cache --modality both
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch

LANDSAT_MEAN = torch.tensor([0.06259285658597946, 0.0880340114235878, 0.09441816806793213,
                             0.2327403724193573, 0.19073842465877533, 0.12976829707622528])
LANDSAT_STD = torch.tensor([0.039894334971904755, 0.049978554248809814, 0.0687960833311081,
                            0.092967689037323, 0.09390033036470413, 0.0819208025932312])

RGB_MEAN = torch.tensor([0.4155880808830261, 0.41815927624702454, 0.3903605341911316])
RGB_STD = torch.tensor([0.24812281131744385, 0.24405813217163086, 0.2482403963804245])


def _load_pt(path: Path) -> torch.Tensor:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _unnormalize(tensor: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    c = tensor.shape[0]
    return tensor * std[:c].view(c, 1, 1) + mean[:c].view(c, 1, 1)


def _landsat_to_display(tensor: torch.Tensor) -> torch.Tensor:
    raw = _unnormalize(tensor, LANDSAT_MEAN, LANDSAT_STD)
    bgr = raw[:3]
    rgb = bgr[[2, 1, 0]]
    rgb = (rgb / 0.3).clamp(0, 1)
    return rgb.permute(1, 2, 0)


def _rgb_to_display(tensor: torch.Tensor) -> torch.Tensor:
    raw = _unnormalize(tensor, RGB_MEAN, RGB_STD)
    return raw.clamp(0, 1).permute(1, 2, 0)


def _get_index(path: Path) -> int | None:
    stem = path.stem
    for prefix in ("image_", "rgb_img_"):
        if stem.startswith(prefix):
            try:
                return int(stem[len(prefix):])
            except ValueError:
                pass
    return None


def display_images(preprocessed_dir: str, modality: str, max_images: int):
    base = Path(preprocessed_dir)
    landsat_dir = base / "landsat"
    rgb_dir = base / "fmow_rgb"

    show_landsat = modality in ("landsat", "both")

    if modality == "both":
        landsat_by_idx = {_get_index(p): p for p in landsat_dir.glob("image_*.pt") if _get_index(p) is not None}
        rgb_by_idx = {_get_index(p): p for p in rgb_dir.glob("rgb_img_*.pt") if _get_index(p) is not None}
        common_idxs = sorted(set(landsat_by_idx) & set(rgb_by_idx))[:max_images]
        if not common_idxs:
            raise FileNotFoundError(f"No matching index pairs found in {landsat_dir} and {rgb_dir}")
        print(f"Displaying {len(common_idxs)} paired images (Landsat + RGB)")

        plt.ion()
        fig, axes = plt.subplots(1, 2, figsize=(14, 7))
        for i, idx in enumerate(common_idxs, start=1):
            ls_img = _landsat_to_display(_load_pt(landsat_by_idx[idx]))
            rgb_img = _rgb_to_display(_load_pt(rgb_by_idx[idx]))
            axes[0].clear()
            axes[0].imshow(ls_img)
            axes[0].set_title(f"Landsat — image_{idx}.pt")
            axes[0].axis("off")
            axes[1].clear()
            axes[1].imshow(rgb_img)
            axes[1].set_title(f"RGB — rgb_img_{idx}.pt")
            axes[1].axis("off")
            fig.suptitle(f"{i}/{len(common_idxs)}", fontsize=12)
            fig.tight_layout()
            fig.canvas.draw_idle()
            plt.show(block=False)
            plt.pause(0.001)
            if i < len(common_idxs):
                input("Press Enter for the next pair...")
    else:
        if show_landsat:
            folder = landsat_dir
            pattern = "image_*.pt"
            to_display = _landsat_to_display
            label = "Landsat"
        else:
            folder = rgb_dir
            pattern = "rgb_img_*.pt"
            to_display = _rgb_to_display
            label = "RGB"

        files = sorted(folder.glob(pattern))[:max_images]
        if not files:
            raise FileNotFoundError(f"No {pattern} files found in {folder}")
        print(f"Displaying {len(files)} {label} images")

        plt.ion()
        fig, ax = plt.subplots(figsize=(8, 8))
        for i, path in enumerate(files, start=1):
            img = to_display(_load_pt(path))
            ax.clear()
            ax.imshow(img)
            ax.set_title(f"{i}/{len(files)}: {path.name}", fontsize=10)
            ax.axis("off")
            fig.tight_layout()
            fig.canvas.draw_idle()
            plt.show(block=False)
            plt.pause(0.001)
            if i < len(files):
                input("Press Enter for the next image...")

    plt.ioff()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Display preprocessed .pt images by reversing normalization"
    )
    parser.add_argument("--preprocessed-dir", type=str, required=True,
                        help="Path to preprocessed cache (contains landsat/ and fmow_rgb/ subdirs)")
    parser.add_argument("--modality", choices=["landsat", "rgb", "both"], default="both",
                        help="Which modality to display (default: both)")
    parser.add_argument("--max-images", type=int, default=1000)
    args = parser.parse_args()

    display_images(args.preprocessed_dir, args.modality, args.max_images)
