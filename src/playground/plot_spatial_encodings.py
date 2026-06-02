"""Generate three presentation-ready figures for spatial encoding visualisation.

Loads a single (HR, LR) pair from the preprocessed cache and produces:
  1. Position encodings -- 2x2 grid (HR/LR x X/Y)
  2. Fourier encodings   -- 4x8 grid (4 freq bands, LR top / HR bottom, x left / y right)
  3. Gaussian overlap mask

Usage:
    uv run python src/dataset_creation/display_spatial_encoding_slides.py \
        --preprocessed-dir /path/to/fmow_preprocessed \
        --metadata-csv /home/datasets4/FMoW_LandSat/fmow_landsat/rgb_metadata_extended.csv
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch

IMG_SIZE = 224


def _get_index(path: Path) -> int | None:
    stem = path.stem
    for prefix in ("image_", "rgb_img_"):
        if stem.startswith(prefix):
            try:
                return int(stem[len(prefix):])
            except ValueError:
                pass
    return None


def build_coord_grids(img_span_km: float, lr_span_km: float):
    S = IMG_SIZE
    lr_res = lr_span_km * 1000.0 / S
    hr_res = img_span_km * 1000.0 / S
    center = S / 2.0
    offset = center * lr_res - center * hr_res
    coord_scale = (S - 1) * lr_res / 2.0
    coord_center = coord_scale

    pixel = torch.arange(S, dtype=torch.float32)
    lr_py, lr_px = torch.meshgrid(pixel * lr_res, pixel * lr_res, indexing="ij")
    coord_grid_lr = (torch.stack([lr_px, lr_py], dim=0) - coord_center) / coord_scale

    hr_py, hr_px = torch.meshgrid(pixel * hr_res + offset, pixel * hr_res + offset, indexing="ij")
    coord_grid_hr = (torch.stack([hr_px, hr_py], dim=0) - coord_center) / coord_scale

    return coord_grid_hr, coord_grid_lr


def build_fourier_channels(coord_grid: torch.Tensor, num_bands: int) -> torch.Tensor:
    """Returns (4, num_bands, H, W): [sin_x, cos_x, sin_y, cos_y] x bands."""
    freqs = math.pi * (2.0 ** torch.arange(num_bands, dtype=torch.float32))
    x, y = coord_grid[0:1], coord_grid[1:2]
    f = freqs[:, None, None]
    return torch.stack([
        torch.sin(f * x),
        torch.cos(f * x),
        torch.sin(f * y),
        torch.cos(f * y),
    ], dim=0)


def build_gaussian_overlap_mask(img_span_km: float, lr_span_km: float):
    S = IMG_SIZE
    ratio = img_span_km / lr_span_km
    sigma = ratio / 2.0
    lin = torch.linspace(-1, 1, S)
    gy, gx = torch.meshgrid(lin, lin, indexing="ij")
    return torch.exp(-(gx ** 2 + gy ** 2) / (2 * sigma ** 2))


def plot_coord_grids(hr_grid, lr_grid):
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.5), constrained_layout=True)
    titles = ["LR", "HR"]
    grids = [lr_grid[0], hr_grid[0]]

    for col, (title, grid) in enumerate(zip(titles, grids)):
        ax = axes[col]
        im = ax.imshow(grid.cpu(), cmap="viridis", vmin=-1, vmax=1)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.axis("off")

    fig.colorbar(im, ax=axes, fraction=0.03, pad=0.02)
    fig.suptitle("Position Encoding", fontsize=16, fontweight="bold")
    return fig


def plot_fourier_grid(hr_fourier, lr_fourier):
    """4x4 grid: rows = [LR sin, LR cos, HR sin, HR cos], cols = [f₁..f₄] (x-axis only)."""
    num_bands = hr_fourier.shape[1]
    fig, axes = plt.subplots(4, num_bands, figsize=(1.6 * num_bands + 1.4, 1.6 * 4),
                             constrained_layout=True)

    row_specs = [
        ("sin", lr_fourier, 0),
        ("cos", lr_fourier, 1),
        ("sin", hr_fourier, 0),
        ("cos", hr_fourier, 1),
    ]

    freqs = math.pi * (2.0 ** torch.arange(num_bands, dtype=torch.float32))
    freq_labels = []
    for b in range(num_bands):
        mult = int(2 ** b)
        freq_labels.append(f"f{chr(0x2080 + b + 1)} = {mult}π" if mult > 1 else f"f{chr(0x2080 + b + 1)} = π")

    im = None
    for row_idx, (row_label, fourier, trig_idx) in enumerate(row_specs):
        for band_idx in range(num_bands):
            ax = axes[row_idx, band_idx]
            im = ax.imshow(fourier[trig_idx, band_idx].cpu(), cmap="RdBu_r", vmin=-1, vmax=1)
            ax.axis("off")
            if row_idx == 0:
                ax.set_title(freq_labels[band_idx], fontsize=12, fontweight="bold")

        axes[row_idx, 0].set_ylabel(
            row_label, fontsize=11, rotation=0, labelpad=45, va="center",
        )
        axes[row_idx, 0].yaxis.set_visible(True)

    fig.text(0.01, 0.70, "LR", fontsize=14, fontweight="bold", va="center", rotation=90)
    fig.text(0.01, 0.23, "HR", fontsize=14, fontweight="bold", va="center", rotation=90)
    fig.colorbar(im, ax=axes, fraction=0.05, pad=0.02)
    fig.suptitle(f"Frequency Encoding ({num_bands} bands)", fontsize=16, fontweight="bold")
    return fig


def plot_overlap_mask(mask):
    fig, ax = plt.subplots(1, 1, figsize=(5, 5), constrained_layout=True)
    im = ax.imshow(mask.cpu(), cmap="gray", vmin=0, vmax=1)
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.05, pad=0.02)
    fig.suptitle("Gaussian Overlap Mask", fontsize=16, fontweight="bold")
    return fig


def main():
    parser = argparse.ArgumentParser(
        description="Generate spatial encoding figures for presentation slides"
    )
    parser.add_argument("--preprocessed-dir", type=str, required=True)
    parser.add_argument("--metadata-csv", type=str,
                        default="/home/datasets4/FMoW_LandSat/fmow_landsat/rgb_metadata_extended.csv")
    parser.add_argument("--lr-extension-factor", type=float, default=3.0)
    parser.add_argument("--sample-index", type=int, default=None,
                        help="Specific sample index to display (default: first available)")
    args = parser.parse_args()

    base = Path(args.preprocessed_dir)
    landsat_dir = base / "landsat"
    rgb_dir = base / "fmow_rgb"

    landsat_by_idx = {_get_index(p): p for p in landsat_dir.glob("image_*.pt") if _get_index(p) is not None}
    rgb_by_idx = {_get_index(p): p for p in rgb_dir.glob("rgb_img_*.pt") if _get_index(p) is not None}
    common_idxs = sorted(set(landsat_by_idx) & set(rgb_by_idx))

    if not common_idxs:
        raise FileNotFoundError(f"No matching index pairs found in {landsat_dir} and {rgb_dir}")

    if args.sample_index is not None:
        if args.sample_index not in common_idxs:
            raise ValueError(f"Sample index {args.sample_index} not available. Choose from: {common_idxs[:20]}...")
        idx = args.sample_index
    else:
        idx = common_idxs[0]

    df = pd.read_csv(args.metadata_csv)
    df = df[df["split"] != "seq"].reset_index(drop=True)
    max_hr_span = df["img_span_km"].max()
    lr_span_km = max_hr_span * args.lr_extension_factor

    if idx >= len(df):
        raise ValueError(f"Sample index {idx} out of range for metadata ({len(df)} rows)")
    img_span_km = df.loc[idx, "img_span_km"]

    print(f"Sample {idx}: img_span_km={img_span_km:.4f}, lr_span_km={lr_span_km:.4f}")

    hr_grid, lr_grid = build_coord_grids(img_span_km, lr_span_km)
    hr_fourier = build_fourier_channels(hr_grid, num_bands=4)
    lr_fourier = build_fourier_channels(lr_grid, num_bands=4)
    mask = build_gaussian_overlap_mask(img_span_km, lr_span_km)

    repo_root = Path(__file__).resolve().parents[2]
    saves = [
        ("coord_encoding.png", plot_coord_grids(hr_grid, lr_grid)),
        ("frequency_encoding.png", plot_fourier_grid(hr_fourier, lr_fourier)),
        ("overlap_mask.png", plot_overlap_mask(mask)),
    ]

    for filename, fig in saves:
        path = repo_root / filename
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
