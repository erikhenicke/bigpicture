"""
display_spatial_encoding.py

Display preprocessed FMoW RGB + Landsat images alongside spatial encoding tensors.

Loads .pt files directly from the preprocessed cache (no WILDS dependency) and
computes spatial tensors from the metadata CSV. Shows a multi-panel figure per
sample: RGB, Landsat, coordinate grids (HR/LR), and overlap mask.

Standalone reimplementation (for visualization only) of the spatial-encoding
logic in `src/dataset/fmow_multiscale_dataset.py`
(`FMoWMultiscaleDataset._build_spatial_tensors`, coordinate grids and overlap
mask) and `src/models/components/spatial_encoding.py`
(`SpatialEncoding.forward`, Fourier features).

Functions:
    build_coord_grids: Build the normalized HR/LR pixel-coordinate grids for
        one sample.
    build_fourier_channels: Compute raw sin/cos Fourier features from a
        coordinate grid (before the learned projection `SpatialEncoding`
        would apply).
    build_overlap_mask: Build the binary/Gaussian mask marking where the HR
        crop overlaps the LR image.
    display: Main plotting loop; for each sample builds the requested panels
        (RGB, Landsat, coord grids, overlap mask, Fourier channel grid) and
        shows them interactively.
    _load_pt / _unnorm / _rgb_to_display / _landsat_to_display / _get_index:
        Helpers for loading `.pt` tensors, reversing normalization for
        display, and parsing sample indices from filenames.

Usage:
    uv run python src/dataset_creation/display_spatial_encoding.py \
        --preprocessed-dir /path/to/fmow_preprocessed \
        --metadata-csv /home/datasets4/FMoW_LandSat/fmow_landsat/rgb_metadata_extended.csv \
        --coord-channels --overlap-mask

    uv run python src/dataset_creation/display_spatial_encoding.py \
        --preprocessed-dir /path/to/fmow_preprocessed \
        --metadata-csv /home/datasets4/FMoW_LandSat/fmow_landsat/rgb_metadata_extended.csv \
        --fourier-channels --fourier-bands 4

    uv run python src/dataset_creation/display_spatial_encoding.py \
        --preprocessed-dir /path/to/fmow_preprocessed \
        --metadata-csv /home/datasets4/FMoW_LandSat/fmow_landsat/rgb_metadata_extended.csv \
        --overlap-mask --overlap-mask-type gaussian
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pandas as pd
import torch

IMG_SIZE = 224

RGB_MEAN = torch.tensor([0.4155880808830261, 0.41815927624702454, 0.3903605341911316])
RGB_STD = torch.tensor([0.24812281131744385, 0.24405813217163086, 0.2482403963804245])

LANDSAT_MEAN = torch.tensor([0.06259285658597946, 0.0880340114235878, 0.09441816806793213,
                             0.2327403724193573, 0.19073842465877533, 0.12976829707622528])
LANDSAT_STD = torch.tensor([0.039894334971904755, 0.049978554248809814, 0.0687960833311081,
                            0.092967689037323, 0.09390033036470413, 0.0819208025932312])


def _load_pt(path: Path) -> torch.Tensor:
    """Load a `.pt` tensor from disk onto CPU.

    Args:
        path (Path): Path to the `.pt` file.

    Returns:
        torch.Tensor: The loaded tensor.
    """
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _unnorm(tensor, mean, std):
    """Reverse channel-wise normalization: `tensor * std + mean`.

    Args:
        tensor (torch.Tensor): Normalized image tensor, shape (C, H, W).
        mean (torch.Tensor): Per-channel mean used for normalization, shape
            (>=C,); only the first C entries are used.
        std (torch.Tensor): Per-channel standard deviation used for
            normalization, shape (>=C,); only the first C entries are used.

    Returns:
        torch.Tensor: Unnormalized tensor, same shape as `tensor`.
    """
    c = tensor.shape[0]
    return tensor * std[:c].view(c, 1, 1) + mean[:c].view(c, 1, 1)


def _rgb_to_display(tensor):
    """Convert a normalized RGB tensor to a displayable image.

    Args:
        tensor (torch.Tensor): Normalized RGB tensor, shape (3, H, W), dtype float.

    Returns:
        torch.Tensor: Displayable RGB image, shape (H, W, 3), values in [0, 1].
    """
    return _unnorm(tensor, RGB_MEAN, RGB_STD).clamp(0, 1).permute(1, 2, 0)


def _landsat_to_display(tensor):
    """Convert a normalized Landsat tensor to a true-color image for display.

    Reverses LANDSAT_MEAN/LANDSAT_STD normalization, takes the first 3 bands
    (stored in BGR order), reorders them to RGB, and divides by 0.3 (clamped
    to [0, 1]) to bring typical reflectance values into displayable range.

    Args:
        tensor (torch.Tensor): Normalized Landsat tensor, shape (6, H, W), dtype float.

    Returns:
        torch.Tensor: Displayable RGB image, shape (H, W, 3), values in [0, 1].
    """
    raw = _unnorm(tensor, LANDSAT_MEAN, LANDSAT_STD)
    rgb = raw[:3][[2, 1, 0]]
    return (rgb / 0.3).clamp(0, 1).permute(1, 2, 0)


def _get_index(path: Path) -> int | None:
    """Parse the sample index out of a `.pt` filename.

    Recognizes `image_<idx>.pt` (Landsat) and `rgb_img_<idx>.pt` (RGB) filenames.

    Args:
        path (Path): Path whose stem is checked against the known prefixes.

    Returns:
        int | None: The parsed sample index, or None if the filename doesn't
            match either known prefix or the suffix isn't an integer.
    """
    stem = path.stem
    for prefix in ("image_", "rgb_img_"):
        if stem.startswith(prefix):
            try:
                return int(stem[len(prefix):])
            except ValueError:
                pass
    return None


def build_coord_grids(img_span_km: float, lr_span_km: float):
    """Build normalized HR/LR pixel-coordinate grids for one sample.

    Reimplements the coordinate-grid construction in
    `FMoWMultiscaleDataset._build_spatial_tensors` / `__init__` for
    standalone visualization: each grid holds (x, y) pixel-center
    coordinates (pixel index times ground resolution, in meters), shifted to
    center on the LR image and scaled by half the LR image's physical extent
    so LR coordinates land in [-1, 1]. The HR grid uses the same
    scale/center but its own (finer) ground resolution plus an offset that
    aligns it with the region of the LR image it was cropped from.

    Args:
        img_span_km (float): Physical width/height of the HR (high-resolution)
            image, in kilometers.
        lr_span_km (float): Physical width/height of the LR (low-resolution)
            image, in kilometers.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: `(coord_grid_hr, coord_grid_lr)`,
            each of shape (2, IMG_SIZE, IMG_SIZE) with channel 0 = x,
            channel 1 = y, dtype float32.
    """
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


FOURIER_COMPONENT_LABELS = ["sin(f·x)", "cos(f·x)", "sin(f·y)", "cos(f·y)"]


def build_fourier_channels(coord_grid: torch.Tensor, num_bands: int) -> torch.Tensor:
    """Compute raw Fourier positional-encoding channels from a coordinate grid.

    Standalone (unbatched, unprojected) version of the sin/cos feature
    computation in `SpatialEncoding.forward`: for each of `num_bands`
    frequencies `pi * 2**b`, computes sin/cos of `freq * x` and `freq * y`.
    Unlike `SpatialEncoding`, this does not apply the learned 1x1-conv
    projection to `fourier_proj_dim` channels.

    Args:
        coord_grid (torch.Tensor): Coordinate grid, shape (2, H, W) with
            channel 0 = x, channel 1 = y (as produced by `build_coord_grids`).
        num_bands (int): Number of frequency bands `L`.

    Returns:
        torch.Tensor: Shape (4, num_bands, H, W), dtype float32. Dim 0
            indexes the component in order [sin(f·x), cos(f·x), sin(f·y),
            cos(f·y)] (see `FOURIER_COMPONENT_LABELS`), dim 1 indexes the
            frequency band.
    """
    freqs = math.pi * (2.0 ** torch.arange(num_bands, dtype=torch.float32))
    x, y = coord_grid[0:1], coord_grid[1:2]  # (1, H, W)
    f = freqs[:, None, None]  # (L, 1, 1)
    return torch.stack([
        torch.sin(f * x),  # (L, H, W)
        torch.cos(f * x),
        torch.sin(f * y),
        torch.cos(f * y),
    ], dim=0)  # (4, L, H, W)


def build_overlap_mask(img_span_km: float, lr_span_km: float, mask_type: str):
    """Build the mask marking where the HR crop overlaps the LR image, for one sample.

    Reimplements the overlap-mask construction in
    `FMoWMultiscaleDataset._build_spatial_tensors` for standalone
    visualization. `ratio = img_span_km / lr_span_km` is the fraction of the
    LR image's extent covered by the HR crop; the mask is 1 within that
    square region and 0 outside it (`mask_type="binary"`), or a Gaussian
    bump with `sigma = ratio / 2` centered on the image
    (`mask_type="gaussian"`).

    Args:
        img_span_km (float): Physical width/height of the HR (high-resolution)
            image, in kilometers.
        lr_span_km (float): Physical width/height of the LR (low-resolution)
            image, in kilometers.
        mask_type (str): "gaussian" for a smooth Gaussian mask, anything else
            for a hard-edged binary mask.

    Returns:
        torch.Tensor: Mask of shape (IMG_SIZE, IMG_SIZE), dtype float32,
            values in [0, 1].
    """
    S = IMG_SIZE
    ratio = img_span_km / lr_span_km
    lin = torch.linspace(-1, 1, S)
    gy, gx = torch.meshgrid(lin, lin, indexing="ij")

    if mask_type == "gaussian":
        sigma = ratio / 2.0
        mask = torch.exp(-(gx ** 2 + gy ** 2) / (2 * sigma ** 2))
    else:
        mask = ((gx.abs() <= ratio) & (gy.abs() <= ratio)).float()

    return mask


def display(pairs, span_lookup, lr_span_km, show_coord, show_mask, mask_type,
            show_fourier, fourier_bands):
    """Interactively display RGB/Landsat images with optional spatial-encoding panels.

    For each `(idx, rgb_path, ls_path)` in `pairs`, always shows the RGB and
    Landsat true-color images. If `show_coord` is set, adds HR/LR coordinate
    grid panels (see `build_coord_grids`); if `show_mask` is set, adds the
    overlap mask panel (see `build_overlap_mask`); if `show_fourier` is set
    (and the sample's span is known), replaces the simple panel row with a
    grid showing the sin/cos Fourier channels (see `build_fourier_channels`)
    for both the HR and LR coordinate grids, one row per component/branch
    and one column per frequency band. Pauses for Enter between samples;
    entering "q" stops early. Samples whose index isn't in `span_lookup`
    skip the coord-grid, overlap-mask, and Fourier panels (span unknown).

    Args:
        pairs (list[tuple[int, Path, Path]]): `(sample_idx, rgb_path,
            landsat_path)` triples to display, in order.
        span_lookup (dict[int, float]): Sample index -> HR image span in
            kilometers (`img_span_km`), used to size the coordinate grids,
            overlap mask, and Fourier channels.
        lr_span_km (float): Physical width/height of the LR image, in
            kilometers, shared across all samples.
        show_coord (bool): Whether to show HR/LR coordinate grid panels.
        show_mask (bool): Whether to show the overlap mask panel.
        mask_type (str): "gaussian" or "binary", passed to `build_overlap_mask`.
        show_fourier (bool): Whether to show the Fourier positional-encoding
            channel grid instead of the simple panel row.
        fourier_bands (int): Number of Fourier frequency bands to compute
            and display, passed to `build_fourier_channels`.
    """
    plt.ion()

    for pos, (idx, rgb_path, ls_path) in enumerate(pairs, start=1):
        rgb_img = _rgb_to_display(_load_pt(rgb_path))
        ls_img = _landsat_to_display(_load_pt(ls_path))
        img_span_km = span_lookup.get(idx)

        base_panels = []
        base_panels.append(("RGB", rgb_img, None))
        base_panels.append(("Landsat", ls_img, None))

        if show_coord and img_span_km is not None:
            hr_grid, lr_grid = build_coord_grids(img_span_km, lr_span_km)
            base_panels.append(("HR coord X", hr_grid[0], "viridis"))
            base_panels.append(("HR coord Y", hr_grid[1], "viridis"))
            base_panels.append(("LR coord X", lr_grid[0], "viridis"))
            base_panels.append(("LR coord Y", lr_grid[1], "viridis"))

        if show_mask and img_span_km is not None:
            mask = build_overlap_mask(img_span_km, lr_span_km, mask_type)
            base_panels.append(("Overlap mask", mask, "gray"))

        has_fourier = show_fourier and img_span_km is not None

        if not has_fourier:
            ncols = len(base_panels)
            fig, axes = plt.subplots(1, ncols, figsize=(4 * ncols, 4))
            if ncols == 1:
                axes = [axes]
            for ax, (title, img, cmap) in zip(axes, base_panels):
                if isinstance(img, torch.Tensor):
                    img = img.detach().cpu()
                im = ax.imshow(img, cmap=cmap)
                ax.set_title(title, fontsize=9)
                ax.axis("off")
                if cmap is not None:
                    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        else:
            hr_grid, lr_grid = build_coord_grids(img_span_km, lr_span_km)
            hr_fourier = build_fourier_channels(hr_grid, fourier_bands)  # (4, L, H, W)
            lr_fourier = build_fourier_channels(lr_grid, fourier_bands)

            n_components = 4
            ncols = max(len(base_panels), fourier_bands)
            nrows = 1 + 2 * n_components  # base row + HR block + LR block
            fig = plt.figure(figsize=(3.5 * ncols, 3 * nrows))
            gs = GridSpec(nrows, ncols, figure=fig, hspace=0.35, wspace=0.3)

            for col, (title, img, cmap) in enumerate(base_panels):
                ax = fig.add_subplot(gs[0, col])
                if isinstance(img, torch.Tensor):
                    img = img.detach().cpu()
                im = ax.imshow(img, cmap=cmap)
                ax.set_title(title, fontsize=9)
                ax.axis("off")
                if cmap is not None:
                    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            for col in range(len(base_panels), ncols):
                fig.add_subplot(gs[0, col]).axis("off")

            freqs = math.pi * (2.0 ** torch.arange(fourier_bands, dtype=torch.float32))
            for branch_idx, (branch_name, fourier) in enumerate([
                ("HR", hr_fourier), ("LR", lr_fourier)
            ]):
                row_offset = 1 + branch_idx * n_components
                for comp_idx in range(n_components):
                    for band_idx in range(fourier_bands):
                        ax = fig.add_subplot(gs[row_offset + comp_idx, band_idx])
                        channel = fourier[comp_idx, band_idx].detach().cpu()
                        im = ax.imshow(channel, cmap="RdBu_r", vmin=-1, vmax=1)
                        ax.axis("off")
                        if comp_idx == 0:
                            ax.set_title(f"f={freqs[band_idx]:.1f}", fontsize=8)
                        if band_idx == 0:
                            ax.set_ylabel(
                                f"{branch_name} {FOURIER_COMPONENT_LABELS[comp_idx]}",
                                fontsize=8, rotation=0, labelpad=80, va="center",
                            )
                            ax.yaxis.set_visible(True)
                    for col in range(fourier_bands, ncols):
                        fig.add_subplot(gs[row_offset + comp_idx, col]).axis("off")

        span_str = f"HR span={img_span_km:.2f} km" if img_span_km is not None else "HR span=?"
        fig.suptitle(
            f"Index {idx}  ({pos}/{len(pairs)})  |  {span_str}  |  LR span={lr_span_km:.2f} km",
            fontsize=10,
        )
        fig.tight_layout()
        fig.canvas.draw_idle()
        plt.show(block=False)
        plt.pause(0.001)

        if pos < len(pairs):
            resp = input("Press Enter for the next sample (q to quit)... ")
            plt.close(fig)
            if resp.strip().lower() == "q":
                plt.close("all")
                return

    plt.ioff()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize spatial encoding tensors from preprocessed images")
    parser.add_argument("--preprocessed-dir", type=str, required=True,
                        help="Path to preprocessed cache (contains landsat/ and fmow_rgb/ subdirs)")
    parser.add_argument("--metadata-csv", type=str,
                        default="/home/datasets4/FMoW_LandSat/fmow_landsat/rgb_metadata_extended.csv")
    parser.add_argument("--coord-channels", action="store_true")
    parser.add_argument("--fourier-channels", action="store_true")
    parser.add_argument("--fourier-bands", type=int, default=4)
    parser.add_argument("--overlap-mask", action="store_true")
    parser.add_argument("--overlap-mask-type", choices=["binary", "gaussian"], default="binary")
    parser.add_argument("--lr-extension-factor", type=float, default=3.0)
    parser.add_argument("--max-images", type=int, default=10)
    args = parser.parse_args()

    base = Path(args.preprocessed_dir)
    landsat_dir = base / "landsat"
    rgb_dir = base / "fmow_rgb"

    landsat_by_idx = {_get_index(p): p for p in landsat_dir.glob("image_*.pt") if _get_index(p) is not None}
    rgb_by_idx = {_get_index(p): p for p in rgb_dir.glob("rgb_img_*.pt") if _get_index(p) is not None}
    common_idxs = sorted(set(landsat_by_idx) & set(rgb_by_idx))[:args.max_images]

    if not common_idxs:
        raise FileNotFoundError(f"No matching index pairs found in {landsat_dir} and {rgb_dir}")

    df = pd.read_csv(args.metadata_csv)
    df = df[df["split"] != "seq"].reset_index(drop=True)
    max_hr_span = df["img_span_km"].max()
    lr_span_km = max_hr_span * args.lr_extension_factor

    span_lookup = {}
    for idx in common_idxs:
        if idx < len(df):
            span_lookup[idx] = df.loc[idx, "img_span_km"]

    pairs = [(idx, rgb_by_idx[idx], landsat_by_idx[idx]) for idx in common_idxs]

    print(f"Found {len(pairs)} paired images")
    print(f"max(img_span_km) = {max_hr_span:.4f} km, lr_span_km = {lr_span_km:.4f} km")

    display(pairs, span_lookup, lr_span_km, args.coord_channels, args.overlap_mask, args.overlap_mask_type,
            args.fourier_channels, args.fourier_bands)
