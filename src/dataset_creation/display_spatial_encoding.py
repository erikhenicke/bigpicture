"""Display preprocessed FMoW RGB + Landsat images alongside spatial encoding tensors.

Loads .pt files directly from the preprocessed cache (no WILDS dependency) and
computes spatial tensors from the metadata CSV. Shows a multi-panel figure per
sample: RGB, Landsat, coordinate grids (HR/LR), and overlap mask.

Usage:
    uv run python src/dataset_creation/display_spatial_encoding.py \
        --preprocessed-dir /path/to/fmow_preprocessed \
        --metadata-csv /home/datasets4/FMoW_LandSat/fmow_landsat/rgb_metadata_extended.csv \
        --coord-channels --overlap-mask

    uv run python src/dataset_creation/display_spatial_encoding.py \
        --preprocessed-dir /path/to/fmow_preprocessed \
        --metadata-csv /home/datasets4/FMoW_LandSat/fmow_landsat/rgb_metadata_extended.csv \
        --overlap-mask --overlap-mask-type gaussian
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
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
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _unnorm(tensor, mean, std):
    c = tensor.shape[0]
    return tensor * std[:c].view(c, 1, 1) + mean[:c].view(c, 1, 1)


def _rgb_to_display(tensor):
    return _unnorm(tensor, RGB_MEAN, RGB_STD).clamp(0, 1).permute(1, 2, 0)


def _landsat_to_display(tensor):
    raw = _unnorm(tensor, LANDSAT_MEAN, LANDSAT_STD)
    rgb = raw[:3][[2, 1, 0]]
    return (rgb / 0.3).clamp(0, 1).permute(1, 2, 0)


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


def build_overlap_mask(img_span_km: float, lr_span_km: float, mask_type: str):
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


def display(pairs, span_lookup, lr_span_km, show_coord, show_mask, mask_type):
    plt.ion()

    for pos, (idx, rgb_path, ls_path) in enumerate(pairs, start=1):
        rgb_img = _rgb_to_display(_load_pt(rgb_path))
        ls_img = _landsat_to_display(_load_pt(ls_path))
        img_span_km = span_lookup.get(idx)

        panels = []
        panels.append(("RGB", rgb_img, None))
        panels.append(("Landsat", ls_img, None))

        if show_coord and img_span_km is not None:
            hr_grid, lr_grid = build_coord_grids(img_span_km, lr_span_km)
            panels.append(("HR coord X", hr_grid[0], "viridis"))
            panels.append(("HR coord Y", hr_grid[1], "viridis"))
            panels.append(("LR coord X", lr_grid[0], "viridis"))
            panels.append(("LR coord Y", lr_grid[1], "viridis"))

        if show_mask and img_span_km is not None:
            mask = build_overlap_mask(img_span_km, lr_span_km, mask_type)
            panels.append(("Overlap mask", mask, "gray"))

        ncols = len(panels)
        fig, axes = plt.subplots(1, ncols, figsize=(4 * ncols, 4))
        if ncols == 1:
            axes = [axes]

        for ax, (title, img, cmap) in zip(axes, panels):
            if isinstance(img, torch.Tensor):
                img = img.detach().cpu()
            im = ax.imshow(img, cmap=cmap)
            ax.set_title(title, fontsize=9)
            ax.axis("off")
            if cmap is not None:
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

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
            input("Press Enter for the next sample...")
            plt.close(fig)

    plt.ioff()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize spatial encoding tensors from preprocessed images")
    parser.add_argument("--preprocessed-dir", type=str, required=True,
                        help="Path to preprocessed cache (contains landsat/ and fmow_rgb/ subdirs)")
    parser.add_argument("--metadata-csv", type=str,
                        default="/home/datasets4/FMoW_LandSat/fmow_landsat/rgb_metadata_extended.csv")
    parser.add_argument("--coord-channels", action="store_true")
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

    display(pairs, span_lookup, lr_span_km, args.coord_channels, args.overlap_mask, args.overlap_mask_type)
