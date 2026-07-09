"""
display_preprocessed_images.py

Display preprocessed .pt images by reversing normalization.

Uses the same directory layout and argument structure as check_image_stats.py.
Landsat reflectance is clipped to [0, 0.3] for display (matching
translate_geotiff_to_png.py); RGB is clamped to [0, 1].

Functions:
    display_images: Non-interactive entry point; shows Landsat/RGB/both
        images from a preprocessed cache directory, either paired
        side-by-side (`modality="both"`) or one modality at a time.
    browse_by_class: Interactive entry point; lets the user pick a region and
        FMoW class from the metadata CSV, then displays the matching
        Landsat+RGB pairs via `_display_pairs`.
    _display_pairs: Shared paired-image display loop used by `browse_by_class`.
    _build_class_index / _prompt_region: Menu helpers for `browse_by_class`.
    _load_pt / _unnormalize / _landsat_to_display / _landsat_to_false_color /
        _rgb_to_display / _get_index: Helpers for loading `.pt` tensors,
        reversing their normalization for display, and parsing sample
        indices from filenames.

Usage:
    uv run python src/dataset_creation/display_preprocessed_images.py --preprocessed-dir /path/to/cache --modality both

Browse by class (interactive menu):
    uv run python src/dataset_creation/display_preprocessed_images.py --preprocessed-dir /path/to/cache --metadata-csv data/rgb_metadata_extended.csv
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch

LANDSAT_MEAN = torch.tensor([0.06259285658597946, 0.0880340114235878, 0.09441816806793213,
                             0.2327403724193573, 0.19073842465877533, 0.12976829707622528])
LANDSAT_STD = torch.tensor([0.039894334971904755, 0.049978554248809814, 0.0687960833311081,
                            0.092967689037323, 0.09390033036470413, 0.0819208025932312])

RGB_MEAN = torch.tensor([0.4155880808830261, 0.41815927624702454, 0.3903605341911316])
RGB_STD = torch.tensor([0.24812281131744385, 0.24405813217163086, 0.2482403963804245])

REGION_NAMES = {0: "Asia", 1: "Europe", 2: "Africa", 3: "Americas", 4: "Oceania", 5: "Other"}


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


def _unnormalize(tensor: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
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


def _landsat_to_display(tensor: torch.Tensor) -> torch.Tensor:
    """Convert a normalized Landsat tensor to a true-color image for display.

    Reverses LANDSAT_MEAN/LANDSAT_STD normalization, takes the first 3 bands
    (stored in BGR order) and reorders them to RGB, then divides by 0.3 and
    clamps to [0, 1] to bring typical reflectance values into displayable range.

    Args:
        tensor (torch.Tensor): Normalized Landsat tensor, shape (6, H, W), dtype float.

    Returns:
        torch.Tensor: Displayable RGB image, shape (H, W, 3), values in [0, 1].
    """
    raw = _unnormalize(tensor, LANDSAT_MEAN, LANDSAT_STD)
    bgr = raw[:3]
    rgb = bgr[[2, 1, 0]]
    rgb = (rgb / 0.3).clamp(0, 1)
    return rgb.permute(1, 2, 0)


def _landsat_to_false_color(tensor: torch.Tensor) -> torch.Tensor:
    """Convert a normalized Landsat tensor to a false-color composite for display.

    Reverses normalization, then builds R = max(bands 0-2), G = band 3
    (NIR), B = max(bands 4-5) (SWIR), and divides by 0.3, clamping to [0, 1].

    Args:
        tensor (torch.Tensor): Normalized Landsat tensor, shape (6, H, W), dtype float.

    Returns:
        torch.Tensor: Displayable false-color image, shape (H, W, 3), values in [0, 1].
    """
    raw = _unnormalize(tensor, LANDSAT_MEAN, LANDSAT_STD)
    r = raw[:3].max(dim=0).values
    g = raw[3]
    b = raw[4:6].max(dim=0).values
    rgb = torch.stack([r, g, b], dim=0)
    rgb = (rgb / 0.3).clamp(0, 1)
    return rgb.permute(1, 2, 0)


def _rgb_to_display(tensor: torch.Tensor) -> torch.Tensor:
    """Convert a normalized RGB tensor to a displayable image.

    Args:
        tensor (torch.Tensor): Normalized RGB tensor, shape (3, H, W), dtype float.

    Returns:
        torch.Tensor: Displayable RGB image, shape (H, W, 3), values in [0, 1].
    """
    raw = _unnormalize(tensor, RGB_MEAN, RGB_STD)
    return raw.clamp(0, 1).permute(1, 2, 0)


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


def _display_pairs(landsat_by_idx, rgb_by_idx, idxs, max_images: int, title_prefix: str = ""):
    """Interactively display paired Landsat/RGB/false-color images, one sample at a time.

    Renders a 2x2 figure per sample (Landsat true-color, RGB, and a wide
    false-color panel spanning the bottom row), pausing for Enter between
    samples.

    Args:
        landsat_by_idx (dict[int, Path]): Sample index -> Landsat `.pt` path.
        rgb_by_idx (dict[int, Path]): Sample index -> RGB `.pt` path.
        idxs (list[int]): Sample indices to display, in order.
        max_images (int): Maximum number of samples to display; `idxs` is
            truncated to this length.
        title_prefix (str): Optional label prepended to the figure's
            suptitle (e.g. region/class). Defaults to "".
    """
    idxs = idxs[:max_images]
    print(f"Displaying {len(idxs)} paired images (Landsat + RGB)")
    plt.ion()
    fig = plt.figure(figsize=(14, 12))
    gs = fig.add_gridspec(2, 2)
    ax_ls = fig.add_subplot(gs[0, 0])
    ax_rgb = fig.add_subplot(gs[0, 1])
    ax_fc = fig.add_subplot(gs[1, :])
    for i, idx in enumerate(idxs, start=1):
        ls_tensor = _load_pt(landsat_by_idx[idx])
        ls_img = _landsat_to_display(ls_tensor)
        fc_img = _landsat_to_false_color(ls_tensor)
        rgb_img = _rgb_to_display(_load_pt(rgb_by_idx[idx]))
        ax_ls.clear()
        ax_ls.imshow(ls_img)
        ax_ls.set_title(f"Landsat — image_{idx}.pt")
        ax_ls.axis("off")
        ax_rgb.clear()
        ax_rgb.imshow(rgb_img)
        ax_rgb.set_title(f"RGB — rgb_img_{idx}.pt")
        ax_rgb.axis("off")
        ax_fc.clear()
        ax_fc.imshow(fc_img)
        ax_fc.set_title(f"Landsat false color — R: max(R,G,B)  G: NIR  B: max(SWIR_1,SWIR_2)")
        ax_fc.axis("off")
        label = f"{title_prefix} — " if title_prefix else ""
        fig.suptitle(f"{label}{i}/{len(idxs)}", fontsize=12)
        fig.tight_layout()
        fig.canvas.draw_idle()
        plt.show(block=False)
        plt.pause(0.001)
        if i < len(idxs):
            input("Press Enter for the next pair...")
        else:
            input("Press Enter to return to class menu...")
    plt.ioff()
    plt.close(fig)


def _build_class_index(metadata_csv: str, paired_idxs: set[int], region: int | None = None) -> dict[str, list[int]]:
    """Group paired sample indices by FMoW class, optionally filtered by region.

    Args:
        metadata_csv (str): Path to `rgb_metadata_extended.csv`; row order
            (after dropping `split == "seq"` rows) matches the `.pt` file
            sample indices.
        paired_idxs (set[int]): Sample indices that have both a Landsat and
            an RGB `.pt` file.
        region (int | None): If given, only include samples whose metadata
            `region` column equals this code. Defaults to None (all regions).

    Returns:
        dict[str, list[int]]: Class name -> list of sample indices in that
            class (and region, if filtered).
    """
    df = pd.read_csv(metadata_csv)
    df = df[df["split"] != "seq"].reset_index(drop=True)
    class_to_idxs: dict[str, list[int]] = defaultdict(list)
    for idx in sorted(paired_idxs):
        if idx < len(df):
            if region is not None and df.iloc[idx]["region"] != region:
                continue
            class_to_idxs[df.iloc[idx]["category"]].append(idx)
    return dict(class_to_idxs)


def _prompt_region() -> int | None:
    """Prompt the user to pick a region from `REGION_NAMES`, or all regions.

    Returns:
        int | None: The chosen region code, or None if the user chose "all regions".
    """
    print(f"\n{'='*60}")
    print("Select region:")
    print("   -1  All regions")
    for idx, name in REGION_NAMES.items():
        print(f"  {idx:3d}  {name}")
    print(f"{'='*60}")
    while True:
        choice = input("Enter region number (blank or -1 for all): ").strip()
        if choice in ("", "-1"):
            return None
        try:
            r = int(choice)
        except ValueError:
            print("Please enter a number.")
            continue
        if r not in REGION_NAMES:
            print(f"Invalid number. Enter one of {sorted(REGION_NAMES)} or -1.")
            continue
        return r


def browse_by_class(preprocessed_dir: str, metadata_csv: str, max_images: int):
    """Interactive menu: pick a region and FMoW class, then browse its paired images.

    Scans `preprocessed_dir` for Landsat/RGB `.pt` pairs, then repeatedly
    prompts the user to select a class (optionally after changing the region
    filter) and displays that class's images via `_display_pairs`, until the
    user quits.

    Args:
        preprocessed_dir (str): Path to the preprocessed cache directory
            (containing `landsat/` and `fmow_rgb/` subdirectories).
        metadata_csv (str): Path to `rgb_metadata_extended.csv`, used to look
            up each sample's class and region.
        max_images (int): Maximum number of images to display per selected class.

    Raises:
        FileNotFoundError: If no matching Landsat/RGB index pairs are found
            in `preprocessed_dir`.
    """
    base = Path(preprocessed_dir)
    landsat_dir = base / "landsat"
    rgb_dir = base / "fmow_rgb"

    landsat_by_idx = {_get_index(p): p for p in landsat_dir.glob("image_*.pt") if _get_index(p) is not None}
    rgb_by_idx = {_get_index(p): p for p in rgb_dir.glob("rgb_img_*.pt") if _get_index(p) is not None}
    paired_idxs = set(landsat_by_idx) & set(rgb_by_idx)
    if not paired_idxs:
        raise FileNotFoundError(f"No matching index pairs found in {landsat_dir} and {rgb_dir}")

    region = _prompt_region()
    class_to_idxs = _build_class_index(metadata_csv, paired_idxs, region)
    classes = sorted(class_to_idxs.keys())

    while True:
        region_label = REGION_NAMES.get(region, "All regions") if region is not None else "All regions"
        total = sum(len(v) for v in class_to_idxs.values())
        print(f"\n{'='*60}")
        print(f"Region: {region_label} — {total} paired images across {len(classes)} classes")
        print(f"{'='*60}")
        for i, cls in enumerate(classes):
            print(f"  {i:3d}  {cls} ({len(class_to_idxs[cls])} images)")
        print(f"{'='*60}")
        choice = input("Enter class number, 'r' to change region, or 'q' to quit: ").strip()
        if choice.lower() == "q":
            break
        if choice.lower() == "r":
            region = _prompt_region()
            class_to_idxs = _build_class_index(metadata_csv, paired_idxs, region)
            classes = sorted(class_to_idxs.keys())
            continue
        try:
            cls_idx = int(choice)
            if not (0 <= cls_idx < len(classes)):
                print(f"Invalid number. Enter 0–{len(classes) - 1}.")
                continue
        except ValueError:
            print("Please enter a number, 'r', or 'q'.")
            continue

        cls_name = classes[cls_idx]
        idxs = sorted(class_to_idxs[cls_name])
        _display_pairs(landsat_by_idx, rgb_by_idx, idxs, max_images, title_prefix=f"{region_label} / {cls_name}")


def display_images(preprocessed_dir: str, modality: str, max_images: int):
    """Display preprocessed images non-interactively for a given modality.

    For `modality="both"`, shows paired Landsat/RGB/false-color panels (same
    layout as `_display_pairs`) for images present in both directories,
    pausing for Enter between samples. For `modality="landsat"` or `"rgb"`,
    shows a single-panel figure per image from just that directory.

    Args:
        preprocessed_dir (str): Path to the preprocessed cache directory
            (containing `landsat/` and `fmow_rgb/` subdirectories).
        modality (str): Which modality to display: "landsat", "rgb", or "both".
        max_images (int): Maximum number of images (or pairs) to display.

    Raises:
        FileNotFoundError: If `modality="both"` and no matching index pairs
            are found, or if `modality` is "landsat"/"rgb" and no matching
            files are found in the corresponding directory.
    """
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
        fig = plt.figure(figsize=(14, 12))
        gs = fig.add_gridspec(2, 2)
        ax_ls = fig.add_subplot(gs[0, 0])
        ax_rgb = fig.add_subplot(gs[0, 1])
        ax_fc = fig.add_subplot(gs[1, :])
        for i, idx in enumerate(common_idxs, start=1):
            ls_tensor = _load_pt(landsat_by_idx[idx])
            ls_img = _landsat_to_display(ls_tensor)
            fc_img = _landsat_to_false_color(ls_tensor)
            rgb_img = _rgb_to_display(_load_pt(rgb_by_idx[idx]))
            ax_ls.clear()
            ax_ls.imshow(ls_img)
            ax_ls.set_title(f"Landsat — image_{idx}.pt")
            ax_ls.axis("off")
            ax_rgb.clear()
            ax_rgb.imshow(rgb_img)
            ax_rgb.set_title(f"RGB — rgb_img_{idx}.pt")
            ax_rgb.axis("off")
            ax_fc.clear()
            ax_fc.imshow(fc_img)
            ax_fc.set_title(f"Landsat false color — R: max(R,G,B)  G: NIR  B: max(SWIR_1,SWIR_2)")
            ax_fc.axis("off")
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
    parser.add_argument("--metadata-csv", type=str, default=None,
                        help="Path to rgb_metadata_extended.csv. When provided, enables interactive class browsing.")
    args = parser.parse_args()

    if args.metadata_csv:
        browse_by_class(args.preprocessed_dir, args.metadata_csv, args.max_images)
    else:
        display_images(args.preprocessed_dir, args.modality, args.max_images)
