"""Overlay HR (FMoW) patch on spatially correct upscaled LR (Landsat) image."""

import argparse
from pathlib import Path

import pandas as pd
from PIL import Image


def create_overlay(
    hr_path: Path,
    lr_path: Path,
    img_span_km: float,
    hr_fraction: float = 0.1,
    output_path: Path | None = None,
    nest_into: Path | None = None,
    nested_output_path: Path | None = None,
):
    hr = Image.open(hr_path)
    lr = Image.open(lr_path)

    hr_m_per_px = (img_span_km * 1000) / hr.width
    lr_m_per_px = 30.0
    scale_factor = lr_m_per_px / hr_m_per_px

    lr_upscaled_size = int(lr.width * scale_factor)
    lr_upscaled = lr.resize((lr_upscaled_size, lr_upscaled_size), Image.NEAREST)

    canvas_size = int(hr.width / hr_fraction)
    center = lr_upscaled_size // 2
    crop_box = (
        center - canvas_size // 2,
        center - canvas_size // 2,
        center + canvas_size // 2,
        center + canvas_size // 2,
    )
    canvas = lr_upscaled.crop(crop_box)

    hr_offset = (canvas_size - hr.width) // 2
    canvas.paste(hr, (hr_offset, hr_offset))

    if output_path is None:
        output_path = hr_path.parent / f"overlay_{hr_path.stem}.png"
    canvas.save(output_path)

    extent_km = canvas_size * hr_m_per_px / 1000
    print(f"HR: {hr.width}x{hr.height}px = {img_span_km*1000:.0f}m, {hr_m_per_px:.2f} m/px")
    print(f"LR upscaled: {lr_upscaled_size}px (scale factor {scale_factor:.1f}x)")
    print(f"Canvas: {canvas_size}x{canvas_size}px, showing {extent_km:.1f} km extent")
    print(f"HR fraction: {hr.width/canvas_size:.1%}")
    print(f"Saved to {output_path}")

    if nest_into is not None:
        lr_full = Image.open(nest_into)
        overlay_lr_px = int(canvas_size * hr_m_per_px / lr_m_per_px)
        overlay_small = canvas.resize((overlay_lr_px, overlay_lr_px), Image.LANCZOS)

        offset = (lr_full.width - overlay_lr_px) // 2
        lr_full.paste(overlay_small, (offset, offset))

        if nested_output_path is None:
            nested_output_path = hr_path.parent / f"nested_{hr_path.stem}.png"
        lr_full.save(nested_output_path)

        nested_extent_km = lr_full.width * lr_m_per_px / 1000
        print(f"\nNested: overlay ({overlay_lr_px}px) in full LR ({lr_full.width}px)")
        print(f"Full LR extent: {nested_extent_km:.1f} km")
        print(f"Saved to {nested_output_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("index", type=int, help="Row index in metadata (after removing seq split)")
    parser.add_argument("--hr", type=Path, required=True, help="Path to HR (FMoW) PNG")
    parser.add_argument("--lr", type=Path, required=True, help="Path to LR (Landsat) PNG")
    parser.add_argument("--fraction", type=float, default=0.1, help="HR fraction of canvas width")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--nest-into", type=Path, default=None, help="Nest overlay into this full LR image for 3-level zoom")
    parser.add_argument("--nested-output", type=Path, default=None)
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("data/fmow_landsat/rgb_metadata_extended.csv"),
    )
    args = parser.parse_args()

    df = pd.read_csv(args.metadata)
    df = df[df["split"] != "seq"].reset_index(drop=True)
    row = df.iloc[args.index]
    print(f"Sample: {row['img_filename']}, class: {row['category']}")

    create_overlay(
        hr_path=args.hr,
        lr_path=args.lr,
        img_span_km=row["img_span_km"],
        hr_fraction=args.fraction,
        output_path=args.output,
        nest_into=args.nest_into,
        nested_output_path=args.nested_output,
    )


if __name__ == "__main__":
    main()
