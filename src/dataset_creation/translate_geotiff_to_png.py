"""
translate_geotiff_to_png.py

Converts every multi-band Landsat GeoTIFF in `IMAGES_DIR` (bands read in the order
blue, green, red as bands 1, 2, 3, per the code below) into an 8-bit RGB PNG for visual
inspection, normalizing each band from a reflectance range into [0, 255] via
`normalize_band`. Skips files that already have a corresponding PNG.
"""
import pathlib
import numpy as np
import rasterio
from PIL import Image

PROJECT_ROOT = pathlib.Path(__file__).parent.parent.parent.resolve()
DATA_DIR = PROJECT_ROOT / "data" / "fmow_landsat"
IMAGES_DIR = DATA_DIR / "images"

def normalize_band(band_data, min_val=0, max_val=0.3):
    """Linearly rescale reflectance values to an 8-bit [0, 255] range.

    Args:
        band_data (np.ndarray): Single-band reflectance array of shape (height, width),
            dtype float.
        min_val (float): Reflectance value mapped to 0. Defaults to 0.
        max_val (float): Reflectance value mapped to 255. Defaults to 0.3.

    Returns:
        np.ndarray: Array of the same shape as `band_data`, dtype uint8, clipped to
            [0, 255].
    """
    normalized = np.clip((band_data - min_val) / (max_val - min_val) * 255, 0, 255)
    return normalized.astype(np.uint8)

if __name__ == "__main__":
    for tif_file in IMAGES_DIR.glob("*.tif"):
        png_out = tif_file.with_suffix(".png")
        if not png_out.exists():
            try:
                with rasterio.open(tif_file) as src:
                    # Read RGB bands (assuming bands 3, 2, 1 for R, G, B)
                    red = src.read(3).astype(float)
                    green = src.read(2).astype(float)
                    blue = src.read(1).astype(float)
                    
                    # Normalize bands
                    red_norm = normalize_band(red)
                    green_norm = normalize_band(green)
                    blue_norm = normalize_band(blue)
                    
                    # Stack into RGB image
                    rgb_array = np.dstack((red_norm, green_norm, blue_norm))
                    
                    # Create and save PNG
                    img = Image.fromarray(rgb_array)
                    img.save(png_out)
                    print(f"Converted: {png_out}")
            except Exception as e:
                print(f"Error converting {tif_file}: {e}")