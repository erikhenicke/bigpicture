import pathlib
import subprocess

PROJECT_ROOT = pathlib.Path(__file__).parent.parent.parent.resolve()
DATA_DIR = PROJECT_ROOT / "data" / "fmow_landsat"
IMAGES_DIR = DATA_DIR / "images"

if __name__ == "__main__":
    for tif_file in IMAGES_DIR.glob("*.tif"):
        png_out = tif_file.with_suffix(".png")
        if not png_out.exists():
            subprocess.run([
                "gdal_translate", "-b", "3", "-b", "2", "-b", "1", "-scale", "0", "0.3", "0", "255",
                "-co", "TFW = NO",
                "-co", "STATS = NO",
                str(tif_file), str(png_out)
            ])
