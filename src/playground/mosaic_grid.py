import random
from pathlib import Path

from PIL import Image

IMAGES_DIR = Path("data/fmow_landsat/images")
ROWS, COLS = 10, 20

png_files = sorted(IMAGES_DIR.glob("*.png"))
selected = random.choices(png_files, k=ROWS * COLS)

sample = Image.open(selected[0])
w, h = sample.size

grid = Image.new("RGB", (COLS * w, ROWS * h))
for idx, path in enumerate(selected):
    img = Image.open(path)
    row, col = divmod(idx, COLS)
    grid.paste(img, (col * w, row * h))

out_path = Path("data/fmow_landsat/mosaic_grid.png")
grid.save(out_path)
print(f"Saved {COLS * ROWS} images as {ROWS}x{COLS} grid to {out_path}")
print(f"Grid size: {grid.size[0]}x{grid.size[1]} px")
