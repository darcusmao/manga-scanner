"""Run once to generate synthetic fixture images for the integration test suite."""
from pathlib import Path
from PIL import Image, ImageDraw

OUT_DIR = Path(__file__).parent / "pages"
OUT_DIR.mkdir(exist_ok=True)

for i in range(3):
    img = Image.new("RGB", (800, 1200), color=(240, 240, 240))
    draw = ImageDraw.Draw(img)
    draw.rectangle([100, 100, 400, 200], fill=(0, 0, 0))
    draw.rectangle([400, 400, 700, 500], fill=(0, 0, 0))
    img.save(OUT_DIR / f"page_{i + 1:03d}.png")

print(f"Generated 3 fixture images in {OUT_DIR}")
