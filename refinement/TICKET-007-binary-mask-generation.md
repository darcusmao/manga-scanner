# TICKET-007: Binary Mask Generation

## Summary
Write a function that takes a decoded image array and a list of `BoundingBox` objects and produces a binary mask image where detected text regions are filled with white (255) and the rest is black (0). The mask is consumed directly by the inpainting model.

## Language and Tools
- Python 3.11
- `numpy` (already installed as torch/ultralytics dependency — do not add separately)
- `Pillow` — for image open/convert if needed
- Install Pillow: `uv add Pillow`

## What LaMa Expects
The iopaint/LaMa model expects:
- `image`: HxWx3 uint8 numpy array (RGB)
- `mask`: HxW uint8 numpy array where 255 = inpaint this region, 0 = preserve

The mask must be the same spatial dimensions as the image.

## Implementation

File: `src/manga_scanner/detection/masker.py`

```python
import numpy as np
from pathlib import Path
from PIL import Image
from manga_scanner.types import BoundingBox, MaskResult


def load_image(image_path: Path) -> np.ndarray:
    """Load image to HxWx3 uint8 RGB numpy array."""
    img = Image.open(image_path).convert("RGB")
    return np.array(img)


def generate_mask(
    image: np.ndarray,
    boxes: list[BoundingBox],
    padding: int = 8,
) -> MaskResult:
    """
    Create a binary inpainting mask from bounding boxes.

    padding: pixels to expand each bbox in all four directions.
             Expand slightly beyond the detected boundary so that
             text edge pixels (which often bleed into the background)
             are fully covered.
    """
    h, w = image.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    for box in boxes:
        x1 = max(0, box.x1 - padding)
        y1 = max(0, box.y1 - padding)
        x2 = min(w, box.x2 + padding)
        y2 = min(h, box.y2 + padding)
        mask[y1:y2, x1:x2] = 255

    return MaskResult(original=image, mask=mask, boxes=boxes)
```

## Padding Rationale
Without padding, text pixels that sit at the exact boundary of the detected box are not masked. Inpainting these pixels produces a faint ring artifact around where the text was. Expanding by 8px (configurable via `DetectionConfig.box_padding`) eliminates this. Values above 15px start to remove background content unnecessarily.

## Acceptance Criteria
- `generate_mask(image, boxes).mask.shape == image.shape[:2]`
- Mask dtype is uint8
- Pixels inside each (padded) bbox are 255; all others are 0
- Boxes that extend near the image edge are clamped, not exceeding image bounds
- `load_image` returns a 3-channel uint8 array regardless of whether source is grayscale or RGBA

## Dependencies
- TICKET-001 (project skeleton)
- TICKET-002 (BoundingBox, MaskResult types)
- TICKET-004 (Pillow installed as a side effect of torch/ultralytics; explicit `uv add Pillow` here to be explicit)

## Estimated Effort
1.5 hours
