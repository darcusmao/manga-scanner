# TICKET-011: Crop Extraction Utility

## Summary
Write a function that opens an image file and slices out a `PIL.Image` crop for each `BoundingBox`. Each crop is tagged with its source bounding box and reading-order index so the orchestrator can reassemble translations back to the correct positions.

## Language and Tools
- Python 3.11
- `Pillow` (already installed)
- `numpy` (already installed)

## Implementation

File: `src/manga_scanner/ocr/cropper.py`

```python
import logging
import numpy as np
from pathlib import Path
from PIL import Image
from manga_scanner.types import BoundingBox, CropResult

logger = logging.getLogger(__name__)


def extract_crops(
    image: np.ndarray,
    boxes: list[BoundingBox],
) -> list[CropResult]:
    """
    Slice a region from the image for each bounding box.
    boxes should already be in reading order (output of sort_reading_order).
    Each CropResult.index reflects position in the reading order sequence.
    """
    pil_image = Image.fromarray(image)
    h, w = image.shape[:2]
    crops = []

    for idx, box in enumerate(boxes):
        # Clamp to image bounds to handle detection boxes that slightly overflow
        x1 = max(0, box.x1)
        y1 = max(0, box.y1)
        x2 = min(w, box.x2)
        y2 = min(h, box.y2)

        if x2 <= x1 or y2 <= y1:
            logger.warning(
                "Skipping degenerate bounding box at index %d: (%d,%d,%d,%d)",
                idx, box.x1, box.y1, box.x2, box.y2
            )
            continue

        crop = pil_image.crop((x1, y1, x2, y2))
        crops.append(CropResult(box=box, crop=crop, index=idx))

    return crops
```

## Design Notes

- Takes a `np.ndarray` (already loaded by the masker in TICKET-007) rather than a file path — avoids redundant disk reads.
- Clamping is critical: YOLOv8 bounding boxes can slightly exceed image dimensions due to floating point conversion. Without clamping, `PIL.Image.crop()` silently returns an incorrect region or raises.
- Degenerate boxes (zero or negative area after clamping) are skipped with a warning. The orchestrator's translation list alignment is based on `CropResult.index` values, so gaps are allowed.
- Reading order is assigned by the `idx` enumeration — this function assumes `boxes` has already been sorted by TICKET-006.

## Acceptance Criteria
- `extract_crops(image, boxes)` returns one `CropResult` per valid bounding box
- `crop.size == (box.x2 - box.x1, box.y2 - box.y1)` for non-clamped boxes
- Boxes at the image edge do not raise; they are clamped and cropped correctly
- A box with x1==x2 (degenerate) is skipped with a log warning and no exception

## Dependencies
- TICKET-001 (project skeleton)
- TICKET-002 (BoundingBox, CropResult types)

## Estimated Effort
1.5 hours
