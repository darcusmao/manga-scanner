from __future__ import annotations

import logging

import numpy as np
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
    crops: list[CropResult] = []

    for idx, box in enumerate(boxes):
        x1 = max(0, box.x1)
        y1 = max(0, box.y1)
        x2 = min(w, box.x2)
        y2 = min(h, box.y2)

        if x2 <= x1 or y2 <= y1:
            logger.warning(
                "Skipping degenerate bounding box at index %d: (%d,%d,%d,%d)",
                idx, box.x1, box.y1, box.x2, box.y2,
            )
            continue

        crop = pil_image.crop((x1, y1, x2, y2))
        crops.append(CropResult(box=box, crop=crop, index=idx))

    return crops
