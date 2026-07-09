from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from manga_scanner.config import DetectionConfig
from manga_scanner.types import BoundingBox, DetectionResult, MaskResult

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class BubbleMasker:
    """Converts detection boxes into a binary inpainting mask."""

    def __init__(self, config: DetectionConfig) -> None:
        self.padding = config.box_padding

    def build_mask(self, image: np.ndarray, result: DetectionResult) -> MaskResult:
        """
        Returns a MaskResult with a uint8 mask (0 or 255).
        Each detected box is expanded by self.padding pixels before being filled.
        """
        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

        for box in result.boxes:
            x1 = max(0, box.x1 - self.padding)
            y1 = max(0, box.y1 - self.padding)
            x2 = min(w, box.x2 + self.padding)
            y2 = min(h, box.y2 + self.padding)
            mask[y1:y2, x1:x2] = 255

        if not result.boxes:
            logger.debug("No boxes to mask for %s", result.image_path.name)

        return MaskResult(original=image, mask=mask, boxes=result.boxes)
