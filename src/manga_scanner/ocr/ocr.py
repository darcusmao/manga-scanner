from __future__ import annotations

import gc
import logging
from typing import TYPE_CHECKING

from manga_scanner.config import OCRConfig

if TYPE_CHECKING:
    from PIL import Image as PILImage

logger = logging.getLogger(__name__)


class MangaOCR:
    def __init__(self, config: OCRConfig) -> None:
        logger.info("Loading manga-ocr model...")
        from manga_ocr import MangaOcr
        self.model = MangaOcr()
        logger.info("manga-ocr model loaded.")

    def transcribe(self, crop: PILImage.Image) -> str:
        """Returns Japanese text found in the crop. Empty string if nothing detected."""
        try:
            result = self.model(crop)
            return result.strip()
        except Exception as e:
            logger.warning("OCR failed on crop: %s", e)
            return ""

    def transcribe_batch(self, crops: list[PILImage.Image]) -> list[str]:
        """Sequential transcription. manga-ocr does not expose batch inference."""
        return [self.transcribe(crop) for crop in crops]

    def unload(self) -> None:
        logger.info("Unloading manga-ocr model from VRAM.")
        del self.model
        import torch
        torch.cuda.empty_cache()
        gc.collect()
