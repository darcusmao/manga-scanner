from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    from PIL import Image


@dataclass
class BoundingBox:
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    label: str = "text"

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    @property
    def x_center(self) -> int:
        return self.x1 + self.width // 2

    @property
    def y_center(self) -> int:
        return self.y1 + self.height // 2


@dataclass
class DetectionResult:
    image_path: Path
    boxes: list[BoundingBox]


@dataclass
class MaskResult:
    original: "np.ndarray"    # HxWx3 uint8
    mask: "np.ndarray"        # HxW uint8, 0 or 255
    boxes: list[BoundingBox]  # same boxes used to generate mask


@dataclass
class CropResult:
    box: BoundingBox
    crop: "Image.Image"
    index: int                # position in reading order


@dataclass
class OCRResult:
    box: BoundingBox
    text: str                 # raw Japanese string; empty string if OCR returned nothing
    index: int                # matches CropResult.index


@dataclass
class TranslationResult:
    translations: list[str]   # aligned 1:1 with non-empty OCRResult list
    raw_response: str         # full LLM response string for debugging


@dataclass
class RenderResult:
    image: "Image.Image"
    output_path: Path


@dataclass
class PageJob:
    input_path: Path
    output_path: Path
    page_number: int
    character_profiles: list  # List[CharacterProfile]; loose type to avoid circular import
