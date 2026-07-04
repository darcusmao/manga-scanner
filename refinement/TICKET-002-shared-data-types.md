# TICKET-002: Shared Data Types and Interfaces

## Summary
Define all cross-module data contracts as Python dataclasses in a single `types.py`. This file is the lingua franca of the pipeline — every module imports from it. It must exist before any module ticket is implemented, or every module will define its own incompatible shape.

## Language and Tools
- Python 3.11 standard library `dataclasses` only — no third-party packages
- No Pydantic here; these are runtime transfer objects, not validation targets

## Implementation

File: `src/manga_scanner/types.py`

```python
from __future__ import annotations
from dataclasses import dataclass, field
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
    character_profiles: list  # List[CharacterProfile], typed loosely to avoid circular import
```

## Key Design Decisions

- `BoundingBox` uses integer pixel coordinates. YOLOv8 returns floats; convert at the detection boundary.
- `TYPE_CHECKING` guard on numpy/PIL imports prevents module-level import cost. The type annotations are strings (quoted), so they are not evaluated at runtime.
- `TranslationResult.raw_response` is intentionally kept for debugging malformed LLM output.
- `OCRResult.text` is allowed to be empty string. The orchestrator filters these before calling the translator.
- `PageJob.character_profiles` is typed as `list` (not `List[CharacterProfile]`) to avoid a circular import with the translation module. The orchestrator is responsible for passing the correct type.

## Acceptance Criteria
- All dataclasses importable: `from manga_scanner.types import BoundingBox, PageJob` succeeds
- `BoundingBox(x1=0, y1=0, x2=100, y2=50, confidence=0.9).width == 100`
- No third-party imports at module level

## Dependencies
- TICKET-001 (project skeleton must exist)

## Estimated Effort
1 hour
