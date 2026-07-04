# TICKET-024: Unit Tests — OCR Module and Typesetting

## Summary
Write unit tests for the OCR crop extraction, reading order sorter, text fitting algorithm, and overlay renderer. These tests use mocking and synthetic fixtures to run without a GPU or any downloaded model weights.

## Language and Tools
- Python 3.11
- `pytest` — test runner
- `pytest-mock` — mock/patch utilities
- `Pillow` — for generating synthetic test images
- Install: `uv add --dev pytest pytest-mock`

## Test Coverage

### `tests/test_ocr.py`

```python
import numpy as np
import pytest
from pathlib import Path
from PIL import Image
from unittest.mock import MagicMock, patch
from manga_scanner.types import BoundingBox
from manga_scanner.ocr.cropper import extract_crops


def make_test_image(width=400, height=600) -> np.ndarray:
    return np.zeros((height, width, 3), dtype=np.uint8)


def make_box(x1, y1, x2, y2) -> BoundingBox:
    return BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2, confidence=0.9)


class TestExtractCrops:
    def test_basic_crop_dimensions(self):
        image = make_test_image()
        boxes = [make_box(10, 20, 110, 80)]
        crops = extract_crops(image, boxes)
        assert len(crops) == 1
        assert crops[0].crop.size == (100, 60)  # (x2-x1, y2-y1)

    def test_crop_index_matches_reading_order(self):
        image = make_test_image()
        boxes = [make_box(0, 0, 50, 50), make_box(200, 0, 250, 50)]
        crops = extract_crops(image, boxes)
        assert crops[0].index == 0
        assert crops[1].index == 1

    def test_box_clamped_to_image_bounds(self):
        image = make_test_image(width=100, height=100)
        boxes = [make_box(-5, -5, 110, 110)]  # extends beyond image
        crops = extract_crops(image, boxes)
        assert len(crops) == 1
        assert crops[0].crop.size == (100, 100)

    def test_degenerate_box_skipped(self):
        image = make_test_image()
        boxes = [make_box(50, 50, 50, 50)]  # zero area
        crops = extract_crops(image, boxes)
        assert len(crops) == 0

    def test_empty_box_list(self):
        image = make_test_image()
        crops = extract_crops(image, [])
        assert crops == []


class TestMangaOCRWrapper:
    def test_transcribe_returns_string(self):
        with patch("manga_scanner.ocr.ocr.MangaOcr") as MockMangaOcr:
            mock_instance = MagicMock()
            mock_instance.return_value = "テスト"
            MockMangaOcr.return_value = mock_instance

            from manga_scanner.config import OCRConfig
            from manga_scanner.ocr.ocr import MangaOCR
            ocr = MangaOCR(OCRConfig())
            result = ocr.transcribe(Image.new("RGB", (100, 50)))
            assert isinstance(result, str)

    def test_transcribe_handles_exception(self):
        with patch("manga_scanner.ocr.ocr.MangaOcr") as MockMangaOcr:
            mock_instance = MagicMock()
            mock_instance.side_effect = RuntimeError("model error")
            MockMangaOcr.return_value = mock_instance

            from manga_scanner.config import OCRConfig
            from manga_scanner.ocr.ocr import MangaOCR
            ocr = MangaOCR(OCRConfig())
            result = ocr.transcribe(Image.new("RGB", (10, 10)))
            assert result == ""
```

### `tests/test_typesetting.py`

```python
import pytest
from pathlib import Path
from PIL import Image
from manga_scanner.typesetting.fitter import fit_text, FitResult
from manga_scanner.detection.sorter import sort_reading_order
from manga_scanner.types import BoundingBox

# Use a system font for tests to avoid requiring the project font to be present
import sys
if sys.platform == "darwin":
    TEST_FONT = "/System/Library/Fonts/Helvetica.ttc"
else:
    TEST_FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


class TestFitText:
    def test_short_text_uses_max_font_size(self):
        result = fit_text("Hi", 300, 100, TEST_FONT, max_font_size=24, min_font_size=8)
        assert result.font_size == 24
        assert result.fits is True

    def test_long_text_reduces_font_size(self):
        long_text = "This is a very long sentence that will not fit at a large font size."
        result = fit_text(long_text, 100, 60, TEST_FONT, max_font_size=24, min_font_size=8)
        assert result.font_size < 24

    def test_impossible_text_returns_min_size(self):
        very_long = "W " * 200
        result = fit_text(very_long, 50, 30, TEST_FONT, max_font_size=24, min_font_size=8)
        assert result.font_size == 8
        assert result.fits is False

    def test_empty_text_does_not_raise(self):
        result = fit_text("", 200, 100, TEST_FONT)
        assert isinstance(result, FitResult)

    def test_lines_reconstruct_original_words(self):
        text = "Hello world how are you"
        result = fit_text(text, 300, 200, TEST_FONT)
        reconstructed = " ".join(" ".join(result.lines).split())
        assert reconstructed == text


class TestReadingOrderSorter:
    def make_box(self, x1, y1, x2, y2) -> BoundingBox:
        return BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2, confidence=0.9)

    def test_two_bubbles_same_row_right_to_left(self):
        left = self.make_box(50, 100, 150, 150)
        right = self.make_box(300, 105, 400, 155)
        result = sort_reading_order([left, right], row_threshold=50)
        assert result[0].x_center > result[1].x_center

    def test_two_rows_top_to_bottom(self):
        top = self.make_box(100, 50, 200, 100)
        bottom = self.make_box(100, 400, 200, 450)
        result = sort_reading_order([bottom, top], row_threshold=50)
        assert result[0].y_center < result[1].y_center

    def test_empty_input(self):
        assert sort_reading_order([]) == []

    def test_single_box_returned_unchanged(self):
        box = self.make_box(10, 10, 100, 100)
        assert sort_reading_order([box]) == [box]

    def test_known_three_box_layout(self):
        # Two boxes in top row (right-to-left), one in bottom
        top_right = self.make_box(300, 50, 400, 100)
        top_left = self.make_box(50, 55, 150, 105)
        bottom = self.make_box(200, 400, 300, 450)
        result = sort_reading_order([top_left, bottom, top_right], row_threshold=80)
        assert result[0] == top_right
        assert result[1] == top_left
        assert result[2] == bottom
```

## Running Tests

```bash
uv run pytest tests/ -v
```

## Acceptance Criteria
- All tests pass without a GPU, Ollama, or any downloaded model weights
- `TestMangaOCRWrapper` tests mock the `manga_ocr.MangaOcr` import so no actual model is loaded
- `TestFitText` uses a system font (not the project font) so tests are portable
- `pytest tests/ -v` reports 0 failures

## Dependencies
- TICKET-001 (project skeleton, tests/ directory)
- TICKET-006 (sorter.py)
- TICKET-010 (ocr.py)
- TICKET-011 (cropper.py)
- TICKET-017 (fitter.py)

## Estimated Effort
3 hours
