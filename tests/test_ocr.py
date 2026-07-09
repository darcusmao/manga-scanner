"""Tests for TICKET-011 (crop extraction) and TICKET-010 (OCR wrapper)."""
import numpy as np
import pytest
from pathlib import Path
from PIL import Image
from unittest.mock import MagicMock, patch

from manga_scanner.types import BoundingBox
from manga_scanner.ocr.cropper import extract_crops


def make_image(width=400, height=600) -> np.ndarray:
    return np.zeros((height, width, 3), dtype=np.uint8)


def make_box(x1, y1, x2, y2) -> BoundingBox:
    return BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2, confidence=0.9)


class TestExtractCrops:
    def test_basic_crop_dimensions(self):
        image = make_image()
        boxes = [make_box(10, 20, 110, 80)]
        crops = extract_crops(image, boxes)
        assert len(crops) == 1
        assert crops[0].crop.size == (100, 60)

    def test_crop_index_matches_reading_order(self):
        image = make_image()
        boxes = [make_box(0, 0, 50, 50), make_box(200, 0, 250, 50)]
        crops = extract_crops(image, boxes)
        assert crops[0].index == 0
        assert crops[1].index == 1

    def test_box_clamped_to_image_bounds(self):
        image = make_image(width=100, height=100)
        boxes = [make_box(-5, -5, 110, 110)]
        crops = extract_crops(image, boxes)
        assert len(crops) == 1
        assert crops[0].crop.size == (100, 100)

    def test_degenerate_box_skipped(self):
        image = make_image()
        boxes = [make_box(50, 50, 50, 50)]
        crops = extract_crops(image, boxes)
        assert len(crops) == 0

    def test_empty_box_list(self):
        image = make_image()
        crops = extract_crops(image, [])
        assert crops == []

    def test_multiple_boxes_all_returned(self):
        image = make_image()
        boxes = [make_box(0, 0, 50, 50), make_box(100, 100, 200, 200), make_box(300, 300, 380, 380)]
        crops = extract_crops(image, boxes)
        assert len(crops) == 3

    def test_crop_box_reference_preserved(self):
        image = make_image()
        box = make_box(10, 10, 60, 60)
        crops = extract_crops(image, [box])
        assert crops[0].box is box


def _make_ocr(transcribe_return="テスト", side_effect=None):
    """Build a MangaOCR instance with manga_ocr module fully mocked via sys.modules."""
    from manga_scanner.config import OCRConfig

    mock_instance = MagicMock()
    if side_effect is not None:
        mock_instance.side_effect = side_effect
    else:
        mock_instance.return_value = transcribe_return

    mock_module = MagicMock()
    mock_module.MangaOcr.return_value = mock_instance

    with patch.dict("sys.modules", {"manga_ocr": mock_module}):
        from manga_scanner.ocr.ocr import MangaOCR
        ocr = MangaOCR(OCRConfig())

    ocr.model = mock_instance
    return ocr


class TestMangaOCRWrapper:
    def test_transcribe_returns_string(self):
        ocr = _make_ocr("テスト")
        result = ocr.transcribe(Image.new("RGB", (100, 50)))
        assert isinstance(result, str)
        assert result == "テスト"

    def test_transcribe_strips_whitespace(self):
        ocr = _make_ocr("  hello  ")
        result = ocr.transcribe(Image.new("RGB", (100, 50)))
        assert result == "hello"

    def test_transcribe_handles_exception(self):
        ocr = _make_ocr(side_effect=RuntimeError("model error"))
        result = ocr.transcribe(Image.new("RGB", (10, 10)))
        assert result == ""

    def test_transcribe_batch_returns_list(self):
        ocr = _make_ocr("text")
        crops = [Image.new("RGB", (50, 50)) for _ in range(3)]
        results = ocr.transcribe_batch(crops)
        assert len(results) == 3
        assert all(isinstance(r, str) for r in results)
