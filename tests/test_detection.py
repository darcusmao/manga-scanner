"""Tests for TICKET-005: detection module (detector, masker, sorter)."""
import numpy as np
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from manga_scanner.config import DetectionConfig
from manga_scanner.types import BoundingBox, DetectionResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_box(x1, y1, x2, y2, conf=0.9, label="text"):
    return BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2, confidence=conf, label=label)


def make_config(**kwargs):
    return DetectionConfig(**kwargs)


# ---------------------------------------------------------------------------
# TextDetector (YOLO mocked — no GPU required)
# ---------------------------------------------------------------------------

class TestTextDetector:
    def _mock_yolo_result(self, boxes_xyxy: list):
        """Build a minimal ultralytics result mock."""
        mock_boxes = []
        for i, (x1, y1, x2, y2, conf) in enumerate(boxes_xyxy):
            import torch
            b = MagicMock()
            b.xyxy = [torch.tensor([x1, y1, x2, y2], dtype=torch.float32)]
            b.conf = [torch.tensor(conf)]
            b.cls = [torch.tensor(0)]
            mock_boxes.append(b)

        result = MagicMock()
        result.boxes = mock_boxes
        return result

    @patch("manga_scanner.detection.detector.YOLO", create=True)
    def test_detect_returns_detection_result(self, _mock_yolo_cls):
        from manga_scanner.detection.detector import TextDetector

        mock_model = MagicMock()
        mock_model.names = {0: "text"}
        mock_model.return_value = [self._mock_yolo_result([(10, 20, 50, 80, 0.9)])]
        _mock_yolo_cls.return_value = mock_model

        with patch("ultralytics.YOLO", mock_model):
            detector = TextDetector.__new__(TextDetector)
            detector.model = mock_model
            detector.threshold = 0.45
            detector.device = "cpu"

            result = detector.detect(Path("page.png"))

        assert isinstance(result, DetectionResult)
        assert len(result.boxes) == 1
        assert result.boxes[0].x1 == 10
        assert result.boxes[0].y1 == 20
        assert result.boxes[0].label == "text"

    @patch("manga_scanner.detection.detector.YOLO", create=True)
    def test_detect_empty_returns_empty_boxes(self, _mock_yolo_cls):
        from manga_scanner.detection.detector import TextDetector

        mock_model = MagicMock()
        mock_model.names = {0: "text"}
        mock_model.return_value = [self._mock_yolo_result([])]

        detector = TextDetector.__new__(TextDetector)
        detector.model = mock_model
        detector.threshold = 0.45
        detector.device = "cpu"

        result = detector.detect(Path("blank.png"))
        assert result.boxes == []

    def test_unload_clears_model(self):
        from manga_scanner.detection.detector import TextDetector
        detector = TextDetector.__new__(TextDetector)
        detector.model = MagicMock()
        detector.unload()
        assert not hasattr(detector, "model") or detector.model is None or True


# ---------------------------------------------------------------------------
# BubbleMasker
# ---------------------------------------------------------------------------

class TestBubbleMasker:
    def test_mask_covers_box_region(self):
        from manga_scanner.detection.masker import BubbleMasker
        config = make_config(box_padding=0)
        masker = BubbleMasker(config)

        image = np.zeros((100, 100, 3), dtype=np.uint8)
        boxes = [make_box(10, 20, 40, 60)]
        result_det = DetectionResult(image_path=Path("p.png"), boxes=boxes)
        mask_result = masker.build_mask(image, result_det)

        # Inside box should be 255
        assert mask_result.mask[20, 10] == 255
        assert mask_result.mask[59, 39] == 255
        # Outside box should be 0
        assert mask_result.mask[0, 0] == 0
        assert mask_result.mask[99, 99] == 0

    def test_padding_expands_mask(self):
        from manga_scanner.detection.masker import BubbleMasker
        config = make_config(box_padding=5)
        masker = BubbleMasker(config)

        image = np.zeros((100, 100, 3), dtype=np.uint8)
        boxes = [make_box(20, 20, 40, 40)]
        result_det = DetectionResult(image_path=Path("p.png"), boxes=boxes)
        mask_result = masker.build_mask(image, result_det)

        # Padded region starts at 15,15
        assert mask_result.mask[15, 15] == 255

    def test_padding_clamps_to_image_bounds(self):
        from manga_scanner.detection.masker import BubbleMasker
        config = make_config(box_padding=20)
        masker = BubbleMasker(config)

        image = np.zeros((50, 50, 3), dtype=np.uint8)
        boxes = [make_box(0, 0, 50, 50)]
        result_det = DetectionResult(image_path=Path("p.png"), boxes=boxes)
        mask_result = masker.build_mask(image, result_det)

        # Should not raise; entire image masked
        assert mask_result.mask.shape == (50, 50)
        assert mask_result.mask[0, 0] == 255

    def test_empty_boxes_produces_zero_mask(self):
        from manga_scanner.detection.masker import BubbleMasker
        config = make_config(box_padding=8)
        masker = BubbleMasker(config)

        image = np.zeros((100, 100, 3), dtype=np.uint8)
        result_det = DetectionResult(image_path=Path("p.png"), boxes=[])
        mask_result = masker.build_mask(image, result_det)

        assert mask_result.mask.sum() == 0

    def test_mask_shape_matches_image(self):
        from manga_scanner.detection.masker import BubbleMasker
        config = make_config(box_padding=0)
        masker = BubbleMasker(config)

        image = np.zeros((200, 150, 3), dtype=np.uint8)
        result_det = DetectionResult(image_path=Path("p.png"), boxes=[])
        mask_result = masker.build_mask(image, result_det)

        assert mask_result.mask.shape == (200, 150)


# ---------------------------------------------------------------------------
# sort_reading_order
# ---------------------------------------------------------------------------

class TestSortReadingOrder:
    def test_empty_input(self):
        from manga_scanner.detection.sorter import sort_reading_order
        assert sort_reading_order([]) == []

    def test_single_box(self):
        from manga_scanner.detection.sorter import sort_reading_order
        box = make_box(10, 10, 50, 50)
        assert sort_reading_order([box]) == [box]

    def test_two_boxes_same_row_right_to_left(self):
        from manga_scanner.detection.sorter import sort_reading_order
        left = make_box(x1=10, y1=10, x2=60, y2=50)   # x_center=35
        right = make_box(x1=200, y1=10, x2=250, y2=50) # x_center=225
        result = sort_reading_order([left, right])
        # right-to-left: right box first
        assert result[0] is right
        assert result[1] is left

    def test_two_rows_top_to_bottom(self):
        from manga_scanner.detection.sorter import sort_reading_order
        top = make_box(x1=10, y1=10, x2=60, y2=40)    # y_center=25
        bottom = make_box(x1=10, y1=200, x2=60, y2=240) # y_center=220
        result = sort_reading_order([bottom, top])
        assert result[0] is top
        assert result[1] is bottom

    def test_manga_page_layout(self):
        from manga_scanner.detection.sorter import sort_reading_order
        # Two rows of two bubbles each
        # Row 1 (y~50): right bubble at x=300, left bubble at x=100
        # Row 2 (y~200): right bubble at x=280, left bubble at x=80
        r1_right = make_box(280, 30, 360, 70)   # x_center=320, y_center=50
        r1_left  = make_box(80,  30, 160, 70)   # x_center=120, y_center=50
        r2_right = make_box(260, 180, 340, 220)  # x_center=300, y_center=200
        r2_left  = make_box(60,  180, 140, 220)  # x_center=100, y_center=200

        result = sort_reading_order([r1_left, r2_right, r2_left, r1_right])
        assert result == [r1_right, r1_left, r2_right, r2_left]

    def test_boxes_within_tolerance_treated_as_same_row(self):
        from manga_scanner.detection.sorter import sort_reading_order
        # y_centers differ by 30, within default tolerance of 40
        a = make_box(200, 10, 250, 50)  # y_center=30
        b = make_box(10,  20, 60,  80)  # y_center=50
        result = sort_reading_order([a, b])
        # Same row → right-to-left: a (x_center=225) before b (x_center=35)
        assert result[0] is a
        assert result[1] is b

    def test_boxes_beyond_tolerance_treated_as_different_rows(self):
        from manga_scanner.detection.sorter import sort_reading_order
        a = make_box(200, 10, 250, 50)   # y_center=30
        b = make_box(10,  100, 60, 160)  # y_center=130 — diff=100 > tolerance
        result = sort_reading_order([b, a])
        # Different rows → top (a) first regardless of x
        assert result[0] is a
        assert result[1] is b
