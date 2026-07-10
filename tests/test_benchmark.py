"""Tests for benchmark.py pure functions (CER, IoU, detection metrics)."""
import json
import sys
import pytest
from pathlib import Path

# Import directly from scripts/ without installing
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from benchmark import (
    cer,
    corpus_cer,
    iou,
    detection_metrics,
    DetectionCounts,
    compute_ocr_metrics,
    compute_detection_metrics_all,
    write_results_md,
)


class TestCER:
    def test_identical_strings_zero(self):
        assert cer("テスト", "テスト") == 0.0

    def test_empty_hypothesis_full_error(self):
        assert cer("", "abc") == 1.0

    def test_empty_reference_zero_when_hypothesis_also_empty(self):
        assert cer("", "") == 0.0

    def test_empty_reference_nonempty_hypothesis(self):
        assert cer("abc", "") == 1.0

    def test_single_substitution(self):
        # "a" -> "b": 1 edit, 1 reference char
        assert cer("b", "a") == 1.0

    def test_partial_match(self):
        # "abc" vs "axc": 1 edit / 3 chars = 0.333...
        assert abs(cer("axc", "abc") - 1 / 3) < 1e-9

    def test_corpus_cer_weighted(self):
        # pair 1: "ab" vs "ab" — 0 edits; pair 2: "x" vs "abc" — 3 edits / 3 chars
        # total: 3 edits / 5 chars = 0.6
        result = corpus_cer(["ab", "x"], ["ab", "abc"])
        assert abs(result - 3 / 5) < 1e-9

    def test_corpus_cer_empty_lists(self):
        assert corpus_cer([], []) == 0.0


class TestIoU:
    def test_identical_boxes(self):
        assert iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0

    def test_no_overlap(self):
        assert iou([0, 0, 5, 5], [10, 10, 20, 20]) == 0.0

    def test_partial_overlap(self):
        # A=[0,0,4,4] area=16; B=[2,2,6,6] area=16; inter=[2,2,4,4]=4; union=28
        score = iou([0, 0, 4, 4], [2, 2, 6, 6])
        assert abs(score - 4 / 28) < 1e-9

    def test_b_inside_a(self):
        # B is entirely inside A; inter = area_B
        # A=[0,0,10,10] area=100; B=[2,2,4,4] area=4; union=100; iou=4/100
        score = iou([0, 0, 10, 10], [2, 2, 4, 4])
        assert abs(score - 4 / 100) < 1e-9

    def test_zero_area_box(self):
        assert iou([5, 5, 5, 5], [0, 0, 10, 10]) == 0.0


class TestDetectionMetrics:
    def test_perfect_match(self):
        boxes = [[0, 0, 10, 10]]
        counts = detection_metrics(boxes, boxes)
        assert counts.tp == 1
        assert counts.fp == 0
        assert counts.fn == 0
        assert counts.precision == 1.0
        assert counts.recall == 1.0

    def test_all_false_positives(self):
        pred = [[100, 100, 200, 200]]
        gt = [[0, 0, 10, 10]]
        counts = detection_metrics(pred, gt)
        assert counts.tp == 0
        assert counts.fp == 1
        assert counts.fn == 1

    def test_all_false_negatives(self):
        counts = detection_metrics([], [[0, 0, 10, 10], [20, 20, 30, 30]])
        assert counts.tp == 0
        assert counts.fp == 0
        assert counts.fn == 2
        assert counts.recall == 0.0

    def test_precision_and_recall_computed(self):
        # 2 TP, 1 FP, 1 FN
        counts = DetectionCounts(tp=2, fp=1, fn=1)
        assert abs(counts.precision - 2 / 3) < 1e-9
        assert abs(counts.recall - 2 / 3) < 1e-9

    def test_multiple_preds_one_gt(self):
        # Only one pred can match the single GT box; the rest are FP
        gt = [[0, 0, 10, 10]]
        preds = [[0, 0, 10, 10], [0, 0, 10, 10], [0, 0, 10, 10]]
        counts = detection_metrics(preds, gt)
        assert counts.tp == 1
        assert counts.fp == 2
        assert counts.fn == 0


class TestOcrMetrics:
    def _make_gt(self, texts):
        return {
            "pages": [
                {
                    "filename": "page_001.png",
                    "category": "dense_dialogue",
                    "bubbles": [
                        {"index": i, "bbox": [0, 0, 10, 10], "japanese": t, "character": "", "notes": ""}
                        for i, t in enumerate(texts)
                    ],
                }
            ]
        }

    def test_perfect_ocr(self):
        gt = self._make_gt(["テスト", "日本語"])
        our_dumps = {
            "page_001.png": {
                "filename": "page_001.png",
                "page_number": 0,
                "boxes": [],
                "ocr": [
                    {"index": 0, "text": "テスト", "bbox": [0, 0, 10, 10]},
                    {"index": 1, "text": "日本語", "bbox": [0, 0, 10, 10]},
                ],
            }
        }
        metrics = compute_ocr_metrics(gt, our_dumps, {})
        assert len(metrics) == 1
        assert metrics[0].our_cer == 0.0

    def test_missing_dump_is_full_error(self):
        gt = self._make_gt(["テスト"])
        metrics = compute_ocr_metrics(gt, {}, {})
        assert metrics[0].our_cer == 1.0

    def test_placeholder_pages_skipped(self):
        gt = {
            "pages": [
                {
                    "filename": "page_001.png",
                    "category": "minimal",
                    "bubbles": [
                        {"index": 0, "bbox": [0, 0, 0, 0], "japanese": "PLACEHOLDER", "character": "", "notes": ""}
                    ],
                }
            ]
        }
        metrics = compute_ocr_metrics(gt, {}, {})
        assert metrics == []


class TestDetectionMetricsAll:
    def test_skips_placeholder_bboxes(self):
        gt = {
            "pages": [
                {
                    "filename": "page_001.png",
                    "bubbles": [{"index": 0, "bbox": [0, 0, 0, 0], "japanese": ""}],
                }
            ]
        }
        metrics = compute_detection_metrics_all(gt, {})
        assert metrics == []

    def test_counts_fn_when_no_dump(self):
        gt = {
            "pages": [
                {
                    "filename": "page_001.png",
                    "bubbles": [{"index": 0, "bbox": [10, 10, 100, 100], "japanese": "テスト"}],
                }
            ]
        }
        metrics = compute_detection_metrics_all(gt, {})
        assert metrics[0].our.fn == 1
        assert metrics[0].our.tp == 0


class TestWriteResultsMd:
    def test_creates_file(self, tmp_path):
        output = tmp_path / "results.md"
        write_results_md([], [], output)
        assert output.exists()
        content = output.read_text()
        assert "# Benchmark Results" in content
        assert "## Detection" in content
        assert "## OCR" in content
        assert "## Translation" in content
        assert "## Action items" in content
