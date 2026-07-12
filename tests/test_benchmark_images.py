"""Tests using the real benchmark PNG images (ML models mocked throughout).

Covers things the 800×1200 synthetic fixtures cannot:
- RGBA source images converted to RGB
- Large dimensions (~2036×1598)
- Filenames with spaces and special characters
- 4-page chapter run

All tests skip if no PNGs are present in tests/fixtures/benchmark/.
"""
import sys
import numpy as np
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
from PIL import Image

BENCHMARK_DIR = Path(__file__).parent / "fixtures" / "benchmark"

if sys.platform == "darwin":
    _TEST_FONT = "/System/Library/Fonts/Helvetica.ttc"
else:
    _TEST_FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


def _benchmark_images() -> list[Path]:
    return sorted(BENCHMARK_DIR.glob("*.png"))


def _require_images():
    imgs = _benchmark_images()
    if not imgs:
        pytest.skip("No PNG images in tests/fixtures/benchmark/")
    return imgs


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------

class TestBenchmarkImageLoading:
    def test_all_images_load_as_rgb(self):
        images = _require_images()
        from manga_scanner.detection.masker import load_image
        for path in images:
            arr = load_image(path)
            assert arr.ndim == 3, f"{path.name} not 3-dimensional"
            assert arr.shape[2] == 3, f"{path.name} has {arr.shape[2]} channels, expected 3"
            assert arr.dtype == np.uint8

    def test_rgba_source_converted_to_rgb(self):
        images = _require_images()
        rgba_images = [p for p in images if Image.open(p).mode == "RGBA"]
        if not rgba_images:
            pytest.skip("No RGBA images found in benchmark directory")
        from manga_scanner.detection.masker import load_image
        for path in rgba_images:
            arr = load_image(path)
            assert arr.shape[2] == 3, f"{path.name}: RGBA not converted to RGB"

    def test_image_dimensions_are_large(self):
        images = _require_images()
        from manga_scanner.detection.masker import load_image
        for path in images:
            arr = load_image(path)
            h, w = arr.shape[:2]
            assert h > 800 and w > 800, f"{path.name}: unexpectedly small ({w}×{h})"

    def test_filenames_with_spaces_are_found(self):
        images = _require_images()
        space_names = [p for p in images if " " in p.name]
        assert space_names, "Expected screenshots with spaces in filenames"


# ---------------------------------------------------------------------------
# Masking on real image dimensions
# ---------------------------------------------------------------------------

class TestMaskOnBenchmarkImages:
    def test_mask_shape_matches_image(self):
        images = _require_images()
        from manga_scanner.detection.masker import load_image, generate_mask
        from manga_scanner.types import BoundingBox
        path = images[0]
        arr = load_image(path)
        h, w = arr.shape[:2]
        boxes = [BoundingBox(x1=100, y1=100, x2=w // 3, y2=200, confidence=0.9)]
        result = generate_mask(arr, boxes)
        assert result.mask.shape == (h, w)

    def test_mask_covers_box_region(self):
        images = _require_images()
        from manga_scanner.detection.masker import load_image, generate_mask
        from manga_scanner.types import BoundingBox
        arr = load_image(images[0])
        h, w = arr.shape[:2]
        box = BoundingBox(x1=200, y1=200, x2=600, y2=400, confidence=0.9)
        result = generate_mask(arr, [box], padding=0)
        # Interior of box should be masked
        assert result.mask[300, 400] == 255
        # Outside should be zero
        assert result.mask[0, 0] == 0

    def test_mask_with_padding_does_not_exceed_bounds(self):
        images = _require_images()
        from manga_scanner.detection.masker import load_image, generate_mask
        from manga_scanner.types import BoundingBox
        arr = load_image(images[0])
        h, w = arr.shape[:2]
        # Box near the edge — padding should clamp, not raise
        box = BoundingBox(x1=w - 5, y1=h - 5, x2=w + 50, y2=h + 50, confidence=0.9)
        result = generate_mask(arr, [box], padding=20)
        assert result.mask.shape == (h, w)


# ---------------------------------------------------------------------------
# Crop extraction on real images
# ---------------------------------------------------------------------------

class TestCropOnBenchmarkImages:
    def test_crop_from_large_image(self):
        images = _require_images()
        from manga_scanner.detection.masker import load_image
        from manga_scanner.ocr.cropper import extract_crops
        from manga_scanner.types import BoundingBox
        arr = load_image(images[0])
        h, w = arr.shape[:2]
        boxes = [
            BoundingBox(x1=100, y1=100, x2=500, y2=300, confidence=0.9),
            BoundingBox(x1=w // 2, y1=h // 2, x2=w // 2 + 400, y2=h // 2 + 200, confidence=0.85),
        ]
        crops = extract_crops(arr, boxes)
        assert len(crops) == 2
        assert crops[0].crop.size == (400, 200)

    def test_box_clamped_to_large_image_bounds(self):
        images = _require_images()
        from manga_scanner.detection.masker import load_image
        from manga_scanner.ocr.cropper import extract_crops
        from manga_scanner.types import BoundingBox
        arr = load_image(images[0])
        h, w = arr.shape[:2]
        box = BoundingBox(x1=-10, y1=-10, x2=w + 100, y2=h + 100, confidence=0.9)
        crops = extract_crops(arr, [box])
        assert len(crops) == 1
        assert crops[0].crop.size == (w, h)


# ---------------------------------------------------------------------------
# Output path resolution with spaces in filenames
# ---------------------------------------------------------------------------

class TestOutputPathWithSpaces:
    def test_space_in_filename_resolves_to_png(self, tmp_path):
        images = _require_images()
        space_path = next((p for p in images if " " in p.name), None)
        if space_path is None:
            pytest.skip("No space filenames found")
        from manga_scanner.output import resolve_output_path
        resolution = resolve_output_path(
            space_path, BENCHMARK_DIR, tmp_path / "out", skip_existing=False
        )
        assert resolution.path.suffix == ".png"
        assert " " in resolution.path.name

    def test_output_parent_dir_created_for_space_filename(self, tmp_path):
        images = _require_images()
        space_path = next((p for p in images if " " in p.name), None)
        if space_path is None:
            pytest.skip("No space filenames found")
        from manga_scanner.output import resolve_output_path
        output_root = tmp_path / "out"
        resolution = resolve_output_path(
            space_path, BENCHMARK_DIR, output_root, skip_existing=False
        )
        assert resolution.path.parent.exists()


# ---------------------------------------------------------------------------
# Full pipeline integration (all models mocked)
# ---------------------------------------------------------------------------

@pytest.fixture
def benchmark_config():
    from manga_scanner.config import Config, TypesettingConfig
    c = Config()
    object.__setattr__(c, "typesetting", TypesettingConfig(font_path=_TEST_FONT))
    return c


class TestPipelineOnBenchmarkImages:
    def test_pipeline_produces_one_output_per_input(self, benchmark_config, tmp_path):
        images = _require_images()
        from manga_scanner.types import BoundingBox, DetectionResult, TranslationResult

        def make_detection(image_path):
            from manga_scanner.detection.masker import load_image
            arr = load_image(image_path)
            h, w = arr.shape[:2]
            return DetectionResult(
                image_path=image_path,
                boxes=[
                    BoundingBox(x1=100, y1=100, x2=w // 3, y2=200, confidence=0.9),
                    BoundingBox(x1=w // 2, y1=h // 2, x2=w - 100, y2=h // 2 + 100, confidence=0.85),
                ],
            )

        with (
            patch("manga_scanner.pipeline.batch.TextDetector") as MockDet,
            patch("manga_scanner.pipeline.batch.Inpainter") as MockInp,
            patch("manga_scanner.pipeline.batch.MangaOCR") as MockOCR,
            patch("manga_scanner.pipeline.batch.Translator") as MockTrans,
        ):
            mock_det = MagicMock()
            mock_det.detect.side_effect = make_detection
            MockDet.return_value = mock_det

            mock_inp = MagicMock()
            mock_inp.inpaint.side_effect = lambda img, mask: img.copy()
            MockInp.return_value = mock_inp

            mock_ocr = MagicMock()
            mock_ocr.transcribe.return_value = "テスト"
            MockOCR.return_value = mock_ocr

            mock_trans = MagicMock()
            mock_trans.translate_page.return_value = TranslationResult(
                translations=["Text one", "Text two"],
                raw_response='["Text one", "Text two"]',
            )
            mock_trans.close = MagicMock()
            MockTrans.return_value = mock_trans

            from manga_scanner.pipeline.batch import process_chapter
            process_chapter(
                input_dir=BENCHMARK_DIR,
                output_root=tmp_path / "out",
                config=benchmark_config,
                characters_path=Path("characters.json"),
            )

        outputs = list((tmp_path / "out").rglob("*.png"))
        assert len(outputs) == len(images)

    def test_pipeline_output_images_are_rgb_png(self, benchmark_config, tmp_path):
        images = _require_images()
        from manga_scanner.types import BoundingBox, DetectionResult, TranslationResult

        def make_detection(image_path):
            from manga_scanner.detection.masker import load_image
            arr = load_image(image_path)
            h, w = arr.shape[:2]
            return DetectionResult(
                image_path=image_path,
                boxes=[BoundingBox(x1=100, y1=100, x2=w // 4, y2=250, confidence=0.9)],
            )

        with (
            patch("manga_scanner.pipeline.batch.TextDetector") as MockDet,
            patch("manga_scanner.pipeline.batch.Inpainter") as MockInp,
            patch("manga_scanner.pipeline.batch.MangaOCR") as MockOCR,
            patch("manga_scanner.pipeline.batch.Translator") as MockTrans,
        ):
            MockDet.return_value.detect.side_effect = make_detection
            MockInp.return_value.inpaint.side_effect = lambda img, mask: img.copy()
            MockOCR.return_value.transcribe.return_value = "テスト"
            mock_trans = MagicMock()
            mock_trans.translate_page.return_value = TranslationResult(
                translations=["Hello"], raw_response='["Hello"]'
            )
            mock_trans.close = MagicMock()
            MockTrans.return_value = mock_trans

            from manga_scanner.pipeline.batch import process_chapter
            process_chapter(
                input_dir=BENCHMARK_DIR,
                output_root=tmp_path / "out",
                config=benchmark_config,
                characters_path=Path("characters.json"),
            )

        for out in (tmp_path / "out").rglob("*.png"):
            img = Image.open(out)
            assert img.mode == "RGB", f"{out.name} is {img.mode}, expected RGB"
            assert img.format == "PNG" or out.suffix == ".png"

    def test_pipeline_preserves_image_dimensions(self, benchmark_config, tmp_path):
        images = _require_images()
        from manga_scanner.types import BoundingBox, DetectionResult, TranslationResult

        # Map filename → expected size
        expected_sizes: dict[str, tuple[int, int]] = {}
        for p in images:
            img = Image.open(p).convert("RGB")
            expected_sizes[p.stem + ".png"] = img.size  # (width, height)

        def make_detection(image_path):
            from manga_scanner.detection.masker import load_image
            arr = load_image(image_path)
            h, w = arr.shape[:2]
            return DetectionResult(
                image_path=image_path,
                boxes=[BoundingBox(x1=50, y1=50, x2=w // 5, y2=200, confidence=0.9)],
            )

        with (
            patch("manga_scanner.pipeline.batch.TextDetector") as MockDet,
            patch("manga_scanner.pipeline.batch.Inpainter") as MockInp,
            patch("manga_scanner.pipeline.batch.MangaOCR") as MockOCR,
            patch("manga_scanner.pipeline.batch.Translator") as MockTrans,
        ):
            MockDet.return_value.detect.side_effect = make_detection
            MockInp.return_value.inpaint.side_effect = lambda img, mask: img.copy()
            MockOCR.return_value.transcribe.return_value = "テスト"
            mock_trans = MagicMock()
            mock_trans.translate_page.return_value = TranslationResult(
                translations=["Text"], raw_response='["Text"]'
            )
            mock_trans.close = MagicMock()
            MockTrans.return_value = mock_trans

            from manga_scanner.pipeline.batch import process_chapter
            process_chapter(
                input_dir=BENCHMARK_DIR,
                output_root=tmp_path / "out",
                config=benchmark_config,
                characters_path=Path("characters.json"),
            )

        for out in (tmp_path / "out").rglob("*.png"):
            actual_size = Image.open(out).size
            expected = expected_sizes.get(out.name)
            if expected is not None:
                assert actual_size == expected, (
                    f"{out.name}: got {actual_size}, expected {expected}"
                )

    def test_dump_dir_produces_valid_schema(self, benchmark_config, tmp_path):
        import json
        images = _require_images()
        from manga_scanner.types import BoundingBox, DetectionResult, TranslationResult

        def make_detection(image_path):
            from manga_scanner.detection.masker import load_image
            arr = load_image(image_path)
            h, w = arr.shape[:2]
            return DetectionResult(
                image_path=image_path,
                boxes=[BoundingBox(x1=100, y1=100, x2=400, y2=300, confidence=0.91)],
            )

        dump_dir = tmp_path / "dumps"

        with (
            patch("manga_scanner.pipeline.batch.TextDetector") as MockDet,
            patch("manga_scanner.pipeline.batch.Inpainter") as MockInp,
            patch("manga_scanner.pipeline.batch.MangaOCR") as MockOCR,
            patch("manga_scanner.pipeline.batch.Translator") as MockTrans,
        ):
            MockDet.return_value.detect.side_effect = make_detection
            MockInp.return_value.inpaint.side_effect = lambda img, mask: img.copy()
            MockOCR.return_value.transcribe.return_value = "テスト"
            mock_trans = MagicMock()
            mock_trans.translate_page.return_value = TranslationResult(
                translations=["Test"], raw_response='["Test"]'
            )
            mock_trans.close = MagicMock()
            MockTrans.return_value = mock_trans

            from manga_scanner.pipeline.batch import process_chapter
            process_chapter(
                input_dir=BENCHMARK_DIR,
                output_root=tmp_path / "out",
                config=benchmark_config,
                characters_path=Path("characters.json"),
                dump_dir=dump_dir,
            )

        dump_files = list(dump_dir.glob("*.json"))
        assert len(dump_files) == len(images)

        for dump_file in dump_files:
            data = json.loads(dump_file.read_text())
            assert "filename" in data
            assert "page_number" in data
            assert "boxes" in data
            assert "ocr" in data
            # Each box has the right keys
            for box in data["boxes"]:
                assert "bbox" in box and len(box["bbox"]) == 4
                assert "confidence" in box
            # Each OCR result has text
            for ocr in data["ocr"]:
                assert "text" in ocr
                assert "bbox" in ocr
