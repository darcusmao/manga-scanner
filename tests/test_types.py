"""Tests for TICKET-002: shared dataclasses in manga_scanner.types."""
import sys
from pathlib import Path


def test_numpy_and_pil_not_imported_at_module_level():
    # TYPE_CHECKING guard must keep numpy/PIL out of sys.modules
    import manga_scanner.types  # noqa: F401
    assert "numpy" not in sys.modules
    assert "PIL" not in sys.modules


def test_bounding_box_dimensions():
    from manga_scanner.types import BoundingBox
    box = BoundingBox(x1=10, y1=20, x2=50, y2=80, confidence=0.9)
    assert box.width == 40
    assert box.height == 60
    assert box.x_center == 30
    assert box.y_center == 50


def test_bounding_box_default_label():
    from manga_scanner.types import BoundingBox
    box = BoundingBox(x1=0, y1=0, x2=10, y2=10, confidence=0.5)
    assert box.label == "text"


def test_bounding_box_custom_label():
    from manga_scanner.types import BoundingBox
    box = BoundingBox(x1=0, y1=0, x2=10, y2=10, confidence=0.5, label="sfx")
    assert box.label == "sfx"


def test_bounding_box_x_center_rounds_down():
    from manga_scanner.types import BoundingBox
    # width=11 -> x_center = x1 + 5 (integer floor division)
    box = BoundingBox(x1=0, y1=0, x2=11, y2=10, confidence=0.5)
    assert box.x_center == 5


def test_detection_result():
    from manga_scanner.types import BoundingBox, DetectionResult
    boxes = [BoundingBox(x1=0, y1=0, x2=10, y2=10, confidence=0.8)]
    result = DetectionResult(image_path=Path("page.png"), boxes=boxes)
    assert result.image_path == Path("page.png")
    assert len(result.boxes) == 1


def test_detection_result_empty_boxes():
    from manga_scanner.types import DetectionResult
    result = DetectionResult(image_path=Path("blank.png"), boxes=[])
    assert result.boxes == []


def test_ocr_result():
    from manga_scanner.types import BoundingBox, OCRResult
    box = BoundingBox(x1=0, y1=0, x2=50, y2=30, confidence=0.9)
    result = OCRResult(box=box, text="こんにちは", index=0)
    assert result.text == "こんにちは"
    assert result.index == 0


def test_ocr_result_empty_text():
    from manga_scanner.types import BoundingBox, OCRResult
    box = BoundingBox(x1=0, y1=0, x2=10, y2=10, confidence=0.3)
    result = OCRResult(box=box, text="", index=2)
    assert result.text == ""


def test_crop_result():
    from manga_scanner.types import BoundingBox, CropResult
    from PIL import Image
    box = BoundingBox(x1=0, y1=0, x2=20, y2=20, confidence=0.7)
    img = Image.new("RGB", (20, 20))
    result = CropResult(box=box, crop=img, index=1)
    assert result.index == 1


def test_translation_result():
    from manga_scanner.types import TranslationResult
    result = TranslationResult(
        translations=["Hello", "Goodbye"],
        raw_response='["Hello", "Goodbye"]',
    )
    assert len(result.translations) == 2
    assert result.translations[0] == "Hello"


def test_render_result():
    from manga_scanner.types import RenderResult
    from PIL import Image
    img = Image.new("RGB", (100, 100))
    result = RenderResult(image=img, output_path=Path("out/page_001.png"))
    assert result.output_path.suffix == ".png"


def test_page_job():
    from manga_scanner.types import PageJob
    job = PageJob(
        input_path=Path("in/001.png"),
        output_path=Path("out/001.png"),
        page_number=1,
        character_profiles=[],
    )
    assert job.page_number == 1
    assert job.character_profiles == []
