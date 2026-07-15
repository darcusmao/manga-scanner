"""Tests for TICKET-025: end-to-end integration test with all models mocked."""
import sys
import numpy as np
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
from PIL import Image

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "pages"

if sys.platform == "darwin":
    _TEST_FONT = "/System/Library/Fonts/Helvetica.ttc"
else:
    _TEST_FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


@pytest.fixture
def config():
    from manga_scanner.config import Config, TypesettingConfig
    c = Config()
    object.__setattr__(c, "typesetting", TypesettingConfig(font_path=_TEST_FONT))
    return c


@pytest.fixture
def output_dir(tmp_path):
    return tmp_path / "output"


def _make_fake_detection(image_path):
    from manga_scanner.types import BoundingBox, DetectionResult
    return DetectionResult(
        image_path=image_path,
        boxes=[
            BoundingBox(x1=100, y1=100, x2=400, y2=200, confidence=0.9),
            BoundingBox(x1=400, y1=400, x2=700, y2=500, confidence=0.85),
        ],
    )


def _patch_all_models():
    return (
        patch("manga_scanner.pipeline.batch.TextDetector"),
        patch("manga_scanner.pipeline.batch.Inpainter"),
        patch("manga_scanner.pipeline.batch.MangaOCR"),
        patch("manga_scanner.pipeline.batch.create_translator"),
    )


def test_process_chapter_produces_output_files(config, output_dir):
    if not FIXTURE_DIR.exists() or not list(FIXTURE_DIR.glob("*.png")):
        pytest.skip("Fixtures not generated. Run tests/fixtures/generate_fixtures.py first.")

    with (
        patch("manga_scanner.pipeline.batch.TextDetector") as MockDetector,
        patch("manga_scanner.pipeline.batch.Inpainter") as MockInpainter,
        patch("manga_scanner.pipeline.batch.MangaOCR") as MockOCR,
        patch("manga_scanner.pipeline.batch.create_translator") as MockTranslator,
    ):
        mock_det = MagicMock()
        mock_det.detect.side_effect = _make_fake_detection
        MockDetector.return_value = mock_det

        mock_inp = MagicMock()
        mock_inp.inpaint.return_value = np.full((1200, 800, 3), 255, dtype=np.uint8)
        MockInpainter.return_value = mock_inp

        mock_ocr = MagicMock()
        mock_ocr.transcribe.return_value = "テスト"
        MockOCR.return_value = mock_ocr

        from manga_scanner.types import TranslationResult
        mock_trans = MagicMock()
        mock_trans.translate_page.return_value = TranslationResult(
            translations=["Test text", "Another bubble"],
            raw_response='["Test text", "Another bubble"]',
        )
        MockTranslator.return_value = mock_trans

        from manga_scanner.pipeline.batch import process_chapter
        process_chapter(
            input_dir=FIXTURE_DIR,
            output_root=output_dir,
            config=config,
            characters_path=Path("characters.json"),
        )

    output_pages = list(output_dir.rglob("*.png"))
    fixture_pages = list(FIXTURE_DIR.glob("*.png"))
    assert len(output_pages) == len(fixture_pages)

    for out_path in output_pages:
        img = Image.open(out_path)
        assert img.size == (800, 1200)
        assert img.mode == "RGB"


def test_process_chapter_skips_existing(config, output_dir):
    if not FIXTURE_DIR.exists() or not list(FIXTURE_DIR.glob("*.png")):
        pytest.skip("Fixtures not generated.")

    first_page = sorted(FIXTURE_DIR.glob("*.png"))[0]
    expected_out = output_dir / FIXTURE_DIR.name / first_page.name
    expected_out.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (800, 1200)).save(expected_out)

    object.__setattr__(config.pipeline, "skip_existing", True)

    with (
        patch("manga_scanner.pipeline.batch.TextDetector") as MockDetector,
        patch("manga_scanner.pipeline.batch.Inpainter"),
        patch("manga_scanner.pipeline.batch.MangaOCR"),
        patch("manga_scanner.pipeline.batch.create_translator"),
    ):
        mock_det = MagicMock()
        mock_det.detect.side_effect = _make_fake_detection
        MockDetector.return_value = mock_det

        from manga_scanner.pipeline.batch import process_chapter
        process_chapter(FIXTURE_DIR, output_dir, config, Path("characters.json"))

    total_pages = len(list(FIXTURE_DIR.glob("*.png")))
    assert mock_det.detect.call_count == total_pages - 1


def test_process_chapter_empty_dir(config, output_dir, tmp_path):
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    from manga_scanner.pipeline.batch import process_chapter
    process_chapter(empty_dir, output_dir, config, Path("characters.json"))


def test_output_path_resolution(tmp_path):
    from manga_scanner.output import resolve_output_path
    input_path = tmp_path / "ch01" / "p001.jpg"
    input_root = tmp_path / "ch01"
    output_root = tmp_path / "output"
    input_path.parent.mkdir()
    input_path.touch()

    resolution = resolve_output_path(input_path, input_root, output_root, skip_existing=False)
    assert resolution.path.suffix == ".png"
    assert resolution.path.name == "p001.png"
    assert not resolution.skip


def test_output_path_skip_when_exists(tmp_path):
    from manga_scanner.output import resolve_output_path
    input_path = tmp_path / "ch01" / "p001.png"
    input_root = tmp_path / "ch01"
    output_root = tmp_path / "output"
    input_path.parent.mkdir()
    input_path.touch()

    resolution = resolve_output_path(input_path, input_root, output_root, skip_existing=False)
    resolution.path.touch()

    resolution2 = resolve_output_path(input_path, input_root, output_root, skip_existing=True)
    assert resolution2.skip is True


def test_vram_clear_does_not_raise():
    from manga_scanner.vram import clear_cuda_cache
    clear_cuda_cache()


def test_managed_model_calls_unload_on_exception():
    from manga_scanner.vram import managed_model

    mock = MagicMock()
    mock.unload = MagicMock()

    with pytest.raises(ValueError):
        with managed_model(lambda: mock) as m:
            raise ValueError("test error")

    mock.unload.assert_called_once()


def test_cli_help():
    import subprocess, sys
    result = subprocess.run(
        [sys.executable, "-m", "manga_scanner.cli", "--help"],
        capture_output=True, text=True,
    )
    # click --help always exits 0
    assert "manga-scan" in result.stdout or result.returncode == 0
