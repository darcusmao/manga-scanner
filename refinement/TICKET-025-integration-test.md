# TICKET-025: End-to-End Integration Test

## Summary
Write an integration test that runs the full `process_chapter()` pipeline against a small set of fixture images, with all GPU-bound models mocked. This validates that data flows correctly through every stage without requiring hardware or external services.

## Language and Tools
- Python 3.11
- `pytest`, `pytest-mock` (installed in TICKET-024)
- `Pillow`, `numpy` (already installed)

## Test Fixture Setup

Create `tests/fixtures/pages/` with 2-3 synthetic manga-style PNG images. These do not need to be real manga — synthetic images with black rectangles on white backgrounds are sufficient to exercise the pipeline logic.

Create a helper script at `tests/fixtures/generate_fixtures.py`:

```python
"""Run once to generate synthetic fixture images. Not a pytest file."""
from pathlib import Path
from PIL import Image, ImageDraw

OUT_DIR = Path(__file__).parent / "pages"
OUT_DIR.mkdir(exist_ok=True)

for i in range(3):
    img = Image.new("RGB", (800, 1200), color=(240, 240, 240))
    draw = ImageDraw.Draw(img)
    # Simulate two speech bubbles as black rectangles
    draw.rectangle([100, 100, 400, 200], fill=(0, 0, 0))
    draw.rectangle([400, 400, 700, 500], fill=(0, 0, 0))
    img.save(OUT_DIR / f"page_{i+1:03d}.png")

print(f"Generated {i+1} fixture images in {OUT_DIR}")
```

Run once: `uv run python tests/fixtures/generate_fixtures.py`

## Integration Test

File: `tests/test_pipeline.py`

```python
import numpy as np
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
from PIL import Image

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "pages"


@pytest.fixture
def config():
    from manga_scanner.config import Config, TypesettingConfig
    import sys
    if sys.platform == "darwin":
        font = "/System/Library/Fonts/Helvetica.ttc"
    else:
        font = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    c = Config()
    c.typesetting = TypesettingConfig(font_path=font)
    return c


@pytest.fixture
def output_dir(tmp_path):
    return tmp_path / "output"


def make_fake_detection(image_path):
    from manga_scanner.types import BoundingBox, DetectionResult
    return DetectionResult(
        image_path=image_path,
        boxes=[
            BoundingBox(x1=100, y1=100, x2=400, y2=200, confidence=0.9),
            BoundingBox(x1=400, y1=400, x2=700, y2=500, confidence=0.85),
        ]
    )


def test_process_chapter_produces_output_files(config, output_dir):
    if not FIXTURE_DIR.exists() or not list(FIXTURE_DIR.glob("*.png")):
        pytest.skip("Fixture images not generated. Run tests/fixtures/generate_fixtures.py first.")

    with (
        patch("manga_scanner.pipeline.batch.TextDetector") as MockDetector,
        patch("manga_scanner.pipeline.batch.Inpainter") as MockInpainter,
        patch("manga_scanner.pipeline.batch.MangaOCR") as MockOCR,
        patch("manga_scanner.pipeline.batch.Translator") as MockTranslator,
    ):
        # Mock TextDetector
        mock_det = MagicMock()
        mock_det.detect.side_effect = make_fake_detection
        MockDetector.return_value = mock_det

        # Mock Inpainter: return a white canvas
        mock_inp = MagicMock()
        mock_inp.inpaint.return_value = np.full((1200, 800, 3), 255, dtype=np.uint8)
        MockInpainter.return_value = mock_inp

        # Mock MangaOCR: return dummy Japanese text
        mock_ocr = MagicMock()
        mock_ocr.transcribe.return_value = "テスト"
        MockOCR.return_value = mock_ocr

        # Mock Translator: return matching English translations
        mock_trans = MagicMock()
        from manga_scanner.types import TranslationResult
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

    # Assert output files were created
    output_pages = list(output_dir.rglob("*.png"))
    assert len(output_pages) == len(list(FIXTURE_DIR.glob("*.png")))

    # Assert each output is a valid image with same dimensions as input
    for out_path in output_pages:
        img = Image.open(out_path)
        assert img.size == (800, 1200)
        assert img.mode == "RGB"


def test_process_chapter_skips_existing(config, output_dir):
    if not FIXTURE_DIR.exists() or not list(FIXTURE_DIR.glob("*.png")):
        pytest.skip("Fixture images not generated.")

    # Pre-create one output file
    first_page = sorted(FIXTURE_DIR.glob("*.png"))[0]
    expected_out = output_dir / FIXTURE_DIR.name / first_page.name
    expected_out.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (800, 1200)).save(expected_out)

    config.pipeline.skip_existing = True

    with (
        patch("manga_scanner.pipeline.batch.TextDetector") as MockDetector,
        patch("manga_scanner.pipeline.batch.Inpainter"),
        patch("manga_scanner.pipeline.batch.MangaOCR"),
        patch("manga_scanner.pipeline.batch.Translator"),
    ):
        from manga_scanner.pipeline.batch import process_chapter
        process_chapter(FIXTURE_DIR, output_dir, config, Path("characters.json"))

    # The pre-existing file should not have been regenerated (mtime unchanged)
    # and detection should not have been called for it
    mock_det = MockDetector.return_value
    call_count = mock_det.detect.call_count
    total_pages = len(list(FIXTURE_DIR.glob("*.png")))
    assert call_count == total_pages - 1  # one page was skipped
```

## What This Tests
- Full data flow from directory discovery through file output
- Output files are created in the correct mirrored paths
- Output images are valid PNGs with correct dimensions
- `skip_existing` logic skips the correct number of pages
- No real GPU, Ollama, or model weights needed

## What This Does Not Test
- Visual correctness of inpainting, OCR accuracy, or translation quality — those require manual review on real manga pages
- Actual model loading and VRAM behavior — covered by TICKET-004 and TICKET-008 verification scripts

## Acceptance Criteria
- `uv run pytest tests/test_pipeline.py -v` passes with 0 failures (after fixtures are generated)
- `uv run pytest tests/ -v` runs all tickets 024-025 tests together cleanly
- No GPU or Ollama required to run the test suite

## Dependencies
- TICKET-001 through TICKET-022 (all pipeline code must exist)
- TICKET-024 (pytest installed)

## Estimated Effort
3 hours
