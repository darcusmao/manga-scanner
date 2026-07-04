# TICKET-021: Single-Page Pipeline Orchestrator

## Summary
Write `process_page()` — the function that chains all five pipeline stages for a single image: detect, mask, inpaint, OCR, translate, render, save. All image data passes through memory without intermediate disk writes. Per-stage error handling ensures a failure in one stage produces a degraded output rather than crashing the batch.

## Language and Tools
- Python 3.11
- `Pillow`, `numpy` (already installed)
- All project modules (TICKETS 005-019 must be complete)

## Implementation

File: `src/manga_scanner/pipeline/orchestrator.py`

```python
import logging
import time
from pathlib import Path
from PIL import Image
import numpy as np

from manga_scanner.config import Config
from manga_scanner.detection.detector import TextDetector
from manga_scanner.detection.masker import generate_mask, load_image
from manga_scanner.detection.sorter import sort_reading_order
from manga_scanner.inpainting.inpainter import Inpainter
from manga_scanner.ocr.ocr import MangaOCR
from manga_scanner.ocr.cropper import extract_crops
from manga_scanner.translation.translator import Translator
from manga_scanner.translation.models import CharacterProfile
from manga_scanner.typesetting.renderer import render_translations
from manga_scanner.types import PageJob, RenderResult, OCRResult

logger = logging.getLogger(__name__)


def process_page(
    job: PageJob,
    detector: TextDetector,
    inpainter: Inpainter,
    ocr: MangaOCR,
    translator: Translator,
    config: Config,
) -> RenderResult:
    """
    Process a single manga page through the full pipeline.

    Models are passed in as pre-constructed instances (managed by the batch
    processor or caller) rather than created here. This avoids per-page
    model reloading and gives the caller control over VRAM lifecycle.
    """
    t_start = time.monotonic()
    logger.info("Processing page: %s", job.input_path.name)

    # Stage 1: Load image
    image = load_image(job.input_path)

    # Stage 2: Detect text regions
    t = time.monotonic()
    detection = detector.detect(job.input_path)
    logger.info("  Detection: %d boxes found (%.2fs)", len(detection.boxes), time.monotonic() - t)

    if not detection.boxes:
        logger.warning("  No text detected. Saving original image.")
        output = Image.fromarray(image)
        output.save(job.output_path, format="PNG")
        return RenderResult(image=output, output_path=job.output_path)

    # Stage 3: Sort into reading order
    ordered_boxes = sort_reading_order(detection.boxes)

    # Stage 4: Generate inpainting mask
    mask_result = generate_mask(image, ordered_boxes, padding=config.detection.box_padding)

    # Stage 5: Inpaint
    t = time.monotonic()
    try:
        clean_canvas = inpainter.inpaint(mask_result.original, mask_result.mask)
    except Exception as e:
        logger.error("  Inpainting failed: %s. Using original image as canvas.", e)
        clean_canvas = image
    logger.info("  Inpainting complete (%.2fs)", time.monotonic() - t)

    # Stage 6: Extract crops from ORIGINAL image (before inpainting erases the text)
    crops = extract_crops(image, ordered_boxes)

    # Stage 7: OCR all crops
    t = time.monotonic()
    ocr_results: list[OCRResult] = []
    for crop_result in crops:
        text = ocr.transcribe(crop_result.crop)
        ocr_results.append(OCRResult(box=crop_result.box, text=text, index=crop_result.index))
    logger.info("  OCR complete: %d texts (%.2fs)", len(ocr_results), time.monotonic() - t)

    # Filter empty OCR results before sending to LLM
    valid_ocr = [r for r in ocr_results if r.text.strip()]
    if not valid_ocr:
        logger.warning("  All OCR results were empty. Saving inpainted canvas without text.")
        output = Image.fromarray(clean_canvas)
        output.save(job.output_path, format="PNG")
        return RenderResult(image=output, output_path=job.output_path)

    # Stage 8: Translate
    t = time.monotonic()
    jp_texts = [r.text for r in valid_ocr]
    translation_result = translator.translate_page(
        jp_texts, job.character_profiles, job.page_number
    )
    logger.info("  Translation complete (%.2fs)", time.monotonic() - t)

    # Stage 9: Render translated text onto clean canvas
    clean_pil = Image.fromarray(clean_canvas)
    valid_boxes = [r.box for r in valid_ocr]
    output = render_translations(
        clean_pil, translation_result.translations, valid_boxes, config.typesetting
    )

    # Stage 10: Save
    output.save(job.output_path, format="PNG")
    logger.info(
        "Page %s complete in %.2fs", job.input_path.name, time.monotonic() - t_start
    )

    return RenderResult(image=output, output_path=job.output_path)
```

## Critical Design Note: Crop Source
Crops (Stage 6) must be extracted from the **original image** (before inpainting), not the clean canvas. The inpainter has already erased the Japanese text from the canvas — OCRing the canvas would yield empty strings.

## Per-Stage Error Handling Strategy

| Stage | Failure | Recovery |
|---|---|---|
| Detection | Exception | Raise — unrecoverable, no boxes to work with |
| Inpainting | Exception | Use original image as canvas, log ERROR |
| OCR single crop | Exception | Return empty string for that crop (handled in MangaOCR.transcribe) |
| Translation | All retries exhausted | Return JP text as fallback (handled in Translator) |
| Rendering | Exception | Raise — indicates a code bug, not a data issue |

## Acceptance Criteria
- Given valid inputs and all models loaded, `process_page()` produces a PNG at `job.output_path`
- If detection returns no boxes, the original image is saved as-is and the function returns normally
- If inpainting raises, the function continues using the original image as the canvas
- All stages emit timing logs at INFO level
- The function does not create or destroy models — it only uses what is passed in

## Dependencies
- TICKET-002 through TICKET-019 (all module implementations)
- TICKET-020 (VRAM utilities, though orchestrator does not call them directly — caller manages lifecycle)

## Estimated Effort
4 hours
