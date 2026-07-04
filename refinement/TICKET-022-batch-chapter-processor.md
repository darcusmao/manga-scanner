# TICKET-022: Batch Chapter Directory Processor

## Summary
Write `process_chapter()` — the function that walks a directory of manga page images, constructs `PageJob` objects, manages model lifetimes across the full chapter, and calls `process_page()` for each page. This is the primary entry point for real-world usage.

## Language and Tools
- Python 3.11
- `tqdm` — progress bar
- Install: `uv add tqdm`
- All project modules

## VRAM Strategy for Batch Processing

Based on TICKET-020, the batch processor uses a staged approach rather than full pipeline per page:

```
Stage A: Detect all pages    → keep TextDetector loaded across all pages
Stage B: Inpaint all pages   → keep Inpainter loaded across all pages  
Stage C: OCR all pages       → keep MangaOCR loaded across all pages
Stage D: Translate all pages → call Translator for each page (Ollama manages VRAM)
Stage E: Render all pages    → CPU only, no VRAM
```

This means intermediate results (detection results, inpainted arrays, OCR results) must be held in memory between stages. For a 20-page chapter at 2000x3000px:
- Detection results: negligible (just bounding box lists)
- Inpainted images: 20 * 2000 * 3000 * 3 bytes = ~360 MB RAM
- OCR results: negligible (just strings)

360 MB is acceptable for typical system RAM (16GB+). If processing very high-resolution scans or very long chapters, add an option to write intermediates to a temp directory.

## Implementation

File: `src/manga_scanner/pipeline/batch.py`

```python
import logging
from pathlib import Path
from dataclasses import dataclass
import numpy as np
from PIL import Image
from tqdm import tqdm

from manga_scanner.config import Config
from manga_scanner.detection.detector import TextDetector
from manga_scanner.detection.masker import generate_mask, load_image
from manga_scanner.detection.sorter import sort_reading_order
from manga_scanner.inpainting.inpainter import Inpainter
from manga_scanner.ocr.ocr import MangaOCR
from manga_scanner.ocr.cropper import extract_crops
from manga_scanner.translation.translator import Translator
from manga_scanner.translation.models import CharacterProfile, load_character_profiles
from manga_scanner.typesetting.renderer import render_translations
from manga_scanner.types import DetectionResult, OCRResult
from manga_scanner.output import resolve_output_path
from manga_scanner.vram import managed_model, clear_cuda_cache, log_vram

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


@dataclass
class PageIntermediate:
    input_path: Path
    output_path: Path
    page_number: int
    detection: DetectionResult
    clean_canvas: np.ndarray        # inpainted image
    original_image: np.ndarray      # for OCR crops
    ocr_results: list[OCRResult]


def collect_pages(input_dir: Path) -> list[Path]:
    pages = sorted(
        p for p in input_dir.iterdir()
        if p.suffix.lower() in IMAGE_EXTENSIONS
    )
    return pages


def process_chapter(
    input_dir: Path,
    output_root: Path,
    config: Config,
    characters_path: Path,
) -> None:
    characters = load_character_profiles(characters_path)
    pages = collect_pages(input_dir)
    if not pages:
        logger.warning("No image files found in %s", input_dir)
        return

    # Resolve output paths and apply skip_existing filter
    resolutions = [
        resolve_output_path(p, input_dir, output_root, config.pipeline.skip_existing)
        for p in pages
    ]
    to_process = [(p, r) for p, r in zip(pages, resolutions) if not r.skip]
    skipped = len(pages) - len(to_process)
    if skipped:
        logger.info("Skipping %d already-processed pages.", skipped)
    if not to_process:
        logger.info("All pages already processed.")
        return

    intermediates: list[PageIntermediate] = []
    failed_pages: list[Path] = []

    # Stage A: Detection (all pages)
    logger.info("Stage A: Detecting text regions across %d pages...", len(to_process))
    with managed_model(lambda: TextDetector(config.detection)) as detector:
        for idx, (page_path, resolution) in enumerate(tqdm(to_process, desc="Detect")):
            try:
                detection = detector.detect(page_path)
                image = load_image(page_path)
                intermediates.append(PageIntermediate(
                    input_path=page_path,
                    output_path=resolution.path,
                    page_number=idx,
                    detection=detection,
                    clean_canvas=image,   # placeholder, filled in Stage B
                    original_image=image,
                    ocr_results=[],
                ))
            except Exception as e:
                logger.error("Detection failed for %s: %s", page_path.name, e)
                failed_pages.append(page_path)

    log_vram("after detection stage")

    # Stage B: Inpainting (all pages)
    logger.info("Stage B: Inpainting %d pages...", len(intermediates))
    with managed_model(lambda: Inpainter(config.inpainting)) as inpainter:
        for item in tqdm(intermediates, desc="Inpaint"):
            if not item.detection.boxes:
                continue
            try:
                ordered = sort_reading_order(item.detection.boxes)
                mask_result = generate_mask(
                    item.original_image, ordered, padding=config.detection.box_padding
                )
                item.clean_canvas = inpainter.inpaint(mask_result.original, mask_result.mask)
                item.detection = item.detection.__class__(
                    image_path=item.detection.image_path,
                    boxes=ordered,
                )
            except Exception as e:
                logger.error("Inpainting failed for %s: %s", item.input_path.name, e)
                # clean_canvas stays as the original image

    log_vram("after inpainting stage")

    # Stage C: OCR (all pages)
    logger.info("Stage C: Running OCR across all pages...")
    with managed_model(lambda: MangaOCR(config.ocr)) as ocr:
        for item in tqdm(intermediates, desc="OCR"):
            if not item.detection.boxes:
                continue
            crops = extract_crops(item.original_image, item.detection.boxes)
            for crop_result in crops:
                text = ocr.transcribe(crop_result.crop)
                item.ocr_results.append(
                    OCRResult(box=crop_result.box, text=text, index=crop_result.index)
                )

    log_vram("after OCR stage")

    # Stage D: Translation (all pages, Ollama manages its own VRAM)
    logger.info("Stage D: Translating pages...")
    translator = Translator(config.translation)
    for item in tqdm(intermediates, desc="Translate"):
        valid_ocr = [r for r in item.ocr_results if r.text.strip()]
        if not valid_ocr:
            logger.warning("No valid OCR text for %s, saving inpainted canvas.", item.input_path.name)
            Image.fromarray(item.clean_canvas).save(item.output_path, format="PNG")
            continue
        try:
            jp_texts = [r.text for r in valid_ocr]
            translation = translator.translate_page(jp_texts, characters, item.page_number)

            # Stage E: Render and save
            output = render_translations(
                Image.fromarray(item.clean_canvas),
                translation.translations,
                [r.box for r in valid_ocr],
                config.typesetting,
            )
            output.save(item.output_path, format="PNG")
            logger.info("Saved: %s", item.output_path.name)
        except Exception as e:
            logger.error("Render/save failed for %s: %s", item.input_path.name, e)
            failed_pages.append(item.input_path)

    translator.close()

    # Summary
    processed = len(intermediates) - len(failed_pages)
    logger.info(
        "Chapter complete. Processed: %d  Skipped: %d  Failed: %d",
        processed, skipped, len(failed_pages)
    )
    if failed_pages:
        logger.warning("Failed pages: %s", [p.name for p in failed_pages])
```

## Acceptance Criteria
- `process_chapter(input_dir, output_root, config, characters_path)` processes all `.png`/`.jpg` files in `input_dir`
- Output files appear in the expected mirrored path under `output_root`
- Pages that already exist in output are skipped when `skip_existing=True`
- A failure on one page does not abort processing of subsequent pages
- Final log line reports processed/skipped/failed counts
- Progress bars display for each stage

## Dependencies
- TICKET-001 through TICKET-021 (all modules)
- TICKET-019 (output path management)
- TICKET-020 (VRAM lifecycle utilities)

## Estimated Effort
4 hours
