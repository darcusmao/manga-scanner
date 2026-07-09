from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
from PIL import Image

from manga_scanner.config import Config
from manga_scanner.detection.detector import TextDetector
from manga_scanner.detection.masker import generate_mask, load_image
from manga_scanner.detection.sorter import sort_reading_order
from manga_scanner.inpainting.inpainter import Inpainter
from manga_scanner.ocr.cropper import extract_crops
from manga_scanner.ocr.ocr import MangaOCR
from manga_scanner.translation.translator import Translator
from manga_scanner.typesetting.renderer import render_translations
from manga_scanner.types import OCRResult, PageJob, RenderResult

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
    Models are passed in as pre-constructed instances managed by the caller.
    """
    t_start = time.monotonic()
    logger.info("Processing page: %s", job.input_path.name)

    image = load_image(job.input_path)

    t = time.monotonic()
    detection = detector.detect(job.input_path)
    logger.info("  Detection: %d boxes found (%.2fs)", len(detection.boxes), time.monotonic() - t)

    if not detection.boxes:
        logger.warning("  No text detected. Saving original image.")
        output = Image.fromarray(image)
        output.save(job.output_path, format="PNG")
        return RenderResult(image=output, output_path=job.output_path)

    ordered_boxes = sort_reading_order(detection.boxes, config.detection.row_threshold)

    mask_result = generate_mask(image, ordered_boxes, padding=config.detection.box_padding)

    t = time.monotonic()
    try:
        clean_canvas = inpainter.inpaint(mask_result.original, mask_result.mask)
    except Exception as e:
        logger.error("  Inpainting failed: %s. Using original image as canvas.", e)
        clean_canvas = image
    logger.info("  Inpainting complete (%.2fs)", time.monotonic() - t)

    # Crops come from the ORIGINAL image — inpainting already erased the text
    crops = extract_crops(image, ordered_boxes)

    t = time.monotonic()
    ocr_results: list[OCRResult] = []
    for crop_result in crops:
        text = ocr.transcribe(crop_result.crop)
        ocr_results.append(OCRResult(box=crop_result.box, text=text, index=crop_result.index))
    logger.info("  OCR complete: %d texts (%.2fs)", len(ocr_results), time.monotonic() - t)

    valid_ocr = [r for r in ocr_results if r.text.strip()]
    if not valid_ocr:
        logger.warning("  All OCR results were empty. Saving inpainted canvas without text.")
        output = Image.fromarray(clean_canvas)
        output.save(job.output_path, format="PNG")
        return RenderResult(image=output, output_path=job.output_path)

    t = time.monotonic()
    jp_texts = [r.text for r in valid_ocr]
    translation_result = translator.translate_page(jp_texts, job.character_profiles, job.page_number)
    logger.info("  Translation complete (%.2fs)", time.monotonic() - t)

    clean_pil = Image.fromarray(clean_canvas)
    valid_boxes = [r.box for r in valid_ocr]
    output = render_translations(clean_pil, translation_result.translations, valid_boxes, config.typesetting)

    output.save(job.output_path, format="PNG")
    logger.info("Page %s complete in %.2fs", job.input_path.name, time.monotonic() - t_start)

    return RenderResult(image=output, output_path=job.output_path)
