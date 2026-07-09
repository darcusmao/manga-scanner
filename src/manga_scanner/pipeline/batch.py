from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from manga_scanner.config import Config
from manga_scanner.detection.detector import TextDetector
from manga_scanner.detection.masker import generate_mask, load_image
from manga_scanner.detection.sorter import sort_reading_order
from manga_scanner.inpainting.inpainter import Inpainter
from manga_scanner.ocr.cropper import extract_crops
from manga_scanner.ocr.ocr import MangaOCR
from manga_scanner.output import resolve_output_path
from manga_scanner.translation.models import CharacterProfile, load_character_profiles
from manga_scanner.translation.translator import Translator
from manga_scanner.typesetting.renderer import render_translations
from manga_scanner.types import DetectionResult, OCRResult
from manga_scanner.vram import clear_cuda_cache, log_vram, managed_model

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


@dataclass
class _PageIntermediate:
    input_path: Path
    output_path: Path
    page_number: int
    detection: DetectionResult
    clean_canvas: np.ndarray
    original_image: np.ndarray
    ocr_results: list[OCRResult] = field(default_factory=list)


def collect_pages(input_dir: Path) -> list[Path]:
    return sorted(p for p in input_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)


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

    intermediates: list[_PageIntermediate] = []
    failed_pages: list[Path] = []

    # Stage A: Detection
    logger.info("Stage A: Detecting text regions across %d pages...", len(to_process))
    with managed_model(lambda: TextDetector(config.detection)) as detector:
        for idx, (page_path, resolution) in enumerate(tqdm(to_process, desc="Detect")):
            try:
                detection = detector.detect(page_path)
                image = load_image(page_path)
                intermediates.append(_PageIntermediate(
                    input_path=page_path,
                    output_path=resolution.path,
                    page_number=idx,
                    detection=detection,
                    clean_canvas=image,
                    original_image=image,
                ))
            except Exception as e:
                logger.error("Detection failed for %s: %s", page_path.name, e)
                failed_pages.append(page_path)

    log_vram("after detection stage")

    # Stage B: Inpainting
    logger.info("Stage B: Inpainting %d pages...", len(intermediates))
    with managed_model(lambda: Inpainter(config.inpainting)) as inpainter:
        for item in tqdm(intermediates, desc="Inpaint"):
            if not item.detection.boxes:
                continue
            try:
                ordered = sort_reading_order(
                    item.detection.boxes, config.detection.row_threshold
                )
                mask_result = generate_mask(
                    item.original_image, ordered, padding=config.detection.box_padding
                )
                item.clean_canvas = inpainter.inpaint(mask_result.original, mask_result.mask)
                item.detection = DetectionResult(
                    image_path=item.detection.image_path,
                    boxes=ordered,
                )
            except Exception as e:
                logger.error("Inpainting failed for %s: %s", item.input_path.name, e)

    log_vram("after inpainting stage")

    # Stage C: OCR
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

    # Stage D: Translation + Stage E: Render/Save
    logger.info("Stage D: Translating and rendering pages...")
    translator = Translator(config.translation)
    for item in tqdm(intermediates, desc="Translate"):
        valid_ocr = [r for r in item.ocr_results if r.text.strip()]
        if not valid_ocr:
            logger.warning(
                "No valid OCR text for %s, saving inpainted canvas.", item.input_path.name
            )
            Image.fromarray(item.clean_canvas).save(item.output_path, format="PNG")
            continue
        try:
            jp_texts = [r.text for r in valid_ocr]
            translation = translator.translate_page(jp_texts, characters, item.page_number)
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

    processed = len(intermediates) - len(failed_pages)
    logger.info(
        "Chapter complete. Processed: %d  Skipped: %d  Failed: %d",
        processed, skipped, len(failed_pages),
    )
    if failed_pages:
        logger.warning("Failed pages: %s", [p.name for p in failed_pages])
