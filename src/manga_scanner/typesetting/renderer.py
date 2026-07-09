from __future__ import annotations

import logging

from PIL import Image, ImageDraw, ImageFont

from manga_scanner.config import TypesettingConfig
from manga_scanner.types import BoundingBox
from manga_scanner.typesetting.fitter import fit_text, FitResult

logger = logging.getLogger(__name__)


def render_translations(
    base_image: Image.Image,
    translations: list[str],
    boxes: list[BoundingBox],
    config: TypesettingConfig,
) -> Image.Image:
    """
    Draw translated text onto base_image at each bounding box location.
    translations and boxes must be the same length and in the same order.
    Returns a new PIL Image (does not mutate base_image).
    """
    assert len(translations) == len(boxes), (
        f"translations ({len(translations)}) and boxes ({len(boxes)}) must be same length"
    )

    output = base_image.copy()
    draw = ImageDraw.Draw(output)

    for text, box in zip(translations, boxes):
        if not text.strip():
            continue

        result = fit_text(
            text=text,
            box_width=box.width,
            box_height=box.height,
            font_path=config.font_path,
            max_font_size=config.max_font_size,
            min_font_size=config.min_font_size,
            padding=config.padding,
        )

        if not result.fits:
            logger.warning(
                "Text did not fit in box %dx%d at min_font_size=%d: %.40s...",
                box.width, box.height, config.min_font_size, text,
            )

        _draw_text_in_box(draw, result, box, config)

    return output


def _draw_text_in_box(
    draw: ImageDraw.ImageDraw,
    result: FitResult,
    box: BoundingBox,
    config: TypesettingConfig,
) -> None:
    if not result.lines:
        return

    font = ImageFont.truetype(config.font_path, result.font_size)
    line_spacing = int(result.font_size * 0.2)

    block_top = box.y1 + (box.height - result.total_height) // 2
    y = block_top

    for line in result.lines:
        line_w = draw.textlength(line, font=font)
        x = box.x1 + (box.width - line_w) // 2
        _draw_outlined_text(draw, (x, y), line, font, config.text_color)
        line_bbox = draw.textbbox((x, y), line, font=font)
        y += (line_bbox[3] - line_bbox[1]) + line_spacing


def _draw_outlined_text(
    draw: ImageDraw.ImageDraw,
    position: tuple[int, int],
    text: str,
    font: ImageFont.FreeTypeFont,
    fill_color: str,
    outline_color: str = "#FFFFFF",
    outline_width: int = 1,
) -> None:
    """Draw text with a 1px white outline for contrast on screentone backgrounds."""
    x, y = position
    for dx in range(-outline_width, outline_width + 1):
        for dy in range(-outline_width, outline_width + 1):
            if dx != 0 or dy != 0:
                draw.text((x + dx, y + dy), text, font=font, fill=outline_color)
    draw.text((x, y), text, font=font, fill=fill_color)
