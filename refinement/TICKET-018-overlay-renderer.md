# TICKET-018: Overlay Renderer

## Summary
Write the `render_translations` function that draws translated English text onto the inpainted canvas image, centered within each bounding box. This is the final visual composition step before the page is saved.

## Language and Tools
- Python 3.11
- `Pillow` (already installed)

## Implementation

File: `src/manga_scanner/typesetting/renderer.py`

```python
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
                box.width, box.height, config.min_font_size, text
            )

        _draw_text_in_box(draw, result, box, config)

    return output


def _draw_text_in_box(
    draw: ImageDraw.ImageDraw,
    result: FitResult,
    box: BoundingBox,
    config: TypesettingConfig,
) -> None:
    font = ImageFont.truetype(config.font_path, result.font_size)
    line_spacing = int(result.font_size * 0.2)

    # Center the text block vertically within the box
    block_top = box.y1 + (box.height - result.total_height) // 2

    y = block_top
    for line in result.lines:
        line_w = draw.textlength(line, font=font)
        # Center each line horizontally
        x = box.x1 + (box.width - line_w) // 2
        # Draw a thin white outline for legibility on dark backgrounds
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
```

## Design Notes

- `base_image.copy()` ensures the inpainted canvas is not mutated — if rendering fails partway through, the clean canvas is not lost.
- The 1px white outline on text is important for manga. Speech bubble backgrounds post-inpainting may contain screentone patterns or grey values. A white outline ensures black text reads clearly without requiring a white background fill rectangle, which would look artificial.
- We do not fill the bounding box with a white rectangle before drawing. LaMa has already erased the original text. Adding a white fill block would produce a flat white patch that clashes with the surrounding screentone. The outline approach is the cleaner solution.
- If the surrounding screentone is very dark and the outline approach is insufficient, add a config option `typesetting.background_fill: bool = false` to optionally draw a semi-transparent white rectangle.

## Acceptance Criteria
- `render_translations(image, ["Hello"], [box], config)` returns a PIL Image of the same size as input
- Text is visually centered within the bounding box
- Lines wrap correctly as computed by `fit_text`
- If `translations` and `boxes` differ in length, an AssertionError is raised immediately (fail loud, not silently)
- Empty string translations are skipped without drawing anything

## Dependencies
- TICKET-001 (project skeleton)
- TICKET-002 (BoundingBox type)
- TICKET-003 (TypesettingConfig)
- TICKET-016 (font file exists)
- TICKET-017 (fit_text function)

## Estimated Effort
3 hours
