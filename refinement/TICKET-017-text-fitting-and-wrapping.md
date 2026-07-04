# TICKET-017: Text Fitting and Multi-line Wrapping Algorithm

## Summary
Write the algorithm that takes an English string and a bounding box dimension and finds the largest font size at which the text fits, along with the correct line breaks. This is the core computational component of the typesetting module.

## Language and Tools
- Python 3.11
- `Pillow` (already installed)
- No additional packages

## The Problem
A speech bubble contains translated English text that may be longer or shorter than the original Japanese. The bubble's pixel dimensions are fixed (from detection). We must find the largest font size where all the text fits within the bubble without overflowing, wrapping naturally at word boundaries.

## Algorithm

Two nested loops:
1. Outer loop: iterate font size from `max_font_size` down to `min_font_size`
2. Inner function: given a font size, greedily wrap text into lines that each fit within `(box_width - 2 * padding)`
3. Check: does total wrapped height fit within `(box_height - 2 * padding)`?
4. Return the first (largest) font size that satisfies the height constraint

This is a linear scan. For typical sizes (8–24pt range = 16 iterations), it is fast enough. A binary search would reduce iterations to ~4 but adds code complexity that is not warranted here.

## Implementation

File: `src/manga_scanner/typesetting/fitter.py`

```python
from dataclasses import dataclass
from PIL import Image, ImageDraw, ImageFont


@dataclass
class FitResult:
    lines: list[str]
    font_size: int
    total_width: int
    total_height: int
    fits: bool    # False if even min_font_size didn't fit (text is rendered at min size anyway)


def _wrap_text(
    text: str,
    font: ImageFont.FreeTypeFont,
    max_width: int,
    draw: ImageDraw.ImageDraw,
) -> list[str]:
    """Greedy word-wrap. Returns lines that each fit within max_width."""
    words = text.split()
    if not words:
        return []

    lines = []
    current_line = words[0]

    for word in words[1:]:
        candidate = current_line + " " + word
        w = draw.textlength(candidate, font=font)
        if w <= max_width:
            current_line = candidate
        else:
            lines.append(current_line)
            current_line = word

    lines.append(current_line)
    return lines


def fit_text(
    text: str,
    box_width: int,
    box_height: int,
    font_path: str,
    max_font_size: int = 24,
    min_font_size: int = 8,
    padding: int = 6,
) -> FitResult:
    """
    Find the largest font size at which `text` fits within the given box dimensions.
    Falls back to min_font_size if nothing fits; marks FitResult.fits=False in that case.
    """
    # Off-screen draw surface for text measurement
    dummy = Image.new("RGB", (1, 1))
    draw = ImageDraw.Draw(dummy)

    usable_w = max(1, box_width - 2 * padding)
    usable_h = max(1, box_height - 2 * padding)

    for size in range(max_font_size, min_font_size - 1, -1):
        font = ImageFont.truetype(font_path, size)
        lines = _wrap_text(text, font, usable_w, draw)
        if not lines:
            continue

        line_heights = [draw.textbbox((0, 0), line, font=font)[3] for line in lines]
        line_spacing = int(size * 0.2)  # 20% of font size as line gap
        total_h = sum(line_heights) + line_spacing * (len(lines) - 1)
        total_w = max(draw.textlength(line, font=font) for line in lines)

        if total_h <= usable_h:
            return FitResult(
                lines=lines,
                font_size=size,
                total_width=int(total_w),
                total_height=int(total_h),
                fits=True,
            )

    # Nothing fit — use minimum size and accept overflow
    font = ImageFont.truetype(font_path, min_font_size)
    lines = _wrap_text(text, font, usable_w, draw)
    line_heights = [draw.textbbox((0, 0), line, font=font)[3] for line in lines]
    total_h = sum(line_heights) + int(min_font_size * 0.2) * (len(lines) - 1)
    total_w = max((draw.textlength(line, font=font) for line in lines), default=0)
    return FitResult(
        lines=lines,
        font_size=min_font_size,
        total_width=int(total_w),
        total_height=int(total_h),
        fits=False,
    )
```

## Edge Cases
- Single very long word with no spaces: `_wrap_text` will not break it (word wrap only). It will overflow horizontally. This is acceptable and mirrors how desktop DTP tools handle it. Log a warning in the renderer if `fits=False`.
- Empty string: `_wrap_text` returns `[]`. `fit_text` should return a no-op `FitResult` — add a guard at the top of `fit_text` for empty text.
- Box smaller than the minimum font: `FitResult.fits=False`. The renderer still draws the text at min size; it may clip.

## Acceptance Criteria
- `fit_text("Hello world", 200, 50, font_path)` returns `FitResult.fits=True` with lines fitting within the box
- `fit_text("A " * 100, 100, 30, font_path)` returns `FitResult.fits=False` (truly unfit text uses min size)
- `fit_text("", 200, 50, font_path)` does not raise
- Returned `FitResult.lines` joined with spaces reconstructs the original text

## Dependencies
- TICKET-001 (project skeleton)
- TICKET-016 (font .ttf file must exist before this can be tested)

## Estimated Effort
3 hours
