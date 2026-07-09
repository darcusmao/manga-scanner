from __future__ import annotations

from dataclasses import dataclass

from PIL import Image, ImageDraw, ImageFont


@dataclass
class FitResult:
    lines: list[str]
    font_size: int
    total_width: int
    total_height: int
    fits: bool


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

    lines: list[str] = []
    current_line = words[0]

    for word in words[1:]:
        candidate = current_line + " " + word
        if draw.textlength(candidate, font=font) <= max_width:
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
    Find the largest font size at which text fits within the given box dimensions.
    Falls back to min_font_size if nothing fits; marks FitResult.fits=False in that case.
    """
    if not text.strip():
        return FitResult(lines=[], font_size=min_font_size, total_width=0, total_height=0, fits=True)

    dummy = Image.new("RGB", (1, 1))
    draw = ImageDraw.Draw(dummy)

    usable_w = max(1, box_width - 2 * padding)
    usable_h = max(1, box_height - 2 * padding)

    for size in range(max_font_size, min_font_size - 1, -1):
        font = ImageFont.truetype(font_path, size)
        lines = _wrap_text(text, font, usable_w, draw)
        if not lines:
            continue

        line_spacing = int(size * 0.2)
        line_heights = [draw.textbbox((0, 0), line, font=font)[3] for line in lines]
        total_h = sum(line_heights) + line_spacing * max(0, len(lines) - 1)
        total_w = int(max(draw.textlength(line, font=font) for line in lines))

        if total_h <= usable_h:
            return FitResult(
                lines=lines,
                font_size=size,
                total_width=total_w,
                total_height=int(total_h),
                fits=True,
            )

    font = ImageFont.truetype(font_path, min_font_size)
    lines = _wrap_text(text, font, usable_w, draw)
    line_spacing = int(min_font_size * 0.2)
    line_heights = [draw.textbbox((0, 0), line, font=font)[3] for line in lines]
    total_h = sum(line_heights) + line_spacing * max(0, len(lines) - 1)
    total_w = int(max((draw.textlength(line, font=font) for line in lines), default=0))

    return FitResult(
        lines=lines,
        font_size=min_font_size,
        total_width=total_w,
        total_height=int(total_h),
        fits=False,
    )
