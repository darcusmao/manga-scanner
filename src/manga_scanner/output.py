from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class OutputResolution:
    path: Path
    skip: bool


def resolve_output_path(
    input_path: Path,
    input_root: Path,
    output_root: Path,
    skip_existing: bool = True,
) -> OutputResolution:
    """
    Mirror the input path structure under output_root.

    Example:
      input_path  = data/input/chapter_01/page_003.png
      input_root  = data/input/chapter_01
      output_root = data/output
      result      = data/output/chapter_01/page_003.png
    """
    try:
        relative = input_path.relative_to(input_root)
        output_path = output_root / input_root.name / relative
    except ValueError:
        output_path = output_root / input_path.name

    output_path = output_path.with_suffix(".png")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    return OutputResolution(
        path=output_path,
        skip=skip_existing and output_path.exists(),
    )
