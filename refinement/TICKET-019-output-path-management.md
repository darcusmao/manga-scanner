# TICKET-019: Output Path Management

## Summary
Write the utility that resolves where output files are written, mirrors the input directory structure under the configured output root, and checks for existing files when `skip_existing` is enabled.

## Language and Tools
- Python 3.11 standard library only (`pathlib`)

## Why This Needs Its Own Module
Without a consistent output path strategy, the batch processor (TICKET-022) and the single-page orchestrator (TICKET-021) would each implement their own path logic, leading to inconsistencies. Centralizing this also makes the `skip_existing` check a single decision point.

## Implementation

File: `src/manga_scanner/output.py`

```python
from dataclasses import dataclass
from pathlib import Path


@dataclass
class OutputResolution:
    path: Path
    skip: bool    # True if the file already exists and skip_existing is enabled


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

    If input_path is not under input_root, use input_path.name only.
    """
    try:
        relative = input_path.relative_to(input_root)
        output_path = output_root / input_root.name / relative
    except ValueError:
        output_path = output_root / input_path.name

    # Always output as PNG regardless of input format
    output_path = output_path.with_suffix(".png")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    return OutputResolution(
        path=output_path,
        skip=skip_existing and output_path.exists(),
    )
```

## Behavior

- Input `chapter_01/page_003.jpg` → output `data/output/chapter_01/page_003.png`
- Output extension is always `.png` — JPEG would introduce compression artifacts on text
- Parent directories are created automatically with `mkdir(parents=True, exist_ok=True)`
- If `skip_existing=True` and the output file already exists, `OutputResolution.skip=True` — the caller decides what to do (log and skip, or overwrite)

## Acceptance Criteria
- `resolve_output_path(Path("data/input/ch01/p001.jpg"), Path("data/input/ch01"), Path("data/output"))` returns a path ending in `data/output/ch01/p001.png`
- Parent directories are created if they do not exist
- Output extension is always `.png`
- `skip=True` when `skip_existing=True` and the output file already exists
- `skip=False` when `skip_existing=False` regardless of whether file exists

## Dependencies
- TICKET-001 (project skeleton)
- TICKET-003 (PipelineConfig has skip_existing flag, PathsConfig has output_dir)

## Estimated Effort
1 hour
