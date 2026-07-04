# TICKET-023: CLI Entry Point

## Summary
Write the `click`-based CLI that exposes two commands: `page` (single image) and `chapter` (directory batch). This is the public interface through which the user interacts with the pipeline.

## Language and Tools
- Python 3.11
- `click` — CLI framework
- Install: `uv add click`

## Commands

### `manga-scan chapter`
Primary command. Processes a full directory of manga scans.

```bash
manga-scan chapter \
  --input data/input/chapter_01 \
  --output data/output \
  --config config.yaml \
  --characters characters.json
```

### `manga-scan page`
Single-page convenience command. Useful for testing pipeline output on one image.

```bash
manga-scan page \
  --input data/input/chapter_01/page_001.png \
  --output data/output/chapter_01/page_001.png \
  --config config.yaml \
  --characters characters.json
```

## Implementation

File: `src/manga_scanner/cli.py`

```python
import logging
import sys
from pathlib import Path
import click

from manga_scanner.config import load_config
from manga_scanner.pipeline.batch import process_chapter
from manga_scanner.translation.models import load_character_profiles


def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


@click.group()
@click.option("--verbose", "-v", is_flag=True, default=False, help="Enable debug logging.")
@click.pass_context
def app(ctx: click.Context, verbose: bool) -> None:
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose
    configure_logging(verbose)


@app.command()
@click.option("--input", "-i", "input_dir", required=True, type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--output", "-o", "output_root", required=True, type=click.Path(path_type=Path))
@click.option("--config", "-c", "config_path", default="config.yaml", type=click.Path(path_type=Path))
@click.option("--characters", "characters_path", default="characters.json", type=click.Path(path_type=Path))
@click.pass_context
def chapter(
    ctx: click.Context,
    input_dir: Path,
    output_root: Path,
    config_path: Path,
    characters_path: Path,
) -> None:
    """Process a directory of manga pages as a single chapter."""
    config = load_config(config_path)
    process_chapter(input_dir, output_root, config, characters_path)


@app.command()
@click.option("--input", "-i", "input_path", required=True, type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--output", "-o", "output_path", required=True, type=click.Path(path_type=Path))
@click.option("--config", "-c", "config_path", default="config.yaml", type=click.Path(path_type=Path))
@click.option("--characters", "characters_path", default="characters.json", type=click.Path(path_type=Path))
@click.pass_context
def page(
    ctx: click.Context,
    input_path: Path,
    output_path: Path,
    config_path: Path,
    characters_path: Path,
) -> None:
    """Process a single manga page image."""
    # Reuse process_chapter by treating the file's parent as a one-file chapter directory
    config = load_config(config_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # For single page, wrap in a minimal batch call targeting only this file
    process_chapter(
        input_dir=input_path.parent,
        output_root=output_path.parent.parent,
        config=config,
        characters_path=characters_path,
    )
```

Note: the `page` command reuses `process_chapter` by targeting the parent directory. This is simpler than duplicating the model lifecycle logic. If the chapter directory contains multiple files, only the one matching the input filename will be processed — to enforce this cleanly, add a `--only` filter to `collect_pages()` in TICKET-022, or accept that a single-file directory is the cleanest solution.

## Entry Point Registration

In `pyproject.toml`:
```toml
[project.scripts]
manga-scan = "manga_scanner.cli:app"
```

After `uv sync`, the `manga-scan` binary is available:
```bash
uv run manga-scan --help
uv run manga-scan chapter --help
```

## Acceptance Criteria
- `uv run manga-scan --help` prints the command group help
- `uv run manga-scan chapter --help` prints chapter-specific options
- `manga-scan chapter --input <missing_dir>` prints a click error and exits with code 2 (click's standard for bad arguments)
- `--verbose` flag causes DEBUG-level logs to appear
- Both `config.yaml` and `characters.json` default gracefully to an empty config/empty profiles when not present

## Dependencies
- TICKET-001 (entry point in pyproject.toml)
- TICKET-003 (load_config)
- TICKET-013 (load_character_profiles)
- TICKET-022 (process_chapter)

## Estimated Effort
2 hours
