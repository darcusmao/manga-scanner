from __future__ import annotations

import logging
from pathlib import Path

import click

from manga_scanner.config import load_config
from manga_scanner.pipeline.batch import process_chapter


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
@click.option(
    "--input", "-i", "input_dir",
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Directory of manga page images.",
)
@click.option(
    "--output", "-o", "output_root",
    required=True,
    type=click.Path(path_type=Path),
    help="Root directory for output files.",
)
@click.option("--config", "-c", "config_path", default="config.yaml", type=click.Path(path_type=Path))
@click.option("--characters", "characters_path", default="characters.json", type=click.Path(path_type=Path))
@click.option(
    "--dump-dir", "dump_dir",
    default=None,
    type=click.Path(path_type=Path),
    help="Save per-page detection + OCR JSON for benchmarking.",
)
@click.pass_context
def chapter(
    ctx: click.Context,
    input_dir: Path,
    output_root: Path,
    config_path: Path,
    characters_path: Path,
    dump_dir: Path | None,
) -> None:
    """Process a directory of manga pages as a single chapter."""
    config = load_config(config_path)
    process_chapter(input_dir, output_root, config, characters_path, dump_dir=dump_dir)


@app.command()
@click.option(
    "--input", "-i", "input_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Single manga page image.",
)
@click.option(
    "--output", "-o", "output_path",
    required=True,
    type=click.Path(path_type=Path),
    help="Output file path.",
)
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
    config = load_config(config_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    process_chapter(
        input_dir=input_path.parent,
        output_root=output_path.parent.parent,
        config=config,
        characters_path=characters_path,
    )
