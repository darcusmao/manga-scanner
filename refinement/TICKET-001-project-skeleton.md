# TICKET-001: Project Skeleton and Package Configuration

## Summary
Establish the canonical directory layout, Python version pin, and pyproject.toml scaffold that every subsequent ticket builds on. Nothing is executable after this ticket — it is purely structural.

## Language and Tools
- Python 3.11 (pin via `.python-version` file)
- `uv` as the package manager (faster than pip, lockfile-native, no separate virtualenv step)
- `hatchling` as the build backend

## Directory Structure

```
manga-scanner/
├── pyproject.toml
├── .python-version          # contains: 3.11
├── config.yaml              # runtime config, populated in TICKET-003
├── characters.json          # character profile data, schema in TICKET-013
├── fonts/                   # .ttf files, populated in TICKET-016
├── models/                  # README noting weight cache locations
├── data/
│   ├── input/               # source chapter directories go here
│   └── output/              # localized output mirrors input structure
├── scripts/
│   ├── check_hardware.py    # TICKET-004
│   └── test_inpaint.py      # TICKET-008
├── src/
│   └── manga_scanner/
│       ├── __init__.py
│       ├── types.py          # TICKET-002
│       ├── config.py         # TICKET-003
│       ├── output.py         # TICKET-019
│       ├── vram.py           # TICKET-020
│       ├── cli.py            # TICKET-023
│       ├── detection/
│       │   ├── __init__.py
│       │   ├── detector.py   # TICKET-005
│       │   ├── masker.py     # TICKET-007
│       │   └── sorter.py     # TICKET-006
│       ├── inpainting/
│       │   ├── __init__.py
│       │   └── inpainter.py  # TICKET-009
│       ├── ocr/
│       │   ├── __init__.py
│       │   ├── ocr.py        # TICKET-010
│       │   └── cropper.py    # TICKET-011
│       ├── translation/
│       │   ├── __init__.py
│       │   ├── models.py     # TICKET-013
│       │   ├── prompt_builder.py  # TICKET-014
│       │   └── translator.py # TICKET-015
│       ├── typesetting/
│       │   ├── __init__.py
│       │   ├── fitter.py     # TICKET-017
│       │   └── renderer.py   # TICKET-018
│       └── pipeline/
│           ├── __init__.py
│           ├── orchestrator.py  # TICKET-021
│           └── batch.py         # TICKET-022
├── tests/
│   ├── fixtures/
│   │   └── pages/           # 2-3 sample images for integration tests
│   ├── test_ocr.py
│   ├── test_typesetting.py
│   └── test_pipeline.py
└── .gitignore
```

## Implementation Steps

1. Install uv: `curl -LsSf https://astral.sh/uv/install.sh | sh`
2. Create `.python-version` with contents `3.11`
3. Run `uv init --name manga-scanner --no-readme` to bootstrap, then replace the generated pyproject.toml with the one below
4. Create all directories (empty `__init__.py` files in each package)
5. Create a `.gitignore` that excludes: `.venv/`, `__pycache__/`, `*.pyc`, `data/`, `models/`, `fonts/*.ttf`, `*.gguf`

### pyproject.toml

```toml
[project]
name = "manga-scanner"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = []

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/manga_scanner"]

[project.scripts]
manga-scan = "manga_scanner.cli:app"

[tool.pytest.ini_options]
testpaths = ["tests"]
```

Dependencies are left empty here. Each subsequent ticket adds its own packages via `uv add`.

## Acceptance Criteria
- `uv run python -c "import manga_scanner"` exits with code 0
- All directories listed above exist
- pyproject.toml is valid: `uv sync` completes without error

## Dependencies
None — this is the root ticket.

## Estimated Effort
1 hour
