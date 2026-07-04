# Project Scan-to-Scan: End-to-End Local Manga Translation Pipeline

An automated, high-performance Computer Vision (CV) and Natural Language Processing (NLP) localization engine designed to ingest raw Japanese manga scans and output fully typeset, contextually accurate English localized pages. This repository orchestrates text detection, deep learning-based image inpainting, specialized optical character recognition (OCR), and quantized local Large Language Models (LLMs) to achieve a seamless, localized comic experience entirely on local consumer hardware.

Zero marginal cost per page. All inference runs locally after one-time model downloads.

---

## Project Scope & Constraints

### In-Scope

* **Headless Orchestration Pipeline:** A Python-based staged batch architecture that processes all pages per stage rather than all stages per page, minimizing model reload overhead across a chapter.
* **Speech Bubble Detection & Masking:** Automatic identification of speech bubbles using YOLOv8 (ultralytics), with padded binary mask generation for clean inpainting boundaries.
* **Neural Art Inpainting:** Seamless removal of original Japanese text using iopaint/LaMa (Large Mask inpainting with Fourier Convolutions), retaining underlying screentone gradients and line art.
* **Domain-Specific Comic OCR:** High-accuracy transcription of vertical and handwritten Japanese comic text using manga-ocr (kha-white/manga-ocr-base vision transformer).
* **Contextual Page-Level Localization:** Batch translation using a locally quantized Qwen2.5-7B-Instruct model served via Ollama, fed structured character profiles to preserve speech registers and pronouns.
* **Dynamic Typesetting Canvas Engine:** A font-size binary-search and greedy word-wrap layout engine built in Pillow that fits translated text into arbitrary speech bubble bounds.

### Out-of-Scope

* **Real-time Video Translation:** Targets static image assets (PNG/JPEG) only.
* **Dynamic Web Dashboard / Frontend:** Runs entirely via a local CLI. No React, no cloud deployment.
* **Universal SFX Redrawing:** Targets narrative speech bubbles. Complex full-page sound effects layered over art are masked globally rather than artistically reconstructed.

---

## Project Goals

1. **Contextual Preservation:** Localization that infers dropped pronouns and distinct speech registers by feeding the LLM full-page dialogue arrays alongside structured character profiles rather than translating bubble-by-bubble.
2. **Artifact-Free Inpainting:** Strip Japanese text while leaving screentones and linework intact. No white bounding box overlays.
3. **High-Throughput Batch Processing:** Process an entire chapter directory without manual intervention. A single CLI command handles detection through final typeset output.

---

## Architecture

```
        +-----------------------------------------------------------+
        |                    Raw Japanese Image                     |
        +-----------------------------------------------------------+
               |                                             |
               v (image array)                               v (image array)
      +------------------+                          +------------------+
      |  YOLOv8          |                          |  iopaint / LaMa  |
      | (Bubble Detect)  |                          | (Text Inpainting)|
      +------------------+                          +------------------+
               |                                             |
               v (BoundingBox list, reading order)           |
      +------------------+                                   |
      |  manga-ocr       |  <-- crops from ORIGINAL image    |
      | (Text Extraction)|      (not the inpainted canvas)   |
      +------------------+                                   |
               |                                             |
               v (JP string array)                           v (clean canvas)
      +------------------+                          +------------------+
      |  Ollama          |                          |  Pillow          |
      |  Qwen2.5-7B      |                          |  Typesetting     |
      | (Translation)    |                          |  Engine          |
      +------------------+                          +------------------+
               |                                             ^
               +------> (EN string array, same order) -------+
                                                             |
                                                             v
                                                    +------------------+
                                                    | Localized Output |
                                                    |     PNG          |
                                                    +------------------+
```

### Staged Batch Execution Order (per chapter)

Processing runs all pages through each stage before moving to the next. This avoids reloading models per page and allows VRAM to be released cleanly between stages.

```
Stage A  YOLOv8 detection       all pages  (~300 MB VRAM)
Stage B  LaMa inpainting        all pages  (~2.5 GB VRAM)  -> unload
Stage C  manga-ocr transcription all pages (~500 MB VRAM)  -> unload
Stage D  Ollama/Qwen translation all pages (~5.0 GB VRAM, separate process)
Stage E  Pillow typesetting      all pages  (CPU only)
```

---

## Technology Stack

| Component          | Library / Tool                          | Notes                                        |
|--------------------|-----------------------------------------|----------------------------------------------|
| Language           | Python 3.11                             |                                              |
| Package manager    | uv                                      | Faster than pip, lockfile-native             |
| Text detection     | ultralytics (YOLOv8)                    | Fine-tuned manga bubble weights preferred    |
| Inpainting         | iopaint (LaMa model)                    | ~500 MB weights, cached at ~/.cache/iopaint/ |
| OCR                | manga-ocr                               | ~420 MB weights from HuggingFace             |
| LLM runtime        | Ollama                                  | Local daemon, port 11434                     |
| LLM model          | qwen2.5:7b-instruct-q4_K_M              | ~4.7 GB download, ~5 GB VRAM at inference    |
| Image processing   | Pillow, numpy                           |                                              |
| HTTP client        | httpx                                   | Ollama REST API calls                        |
| Config             | pydantic-settings + pyyaml              | config.yaml with env var overrides           |
| CLI                | click                                   | manga-scan chapter / manga-scan page         |
| Testing            | pytest + pytest-mock                    |                                              |

---

## Hardware Requirements

- GPU: NVIDIA GPU with CUDA support (8 GB VRAM minimum, 10+ GB recommended)
- RAM: 16 GB minimum (staged intermediates held in memory during chapter processing)
- Disk: ~15 GB for all model weights combined
- OS: macOS or Linux (Windows untested)

On an 8 GB GPU: the staged execution order is mandatory. LaMa (~2.5 GB) and Qwen (~5 GB) cannot coexist in VRAM and are never loaded simultaneously.

---

## Quick Start

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and install
git clone <repo>
cd manga-scanner
uv sync

# Install Ollama and pull the translation model
brew install ollama          # macOS; see ollama.com for Linux
ollama serve &
ollama pull qwen2.5:7b-instruct-q4_K_M

# Verify GPU is visible
uv run python scripts/check_hardware.py

# Verify iopaint works (downloads LaMa weights on first run, ~500 MB)
uv run python scripts/test_inpaint.py

# Run on a chapter directory
uv run manga-scan chapter \
  --input  data/input/chapter_01 \
  --output data/output \
  --config config.yaml \
  --characters characters.json
```

---

## Project Structure

```
manga-scanner/
├── pyproject.toml
├── config.yaml              # pipeline tuning parameters
├── characters.json          # per-series character profiles
├── fonts/                   # .ttf comic lettering fonts
├── models/                  # model weight notes
├── data/
│   ├── input/               # source chapter directories
│   └── output/              # localized output (mirrors input structure)
├── scripts/
│   ├── check_hardware.py    # CUDA verification
│   └── test_inpaint.py      # iopaint API verification
├── src/manga_scanner/
│   ├── types.py             # shared dataclasses (BoundingBox, PageJob, etc.)
│   ├── config.py            # Pydantic config system
│   ├── output.py            # output path resolution
│   ├── vram.py              # VRAM lifecycle management
│   ├── cli.py               # click entry point
│   ├── detection/           # YOLOv8 detector, mask generator, reading order sorter
│   ├── inpainting/          # iopaint/LaMa wrapper
│   ├── ocr/                 # manga-ocr wrapper, crop extractor
│   ├── translation/         # character profiles, prompt builder, Ollama client
│   ├── typesetting/         # text fitter, overlay renderer
│   └── pipeline/            # single-page orchestrator, batch chapter processor
└── tests/
    ├── fixtures/pages/      # synthetic test images
    ├── test_ocr.py
    ├── test_typesetting.py
    └── test_pipeline.py     # full end-to-end with mocked models
```

---

## Configuration

All tunable parameters live in `config.yaml`. Key entries:

```yaml
detection:
  model_path: "models/manga_bubble_detector.pt"
  confidence_threshold: 0.45
  box_padding: 8              # pixels to expand mask beyond detected bbox

translation:
  model_name: "qwen2.5:7b-instruct-q4_K_M"
  temperature: 0.2            # lower = more literal translation
  max_retries: 2
  timeout_seconds: 90

typesetting:
  font_path: "fonts/anime_ace_2.ttf"
  max_font_size: 24
  min_font_size: 8

pipeline:
  skip_existing: true         # resume interrupted chapter runs
```

Override any value with environment variables using the `MANGA_` prefix:
```bash
MANGA_TRANSLATION__TEMPERATURE=0.4 uv run manga-scan chapter ...
```

---

## Character Profiles

Create `characters.json` to preserve speech register and personality across a chapter. Optional — the pipeline runs without it but translations will be generic.

```json
{
  "characters": [
    {
      "name": "Kira",
      "jp_name": "キラ",
      "speech_register": "formal",
      "pronouns": "he/him",
      "speech_notes": "Deliberate, long sentences. Never uses contractions. Occasionally refers to himself by name.",
      "relationships": { "L": "nemesis", "Ryuk": "tool" }
    }
  ]
}
```

Valid speech_register values: formal, casual, rough, archaic, childlike

---

## Implementation Tickets

Granular implementation tickets are in `refinement/`. Each ticket specifies exact packages, install commands, code structure, and acceptance criteria.

| Ticket | Title |
|--------|-------|
| 001 | Project Skeleton and Package Configuration |
| 002 | Shared Data Types and Interfaces |
| 003 | Configuration System |
| 004 | PyTorch and CUDA Hardware Verification |
| 005 | YOLOv8 Text Detection Module |
| 006 | Manga Reading Order Sorting |
| 007 | Binary Mask Generation |
| 008 | iopaint Installation and API Verification |
| 009 | Inpainting Wrapper |
| 010 | manga-ocr Installation and OCR Wrapper |
| 011 | Crop Extraction Utility |
| 012 | Ollama Installation and Qwen2.5-7B Model Pull |
| 013 | Character Profile JSON Schema and Pydantic Model |
| 014 | LLM Prompt Builder |
| 015 | Translation Wrapper, Response Parser, and Retry Logic |
| 016 | Font Selection and Packaging |
| 017 | Text Fitting and Multi-line Wrapping Algorithm |
| 018 | Overlay Renderer |
| 019 | Output Path Management |
| 020 | VRAM Lifecycle Management |
| 021 | Single-Page Pipeline Orchestrator |
| 022 | Batch Chapter Directory Processor |
| 023 | CLI Entry Point |
| 024 | Unit Tests |
| 025 | End-to-End Integration Test |
