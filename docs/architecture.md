# Architecture & Pipeline Flow

## Pipeline Overview

Every chapter run passes all pages through each stage before loading the next model.
This staged design means heavy models (LaMa, Qwen) are never in VRAM at the same time.

```mermaid
flowchart TD
    IMG[/"📚 Input — Manga Page Images\n.png  .jpg  .webp"/]
    CONFIG[/"⚙️  config.yaml  +  characters.json"/]
    CLI["CLI\nmanga-scan chapter  /  manga-scan page"]

    subgraph A["Stage A — Detection  ·  YOLOv8  ·  ~300 MB VRAM"]
        DETECT["TextDetector\nruns YOLOv8 on each page\nreturns BoundingBox list"]
        SORT["sort_reading_order\nright → left within each row\ntop → bottom across rows"]
        DETECT --> SORT
    end

    subgraph B["Stage B — Inpainting  ·  LaMa  ·  ~2.5 GB VRAM"]
        MASK["generate_mask\nexpands each box by box_padding px\nproduces binary uint8 mask"]
        INPAINT["Inpainter\niopaint / LaMa fills masked region\nscreentones and linework preserved"]
        MASK --> INPAINT
    end

    NOTE["⚠️  crops taken from ORIGINAL image\nbefore text is erased —\ninpainting removes the text OCR needs"]

    subgraph C["Stage C — OCR  ·  manga-ocr  ·  ~500 MB VRAM"]
        CROP["extract_crops\nslices each bubble region\nfrom the original image array"]
        OCR["MangaOCR\nmanga-ocr-base vision transformer\nreturns Japanese string per crop"]
        CROP --> OCR
    end

    subgraph D["Stage D — Translation  ·  Qwen2.5-7B  ·  ~5 GB VRAM"]
        PROMPT["build_prompt\nnumbers each JP string\nappends character profile block\n(register, pronouns, relationships)"]
        TRANS["Translator\nhttpx → Ollama REST API\nretries on timeout\nfallback to original JP on failure"]
        PROMPT --> TRANS
    end

    subgraph E["Stage E — Typesetting  ·  CPU only  ·  Pillow"]
        FIT["fit_text\nlinear scan max → min font size\ngreedy word-wrap per size"]
        RENDER["render_translations\n1 px white outline\ncentered text over clean canvas"]
        FIT --> RENDER
    end

    OUT[/"🖼️  Localized PNG\nsame dimensions as input"/]

    VRAM["managed_model\nwraps each model stage\ncalls unload + clear_cuda_cache\non exit — even on error"]

    IMG --> CLI
    CONFIG --> CLI
    CLI --> A

    A -->|"ordered BoundingBox list\n+ original image array"| B
    A -->|"original image array"| NOTE
    NOTE -->|"crops from original"| C
    B -->|"clean canvas\n(text erased)"| C
    C -->|"JP text strings\n+ clean canvas\n+ BoundingBox list"| D
    D -->|"EN text strings\naligned 1-to-1 with boxes"| E
    E --> OUT

    VRAM -.->|"loads / unloads"| A
    VRAM -.->|"loads / unloads"| B
    VRAM -.->|"loads / unloads"| C

    style NOTE fill:#fff8dc,stroke:#d4a017,color:#333
    style VRAM fill:#e8f4f8,stroke:#4a90a4,color:#333
```

---

## VRAM Budget (8 GB GPU)

Stages run sequentially so the two largest models never overlap.

```mermaid
gantt
    title VRAM occupancy across chapter processing
    dateFormat  X
    axisFormat  Stage %s

    section Models
    YOLOv8 — 300 MB      : active, a, 0, 1
    LaMa — 2.5 GB        : active, b, 1, 2
    manga-ocr — 500 MB   : active, c, 2, 3
    Qwen2.5-7B — 5.0 GB  : active, d, 3, 4
    Pillow (CPU)         : active, e, 4, 5
```

---

## Module Map

```mermaid
graph TB
    subgraph entry["Entry points"]
        CLI_M["cli.py\nclick group\nchapter · page commands"]
    end

    subgraph config_g["Configuration"]
        CONFIG_M["config.py\nConfig — pydantic-settings\nload_config — YAML + env vars"]
        TYPES_M["types.py\nBoundingBox  DetectionResult\nCropResult  OCRResult\nTranslationResult  PageJob"]
    end

    subgraph pipeline_g["pipeline/"]
        BATCH_M["batch.py\nprocess_chapter\ncollect_pages\n_dump_intermediates"]
        ORCH_M["orchestrator.py\nprocess_page\nsingle-page flow"]
    end

    subgraph detection_g["detection/"]
        DET_M["detector.py\nTextDetector\nYOLOv8 wrapper"]
        MASK_M["masker.py\nload_image\ngenerate_mask\nBubbleMasker"]
        SORT_M["sorter.py\nsort_reading_order"]
    end

    subgraph inpaint_g["inpainting/"]
        INP_M["inpainter.py\nInpainter\niopaint Python API"]
    end

    subgraph ocr_g["ocr/"]
        OCR_M["ocr.py\nMangaOCR\ntranscribe · transcribe_batch"]
        CROP_M["cropper.py\nextract_crops"]
    end

    subgraph translation_g["translation/"]
        MODELS_M["models.py\nCharacterProfile\nSpeechRegister\nload_character_profiles"]
        PROMPT_M["prompt_builder.py\nbuild_prompt\nSYSTEM_PROMPT"]
        TRANS_M["translator.py\nTranslator\nhttpx · retry · fallback"]
    end

    subgraph typeset_g["typesetting/"]
        FIT_M["fitter.py\nfit_text\nFitResult"]
        REND_M["renderer.py\nrender_translations\n_draw_outlined_text"]
    end

    subgraph infra_g["Infrastructure"]
        OUT_M["output.py\nresolve_output_path\nOutputResolution"]
        VRAM_M["vram.py\nmanaged_model\nclear_cuda_cache\nlog_vram"]
    end

    CLI_M --> BATCH_M
    CLI_M --> CONFIG_M

    BATCH_M --> DET_M
    BATCH_M --> INP_M
    BATCH_M --> OCR_M
    BATCH_M --> TRANS_M
    BATCH_M --> REND_M
    BATCH_M --> VRAM_M
    BATCH_M --> OUT_M
    BATCH_M --> MODELS_M

    DET_M --> TYPES_M
    MASK_M --> TYPES_M
    SORT_M --> TYPES_M
    CROP_M --> TYPES_M
    OCR_M --> TYPES_M
    TRANS_M --> TYPES_M
    TRANS_M --> PROMPT_M
    PROMPT_M --> MODELS_M
    REND_M --> FIT_M
```

---

## Data Flow Between Stages

What each stage consumes and produces, per page:

| Stage | Consumes | Produces |
|---|---|---|
| **A — Detection** | image path | `DetectionResult` (ordered `BoundingBox` list) |
| **B — Inpainting** | original image array + box list → mask | clean canvas `np.ndarray` (text erased) |
| **C — OCR** | crops from **original** array (not clean canvas) | `OCRResult` list (JP string per box) |
| **D — Translation** | JP string list + character profiles | `TranslationResult` (EN string list, 1-to-1) |
| **E — Typesetting** | EN strings + boxes + clean canvas | rendered `Image` saved as PNG |

The key invariant: **OCR always reads from the original image.** The inpainted canvas has the text erased, so reversing the stage order would produce empty OCR results.
