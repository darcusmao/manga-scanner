# TICKET-020: VRAM Lifecycle Management

## Summary
Write the VRAM management utilities that enforce sequential model load/unload sequencing across the pipeline. On a consumer GPU with 8-12GB VRAM, all four inference components (YOLOv8, LaMa, manga-ocr, Qwen2.5-7B via Ollama) cannot coexist in VRAM simultaneously. This module provides the context manager and cache-clearing utilities used by the orchestrator.

## Language and Tools
- Python 3.11 standard library: `gc`, `contextlib`
- `torch` (already installed)

## VRAM Budget Per Component

| Component | Library | Approx. VRAM | Notes |
|---|---|---|---|
| YOLOv8n | ultralytics | ~300 MB | Stays loaded for full chapter |
| LaMa | iopaint | ~2.5 GB | Load per chapter, unload before OCR |
| manga-ocr | manga-ocr | ~500 MB | Load per chapter, unload before LLM |
| Qwen2.5-7B Q4_K_M | Ollama (separate process) | ~5.0 GB | Managed by Ollama daemon |

Total if all loaded simultaneously: ~8.3 GB — exceeds 8 GB cards, marginal on 10 GB cards.

## Safe Sequencing Order Per Chapter

```
1. Load TextDetector (YOLOv8)       → process all pages (detect only)
2. Load Inpainter (LaMa)            → process all pages (inpaint only)
3. Unload Inpainter                  → clear VRAM
4. Load MangaOCR                     → process all pages (OCR all crops)
5. Unload MangaOCR                   → clear VRAM
6. Call Translator (Ollama)          → process all pages (LLM calls)
   (Ollama manages its own VRAM; it evicts after idle timeout)
7. Typeset all pages                  → CPU only (Pillow), no VRAM needed
8. Unload TextDetector               → clean up
```

This "all pages per stage" approach (rather than all stages per page) minimizes model load/unload overhead. The tradeoff is that intermediate results (inpainted images, OCR results) must be held in RAM or written to disk between stages.

If RAM is constrained (typical: 16-32 GB), write intermediate numpy arrays to the scratchpad directory between stages and reload them. If RAM is ample, hold them in Python lists.

## Implementation

File: `src/manga_scanner/vram.py`

```python
import gc
import logging
from contextlib import contextmanager
from typing import TypeVar, Callable, Any

logger = logging.getLogger(__name__)

T = TypeVar("T")


def clear_cuda_cache() -> None:
    """Release all PyTorch VRAM allocations and run the GC."""
    try:
        import torch
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    except ImportError:
        pass
    gc.collect()


def log_vram(label: str = "") -> None:
    """Log current VRAM usage. No-op if CUDA is unavailable."""
    try:
        import torch
        if not torch.cuda.is_available():
            return
        alloc = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        logger.info("VRAM [%s]: allocated=%.2f GB  reserved=%.2f GB", label, alloc, reserved)
    except ImportError:
        pass


@contextmanager
def managed_model(factory: Callable[[], T]):
    """
    Context manager that constructs a model, yields it, then calls
    model.unload() and clears the CUDA cache on exit.

    Usage:
        with managed_model(lambda: TextDetector(config)) as detector:
            results = [detector.detect(p) for p in pages]
    """
    log_vram("before load")
    model = factory()
    log_vram("after load")
    try:
        yield model
    finally:
        if hasattr(model, "unload"):
            model.unload()
        clear_cuda_cache()
        log_vram("after unload")
```

## Ollama-Specific Note

Ollama runs as a separate OS process and manages its own VRAM allocation. There is no Python API to force-unload the Ollama model from VRAM. It will automatically evict the model after an idle period (default: 5 minutes, configurable via `OLLAMA_KEEP_ALIVE` environment variable).

To minimize VRAM contention when moving from OCR to LLM translation:
1. Unload manga-ocr (free ~500 MB PyTorch VRAM)
2. Call `clear_cuda_cache()`
3. Wait briefly for Ollama to load the model if it was evicted (first LLM call will be slower)

Set `OLLAMA_KEEP_ALIVE=0` in the environment to make Ollama evict the model immediately after each request. This frees 5GB VRAM for the next stage but adds cold-start latency to every LLM call. Not recommended for chapter processing where multiple pages call the LLM sequentially.

## Acceptance Criteria
- `clear_cuda_cache()` does not raise on a machine without CUDA
- `log_vram()` logs a formatted VRAM line when CUDA is available, is silent otherwise
- `managed_model(lambda: MyModel())` calls `model.unload()` even if an exception is raised inside the `with` block
- VRAM usage reported by `log_vram("after unload")` is lower than `log_vram("after load")` after a real model unload

## Dependencies
- TICKET-001 (project skeleton)
- TICKET-004 (torch installed)

## Estimated Effort
2 hours
