from __future__ import annotations

import gc
import logging
from contextlib import contextmanager
from typing import TypeVar, Callable

logger = logging.getLogger(__name__)

T = TypeVar("T")


def clear_cuda_cache() -> None:
    """Release all PyTorch VRAM allocations and run the GC."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except (ImportError, RuntimeError, AssertionError):
        pass
    gc.collect()


def log_vram(label: str = "") -> None:
    """Log current VRAM usage. No-op if CUDA is unavailable."""
    try:
        import torch
        if not torch.cuda.is_available():
            return
        alloc = torch.cuda.memory_allocated() / 1024 ** 3
        reserved = torch.cuda.memory_reserved() / 1024 ** 3
        logger.info(
            "VRAM [%s]: allocated=%.2f GB  reserved=%.2f GB",
            label, alloc, reserved,
        )
    except ImportError:
        pass


@contextmanager
def managed_model(factory: Callable[[], T]):
    """
    Construct a model, yield it, then call model.unload() and clear the
    CUDA cache on exit — even if an exception is raised inside the block.

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
