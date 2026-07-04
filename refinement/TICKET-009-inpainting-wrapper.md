# TICKET-009: Inpainting Wrapper

## Summary
Write the `Inpainter` class that exposes a clean `inpaint(image, mask) -> np.ndarray` interface over iopaint/LaMa. This wrapper isolates all iopaint-specific API details from the rest of the pipeline. It also implements `unload()` to release VRAM after chapter processing is complete.

## Language and Tools
- Python 3.11
- `iopaint` (installed in TICKET-008)
- `numpy`, `Pillow` (already installed)

## Important Precondition
TICKET-008 must be completed first and must have recorded which iopaint access path works: Python API or subprocess CLI. The implementation of this wrapper depends on that outcome.

## Implementation — Python API Path (preferred)

File: `src/manga_scanner/inpainting/inpainter.py`

```python
import gc
import logging
import numpy as np
from PIL import Image
from manga_scanner.config import InpaintingConfig

logger = logging.getLogger(__name__)


class Inpainter:
    def __init__(self, config: InpaintingConfig):
        logger.info("Loading LaMa inpainting model (device=%s)...", config.device)
        from iopaint.model_manager import ModelManager
        from iopaint.schema import InpaintRequest, HDStrategy
        self._ModelManager = ModelManager
        self._InpaintRequest = InpaintRequest
        self._HDStrategy = HDStrategy
        self.model = ModelManager(name=config.model_name, device=config.device)
        self.device = config.device
        logger.info("LaMa model loaded.")

    def inpaint(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        image: HxWx3 uint8 RGB
        mask:  HxW uint8 (0=preserve, 255=inpaint)
        returns: HxWx3 uint8 RGB with masked regions inpainted
        """
        pil_image = Image.fromarray(image)
        pil_mask = Image.fromarray(mask)
        req = self._InpaintRequest(hd_strategy=self._HDStrategy.ORIGINAL)
        result = self.model(pil_image, pil_mask, req)
        return np.array(result)

    def unload(self) -> None:
        logger.info("Unloading LaMa model from VRAM.")
        del self.model
        import torch
        torch.cuda.empty_cache()
        gc.collect()
```

## Alternative — Subprocess CLI Path
If TICKET-008 found the Python API to be broken, replace the `inpaint()` method body with:

```python
def inpaint(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    import subprocess, tempfile
    with tempfile.TemporaryDirectory() as tmp:
        from pathlib import Path
        img_path = Path(tmp) / "img.png"
        mask_path = Path(tmp) / "mask.png"
        out_path = Path(tmp) / "out.png"
        Image.fromarray(image).save(img_path)
        Image.fromarray(mask).save(mask_path)
        result = subprocess.run(
            ["iopaint", "run", "--model=lama", f"--device={self.device}",
             f"--image={img_path}", f"--mask={mask_path}", f"--output={out_path}"],
            capture_output=True, check=True
        )
        return np.array(Image.open(out_path).convert("RGB"))
```

Note: the subprocess path is slower (process startup overhead per page) and cannot be unloaded from VRAM between calls since it's a new process each time. Prefer the Python API path.

## Error Handling
Wrap `inpaint()` call in the orchestrator (TICKET-021) with a try/except. If inpainting fails on a specific page, the orchestrator falls back to using the original image canvas (without erasure) and logs a WARNING. This is a degraded but non-crashing output.

## Acceptance Criteria
- `Inpainter(config).inpaint(image, mask)` returns an ndarray of the same shape as `image`
- Output dtype is uint8
- Calling `unload()` after use does not raise; subsequent calls to `inpaint()` should raise (model is deleted) rather than silently fail

## Dependencies
- TICKET-001 (project skeleton)
- TICKET-002 (types for type annotations)
- TICKET-003 (InpaintingConfig)
- TICKET-008 (iopaint verified and API path determined)

## Estimated Effort
2 hours
