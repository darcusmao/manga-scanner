from __future__ import annotations

import gc
import logging

import numpy as np
from PIL import Image

from manga_scanner.config import InpaintingConfig

logger = logging.getLogger(__name__)


class Inpainter:
    def __init__(self, config: InpaintingConfig) -> None:
        logger.info("Loading LaMa inpainting model (device=%s)...", config.device)
        from iopaint.model_manager import ModelManager
        from iopaint.schema import InpaintRequest, HDStrategy
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
        req = self._InpaintRequest(hd_strategy=self._HDStrategy.ORIGINAL)
        # ModelManager expects numpy arrays, not PIL Images
        result = self.model(image, mask, req)
        return result.astype(np.uint8)

    def unload(self) -> None:
        logger.info("Unloading LaMa model from VRAM.")
        del self.model
        import torch
        torch.cuda.empty_cache()
        gc.collect()
