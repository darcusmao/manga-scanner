from __future__ import annotations

import gc
import logging
from pathlib import Path

from manga_scanner.config import DetectionConfig
from manga_scanner.types import BoundingBox, DetectionResult

logger = logging.getLogger(__name__)


class TextDetector:
    def __init__(self, config: DetectionConfig) -> None:
        from ultralytics import YOLO
        logger.info("Loading detection model: %s", config.model_path)
        self.model = YOLO(config.model_path)
        self.threshold = config.confidence_threshold
        self.device = config.device

    def detect(self, image_path: Path) -> DetectionResult:
        results = self.model(
            str(image_path),
            device=self.device,
            conf=self.threshold,
            verbose=False,
        )
        boxes: list[BoundingBox] = []
        for box in results[0].boxes:
            xyxy = box.xyxy[0].cpu().numpy()
            boxes.append(
                BoundingBox(
                    x1=int(xyxy[0]),
                    y1=int(xyxy[1]),
                    x2=int(xyxy[2]),
                    y2=int(xyxy[3]),
                    confidence=float(box.conf[0]),
                    label=self.model.names[int(box.cls[0])],
                )
            )
        if not boxes:
            logger.warning("No text regions detected in %s", image_path.name)
        return DetectionResult(image_path=image_path, boxes=boxes)

    def unload(self) -> None:
        logger.info("Unloading detection model from VRAM.")
        del self.model
        import torch
        torch.cuda.empty_cache()
        gc.collect()
