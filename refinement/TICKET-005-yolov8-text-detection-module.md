# TICKET-005: YOLOv8 Text Detection Module

## Summary
Install ultralytics, pull pretrained weights, and build the `TextDetector` class that takes an image path and returns a `DetectionResult` containing all detected text bounding boxes. This ticket covers both model setup and the extraction module.

## Language and Tools
- Python 3.11
- `ultralytics` (YOLOv8 library by Ultralytics)
- Install: `uv add ultralytics`
- Ultralytics pulls in opencv-python, numpy, and torch (already installed from TICKET-004) as dependencies

## Model Selection Decision

Three candidates ranked by expected accuracy on manga speech bubbles:

1. **CTD (Comic Text Detector)** — manga-image-translator's purpose-built manga detector. Pre-trained on comic panel data, handles irregular bubble shapes and SFX regions separately. Weights available at their repo: `zyddnys/manga-image-translator` under `data/comic-text-detector.pt`. This is the strongest candidate and should be evaluated first.

2. **Fine-tuned YOLOv8 manga bubble detector** — HuggingFace models such as `Atif-Anwer/Manga-speech-bubbles-detection` trained specifically on manga panel layouts. Simpler inference API than CTD since it stays inside the ultralytics ecosystem.

3. **`yolov8n.pt`** (YOLOv8 Nano, general object detection) — 6MB, fast, not manga-specific. Fallback only if the above two underperform on the target series' art style.

Recommendation: Download CTD weights and the best HuggingFace YOLOv8 bubble model, run both on 5 pages from the target series, and pick whichever produces fewer false positives on panel borders and SFX. The `model_path` in `config.yaml` makes this a one-line swap at any point.

```yaml
detection:
  model_path: "models/comic-text-detector.pt"   # or models/manga_bubble_detector.pt
```

## CTD Integration Note

CTD is not an ultralytics model — it has its own inference interface. If CTD wins the evaluation, the `TextDetector` class must wrap CTD's API instead of ultralytics YOLO. The core change is isolated to `detector.py`; everything downstream consuming `DetectionResult` is unaffected. Evaluate this before committing to the implementation below, because CTD integration is meaningfully different code from the YOLOv8 path.

Pull CTD weights:
```bash
# from manga-image-translator repo releases
wget https://github.com/zyddnys/manga-image-translator/releases/download/beta-0.3/comictextdetector.pt -O models/comic-text-detector.pt
```

## Implementation

File: `src/manga_scanner/detection/detector.py`

```python
from pathlib import Path
from ultralytics import YOLO
from manga_scanner.types import BoundingBox, DetectionResult
from manga_scanner.config import DetectionConfig
import logging

logger = logging.getLogger(__name__)


class TextDetector:
    def __init__(self, config: DetectionConfig):
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
        boxes = []
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

    def unload(self):
        del self.model
        import torch, gc
        torch.cuda.empty_cache()
        gc.collect()
```

## Key Implementation Notes

- `verbose=False` suppresses per-inference console spam from ultralytics.
- Keep the `TextDetector` instance alive for the full chapter (do not reinstantiate per page). YOLO loads weights to VRAM on first call; recreating per page wastes 300ms each time.
- Call `unload()` only after processing all pages in the chapter, before loading the inpainter.
- YOLOv8 returns `results[0].boxes` — the `[0]` index is because we pass one image at a time. Batch inference (list of paths) is possible but complicates memory management.
- Bounding box coordinates from ultralytics are floats; convert to `int` at this boundary so the rest of the pipeline works in pixel-integer space.

## Acceptance Criteria
- `uv run python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"` downloads weights and exits 0
- `TextDetector(config).detect(Path("tests/fixtures/pages/sample.png"))` returns a `DetectionResult`
- `DetectionResult.boxes` is a list (may be empty on blank/non-manga images)
- Running on a manga page with speech bubbles returns at least one box with confidence above threshold

## Dependencies
- TICKET-001 (project skeleton)
- TICKET-002 (BoundingBox, DetectionResult types)
- TICKET-003 (DetectionConfig)
- TICKET-004 (PyTorch/CUDA must be installed first)

## Competitive Benchmark (vs manga-image-translator)

manga-image-translator uses CTD as its primary detector for Japanese content and reports it handles vertical text, SFX, and irregular bubble shapes better than CRAFT. Their detection pipeline feeds into inpainting separately from OCR, similar to our approach. Key difference: their detector targets raw text regions, not speech bubble boundaries — meaning their masks are tighter around glyphs whereas ours (bubble-boundary) are larger and more conservative. Larger masks = more inpainting surface area = more LaMa load but cleaner results on text that bleeds into the bubble edge.

Track these metrics across both detectors on the same 5 test pages:
- Recall: bubbles detected / total bubbles on page
- Precision: correct bubble detections / total detections (penalize panel borders, SFX misfires)
- Mask area: average pixels masked per page (proxy for inpainting cost)

Document results in `models/detector_eval.md` before locking in the model.

## Estimated Effort
4 hours (including CTD download, both model evaluations, and threshold tuning)
