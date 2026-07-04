# TICKET-010: manga-ocr Installation and OCR Wrapper

## Summary
Install the `manga-ocr` library, trigger the HuggingFace weight download, and build the `MangaOCR` wrapper class that transcribes a PIL image crop into a Japanese text string.

## Language and Tools
- Python 3.11
- `manga-ocr` — a pre-trained vision transformer OCR model specialized for manga panel text
- Install: `uv add manga-ocr`
- HuggingFace model: `kha-white/manga-ocr-base` (~420MB)
- The library handles CUDA detection automatically; no explicit device configuration needed
- `transformers` and `torch` are pulled as transitive dependencies (already installed)

## Weight Download
Weights are downloaded from HuggingFace on first initialization of `MangaOcr()`. This requires internet access once. They are cached at `~/.cache/huggingface/hub/`. After the first run, the pipeline is fully offline-capable.

Trigger the download manually (so it doesn't add latency on first chapter run):
```bash
uv run python -c "from manga_ocr import MangaOcr; MangaOcr()"
```

## Implementation

File: `src/manga_scanner/ocr/ocr.py`

```python
import gc
import logging
from PIL import Image
from manga_scanner.config import OCRConfig

logger = logging.getLogger(__name__)


class MangaOCR:
    def __init__(self, config: OCRConfig):
        logger.info("Loading manga-ocr model...")
        from manga_ocr import MangaOcr
        self.model = MangaOcr()
        logger.info("manga-ocr model loaded.")

    def transcribe(self, crop: Image.Image) -> str:
        """
        Returns the Japanese text found in the crop image.
        Returns an empty string if the model finds no text or the crop is blank.
        """
        try:
            result = self.model(crop)
            return result.strip()
        except Exception as e:
            logger.warning("OCR failed on crop: %s", e)
            return ""

    def transcribe_batch(self, crops: list[Image.Image]) -> list[str]:
        """Sequential transcription. manga-ocr does not expose batch inference."""
        return [self.transcribe(crop) for crop in crops]

    def unload(self) -> None:
        logger.info("Unloading manga-ocr model from VRAM.")
        del self.model
        import torch
        torch.cuda.empty_cache()
        gc.collect()
```

## Notes on manga-ocr Behavior
- Designed specifically for manga text: handles vertical text, handwritten fonts, and stylized lettering better than general-purpose OCR (Tesseract, EasyOCR)
- Works on individual crops — it is not designed for full-page inference
- Returns a plain string, no confidence scores
- If a crop is nearly blank or contains a non-text drawing, it may return an empty string or noise characters — the orchestrator must filter these

## Competitive Benchmark: manga-ocr vs MIT 48px Model

manga-image-translator ships a custom-trained 48px OCR model (`48px_ctc` variant) that was trained specifically on their detection pipeline's crop outputs. It may outperform manga-ocr on:
- Very small text (< 20px character height)
- Heavily stylized or hand-lettered fonts
- Text partially obscured by screentone

manga-ocr may outperform the 48px model on:
- Standard vertical dialogue text (its primary training distribution)
- Modern digital manga with clean fonts

### How to run the benchmark

1. Pull their OCR weights from their releases (the `48px_ctc` model, ~80MB)
2. Write a minimal inference wrapper using their `ocr` module as reference: `zyddnys/manga-image-translator/blob/main/manga_translator/ocr/`
3. On 30-50 crops sampled from the target series (mix of small, stylized, and standard text), run both models and record:
   - Character error rate (CER) against a manually transcribed ground truth
   - Empty/noise output rate (crops where model returns garbage)

If CER difference is < 5%, stick with manga-ocr (simpler dependency). If the 48px model wins by a material margin on stylized text, integrate it as a second engine and expose `ocr.engine: "manga_ocr" | "48px_ctc"` in the config.

This benchmark should be run during TICKET-010 development on real crops, not synthetic fixtures.

## Acceptance Criteria
- `uv run python -c "from manga_scanner.ocr.ocr import MangaOCR"` imports cleanly
- First instantiation triggers weight download and logs "manga-ocr model loaded."
- `transcribe(crop)` returns a non-empty string on a crop containing Japanese text
- `transcribe(blank_image)` returns `""` without raising

## Dependencies
- TICKET-001 (project skeleton)
- TICKET-003 (OCRConfig)
- TICKET-004 (PyTorch must be installed)

## Estimated Effort
4 hours (including weight download, OCR benchmark on real crops, and variant decision)
