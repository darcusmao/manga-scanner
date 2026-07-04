# TICKET-008: iopaint Installation, Weight Download, and Python API Verification

## Summary
Install iopaint (the actively maintained successor to lama-cleaner), trigger the LaMa model weight download, and verify that the Python API is accessible before writing the wrapper in TICKET-009. This ticket exists specifically because iopaint's internal Python API is not part of its public contract — it must be tested before the wrapper is designed around it.

## Language and Tools
- Python 3.11
- `iopaint` — neural inpainting library
- Install: `uv add iopaint`
- LaMa weights: auto-downloaded to `~/.cache/iopaint/lama/` on first use (~500MB)

## Why Verify Before Wrapping

iopaint exposes a CLI (`iopaint run --model=lama ...`) and a web server but does not document a stable Python API. The `ModelManager` class used for programmatic access is internal. If it has changed or doesn't expose the interface we need, the wrapper design in TICKET-009 must be adjusted to use subprocess/CLI instead. This verification must happen before TICKET-009.

## Verification Script

File: `scripts/test_inpaint.py`

```python
#!/usr/bin/env python3
"""
Verify iopaint Python API is accessible and LaMa can process an image.
Run once during setup. Requires a CUDA GPU (~2.5 GB VRAM).
"""
import numpy as np
from PIL import Image
from pathlib import Path
import sys

OUTPUT_PATH = Path("data/output/inpaint_test.png")
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)


def make_test_inputs():
    # 256x256 gray manga-like image
    img = np.full((256, 256, 3), 200, dtype=np.uint8)
    # draw a simulated text region (black rectangle)
    img[80:120, 60:180] = 0
    # mask covering that region
    mask = np.zeros((256, 256), dtype=np.uint8)
    mask[80:120, 60:180] = 255
    return img, mask


def attempt_python_api(img, mask):
    print("Attempting iopaint Python API...")
    try:
        from iopaint.model_manager import ModelManager
        from iopaint.schema import InpaintRequest, HDStrategy, LDMSampler

        manager = ModelManager(name="lama", device="cuda")
        pil_img = Image.fromarray(img)
        pil_mask = Image.fromarray(mask)
        req = InpaintRequest(hd_strategy=HDStrategy.ORIGINAL)
        result = manager(pil_img, pil_mask, req)
        result.save(OUTPUT_PATH)
        print(f"Python API: OK. Output saved to {OUTPUT_PATH}")
        return True
    except Exception as e:
        print(f"Python API failed: {e}")
        return False


def attempt_subprocess_api(img, mask):
    import subprocess, tempfile
    print("Attempting iopaint via subprocess CLI...")
    with tempfile.TemporaryDirectory() as tmp:
        img_path = Path(tmp) / "img.png"
        mask_path = Path(tmp) / "mask.png"
        Image.fromarray(img).save(img_path)
        Image.fromarray(mask).save(mask_path)
        result = subprocess.run(
            ["iopaint", "run", "--model=lama", "--device=cuda",
             f"--image={img_path}", f"--mask={mask_path}", f"--output={OUTPUT_PATH}"],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            print(f"Subprocess CLI: OK. Output saved to {OUTPUT_PATH}")
            return True
        else:
            print(f"Subprocess CLI failed:\n{result.stderr}")
            return False


if __name__ == "__main__":
    img, mask = make_test_inputs()
    if attempt_python_api(img, mask):
        print("\nResult: Use Python API in TICKET-009.")
    elif attempt_subprocess_api(img, mask):
        print("\nResult: Use subprocess CLI in TICKET-009. Adjust wrapper accordingly.")
    else:
        print("\nResult: iopaint is broken in this environment. Investigate before TICKET-009.")
        sys.exit(1)
```

## Model Variant Selection: lama vs lama_large

manga-image-translator ships `lama_large` as a separate option alongside standard LaMa. iopaint supports both variants. The differences:

| Variant | VRAM | Quality on screentone | Quality on linework | Notes |
|---|---|---|---|---|
| `lama` | ~2.5 GB | Good | Good | Default; suitable for clean-line manga |
| `lama_large` | ~4.5 GB | Better | Better | Better reconstruction on dense crosshatching and toned backgrounds |
| `lama_mpe` | ~3 GB | Good | Excellent | Masked-position encoding variant; better spatial coherence on long text runs |

On an 8 GB GPU, `lama_large` leaves only ~3.5 GB for Qwen2.5-7B (which needs ~5 GB). This means `lama_large` is only viable if Ollama has already evicted its model from VRAM before inpainting begins, or on a 12+ GB card.

Run the verification script against both `lama` and `lama_large` on a page with dense screentone. If visual quality difference is negligible on the target series, use `lama`. If there are noticeable halo artifacts from `lama`, switch to `lama_large` and accept the VRAM constraint.

Update `config.yaml` accordingly:
```yaml
inpainting:
  model_name: "lama"        # or "lama_large" if visual artifacts require it
```

## Weight Download
Running the verification script will trigger the LaMa weight download. This is a ~500MB download from HuggingFace (larger for lama_large). Weights are cached at `~/.cache/iopaint/`. Ensure a stable internet connection for the first run.

## Expected Output on Success (Python API path)
```
Attempting iopaint Python API...
Python API: OK. Output saved to data/output/inpaint_test.png
Result: Use Python API in TICKET-009.
```

Open `data/output/inpaint_test.png` visually and confirm the black rectangle region has been inpainted with a plausible grey background fill.

## Acceptance Criteria
- Script exits 0 for both `lama` and `lama_large` variants
- `data/output/inpaint_test_lama.png` and `data/output/inpaint_test_lama_large.png` both exist
- The inpainted region visually blends with surrounding pixels in both outputs
- The script explicitly prints which code path (Python API or CLI) succeeded — this determines how TICKET-009 is implemented
- A one-line note is added to `models/WEIGHTS.txt` recording which lama variant was chosen and why

## Dependencies
- TICKET-001 (project skeleton)
- TICKET-004 (PyTorch/CUDA)

## Estimated Effort
3 hours (including both model variant downloads, visual comparison, and variant selection decision)
