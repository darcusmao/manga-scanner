# TICKET-004: PyTorch and CUDA Hardware Verification

## Summary
Install PyTorch with CUDA support and write a hardware probe script that confirms the GPU is visible, reports available VRAM, and prints a clear failure message if CUDA is absent. This runs once as a setup check and informs whether CPU fallback paths are needed.

## Language and Tools
- Python 3.11
- PyTorch with CUDA 12.1 support
- Install command (CUDA 12.1):
  ```
  uv add torch torchvision --index-url https://download.pytorch.org/whl/cu121
  ```
- If running CUDA 11.8:
  ```
  uv add torch torchvision --index-url https://download.pytorch.org/whl/cu118
  ```
- PyTorch download is ~2GB. Do this before any other model downloads.
- Do NOT install the CPU-only torch first and then replace it; uv will handle the index URL correctly.

## Determining Your CUDA Version
Before installing, check which CUDA version the driver supports:
```bash
nvidia-smi
# Look for "CUDA Version: XX.X" in the top-right of the output
```
The CUDA version shown by `nvidia-smi` is the maximum supported version. Install the matching or lower PyTorch wheel.

## Implementation

File: `scripts/check_hardware.py`

```python
#!/usr/bin/env python3
"""Run this once after setup to verify GPU visibility before proceeding."""

import sys


def check():
    try:
        import torch
    except ImportError:
        print("FAIL: torch is not installed. Run: uv add torch torchvision")
        sys.exit(1)

    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available:  {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        print(
            "WARNING: CUDA not found. The pipeline will run on CPU and will be significantly slower.\n"
            "Ensure you installed the CUDA-enabled wheel (see TICKET-004).\n"
            "Detection and inpainting will still function but at reduced throughput."
        )
        return

    device_count = torch.cuda.device_count()
    for i in range(device_count):
        props = torch.cuda.get_device_properties(i)
        total_vram = props.total_memory / (1024 ** 3)
        free_vram = (props.total_memory - torch.cuda.memory_allocated(i)) / (1024 ** 3)
        print(f"GPU {i}: {props.name}")
        print(f"  Total VRAM: {total_vram:.1f} GB")
        print(f"  Free  VRAM: {free_vram:.1f} GB")
        if total_vram < 7.5:
            print(
                f"  WARNING: {total_vram:.1f} GB VRAM is tight. "
                "Qwen2.5-7B Q4_K_M requires ~5 GB; LaMa requires ~2.5 GB. "
                "They cannot coexist in VRAM — sequential unloading (TICKET-020) is mandatory."
            )

    print("\nRunning a small CUDA smoke test...")
    x = torch.ones(3, 3, device="cuda")
    assert x.sum().item() == 9.0
    print("CUDA smoke test passed.")


if __name__ == "__main__":
    check()
```

Run with: `uv run python scripts/check_hardware.py`

## VRAM Baseline
Expected values on a consumer GPU (RTX 3080 10GB example):
```
GPU 0: NVIDIA GeForce RTX 3080
  Total VRAM: 10.0 GB
  Free  VRAM: 9.8 GB
CUDA smoke test passed.
```
If total VRAM is less than 8 GB, note that Qwen2.5-7B Q4_K_M may not fit alongside iopaint/LaMa simultaneously. TICKET-020 addresses the sequential load/unload strategy.

## Acceptance Criteria
- `uv run python scripts/check_hardware.py` exits 0 on a CUDA-capable machine
- GPU name, total VRAM, and free VRAM are printed
- If CUDA is unavailable, script exits 0 but prints a clear WARNING (does not crash)

## Dependencies
- TICKET-001 (project skeleton)

## Estimated Effort
1 hour
