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
