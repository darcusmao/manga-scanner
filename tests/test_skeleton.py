"""Tests for TICKET-001: project skeleton — all modules importable without side effects."""
import subprocess
import sys


MODULES = [
    "manga_scanner",
    "manga_scanner.config",
    "manga_scanner.types",
    "manga_scanner.detection.detector",
    "manga_scanner.detection.masker",
    "manga_scanner.detection.sorter",
    "manga_scanner.inpainting.inpainter",
    "manga_scanner.ocr.ocr",
    "manga_scanner.ocr.cropper",
    "manga_scanner.translation.translator",
    "manga_scanner.translation.prompt_builder",
    "manga_scanner.typesetting.fitter",
    "manga_scanner.typesetting.renderer",
    "manga_scanner.pipeline.orchestrator",
    "manga_scanner.pipeline.batch",
]


def test_all_modules_importable():
    # Run in a subprocess so heavy torch side effects don't pollute the test process
    imports = "; ".join(f"import {m}" for m in MODULES)
    result = subprocess.run(
        [sys.executable, "-c", imports],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"One or more modules failed to import:\n{result.stderr}"
    )


def test_hardware_script_exits_zero():
    """TICKET-004: check_hardware.py must exit 0 even without CUDA."""
    result = subprocess.run(
        [sys.executable, "scripts/check_hardware.py"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"check_hardware.py failed:\n{result.stderr}"
    assert "PyTorch version:" in result.stdout


def test_hardware_script_reports_cuda_status():
    result = subprocess.run(
        [sys.executable, "scripts/check_hardware.py"],
        capture_output=True,
        text=True,
    )
    assert "CUDA available:" in result.stdout
