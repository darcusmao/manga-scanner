"""Tests for TICKET-003: configuration system."""
import os
import pytest
from pathlib import Path
from pydantic import ValidationError


def test_defaults_when_file_absent(tmp_path):
    from manga_scanner.config import load_config
    cfg = load_config(tmp_path / "nonexistent.yaml")
    assert cfg.translation.temperature == 0.2
    assert cfg.detection.confidence_threshold == 0.45
    assert cfg.detection.box_padding == 8
    assert cfg.inpainting.model_name == "lama"
    assert cfg.translation.max_retries == 2
    assert cfg.pipeline.skip_existing is True


def test_loads_from_yaml(tmp_path):
    from manga_scanner.config import load_config
    yaml_file = tmp_path / "config.yaml"
    yaml_file.write_text(
        "translation:\n"
        "  temperature: 0.5\n"
        "  model_name: test-model\n"
        "detection:\n"
        "  confidence_threshold: 0.6\n"
    )
    cfg = load_config(yaml_file)
    assert cfg.translation.temperature == 0.5
    assert cfg.translation.model_name == "test-model"
    assert cfg.detection.confidence_threshold == 0.6
    # unspecified keys stay at defaults
    assert cfg.inpainting.model_name == "lama"


def test_env_var_overrides_yaml(tmp_path, monkeypatch):
    from manga_scanner.config import load_config
    yaml_file = tmp_path / "config.yaml"
    yaml_file.write_text("translation:\n  temperature: 0.8\n")
    monkeypatch.setenv("MANGA_TRANSLATION__TEMPERATURE", "0.1")
    cfg = load_config(yaml_file)
    assert cfg.translation.temperature == 0.1


def test_env_var_overrides_defaults(monkeypatch):
    from manga_scanner.config import load_config
    monkeypatch.setenv("MANGA_DETECTION__CONFIDENCE_THRESHOLD", "0.99")
    cfg = load_config(Path("nonexistent.yaml"))
    assert cfg.detection.confidence_threshold == 0.99


def test_invalid_value_raises_at_load_time():
    from manga_scanner.config import Config
    with pytest.raises(ValidationError):
        Config(translation={"temperature": "hot"})


def test_invalid_nested_int_raises():
    from manga_scanner.config import Config
    with pytest.raises(ValidationError):
        Config(detection={"box_padding": "wide"})


def test_all_sub_configs_present():
    from manga_scanner.config import load_config
    cfg = load_config(Path("nonexistent.yaml"))
    assert cfg.detection is not None
    assert cfg.inpainting is not None
    assert cfg.ocr is not None
    assert cfg.translation is not None
    assert cfg.typesetting is not None
    assert cfg.pipeline is not None
    assert cfg.paths is not None


def test_typesetting_defaults():
    from manga_scanner.config import load_config
    cfg = load_config(Path("nonexistent.yaml"))
    assert cfg.typesetting.max_font_size == 24
    assert cfg.typesetting.min_font_size == 8
    assert cfg.typesetting.padding == 6
    assert cfg.typesetting.text_color == "#000000"


def test_paths_default_output_dir():
    from manga_scanner.config import load_config
    cfg = load_config(Path("nonexistent.yaml"))
    assert cfg.paths.output_dir == "data/output"


def test_empty_yaml_file_uses_defaults(tmp_path):
    from manga_scanner.config import load_config
    yaml_file = tmp_path / "config.yaml"
    yaml_file.write_text("")
    cfg = load_config(yaml_file)
    assert cfg.translation.temperature == 0.2
