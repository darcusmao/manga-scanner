# TICKET-003: Configuration System

## Summary
Build a typed configuration system backed by a `config.yaml` file with environment variable overrides. All tunable parameters (thresholds, model names, paths, timeouts) live here rather than being hardcoded. Every module reads from a `Config` object passed at construction time.

## Language and Tools
- Python 3.11
- `pydantic-settings` v2 — settings management with validation and env override
- `pyyaml` — parse the YAML config file
- Install: `uv add pydantic-settings pyyaml`

## Implementation

### `src/manga_scanner/config.py`

```python
from pathlib import Path
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict
import yaml


class DetectionConfig(BaseModel):
    model_path: str = "yolov8n.pt"
    confidence_threshold: float = 0.45
    device: str = "cuda"
    box_padding: int = 8          # pixels to expand each bbox before masking


class InpaintingConfig(BaseModel):
    model_name: str = "lama"
    device: str = "cuda"


class OCRConfig(BaseModel):
    device: str = "cuda"


class TranslationConfig(BaseModel):
    ollama_url: str = "http://localhost:11434"
    model_name: str = "qwen2.5:7b-instruct-q4_K_M"
    temperature: float = 0.2
    max_retries: int = 2
    timeout_seconds: int = 90


class TypesettingConfig(BaseModel):
    font_path: str = "fonts/anime_ace.ttf"
    max_font_size: int = 24
    min_font_size: int = 8
    text_color: str = "#000000"
    padding: int = 6              # pixels of inset from bbox edge


class PipelineConfig(BaseModel):
    skip_existing: bool = True


class PathsConfig(BaseModel):
    output_dir: str = "data/output"


class Config(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="MANGA_",
        env_nested_delimiter="__",
    )

    detection: DetectionConfig = DetectionConfig()
    inpainting: InpaintingConfig = InpaintingConfig()
    ocr: OCRConfig = OCRConfig()
    translation: TranslationConfig = TranslationConfig()
    typesetting: TypesettingConfig = TypesettingConfig()
    pipeline: PipelineConfig = PipelineConfig()
    paths: PathsConfig = PathsConfig()


def load_config(config_path: Path = Path("config.yaml")) -> Config:
    if config_path.exists():
        with open(config_path) as f:
            data = yaml.safe_load(f) or {}
        return Config(**data)
    return Config()
```

### `config.yaml` (default values file, committed to repo)

```yaml
detection:
  model_path: "yolov8n.pt"
  confidence_threshold: 0.45
  device: "cuda"
  box_padding: 8

inpainting:
  model_name: "lama"
  device: "cuda"

ocr:
  device: "cuda"

translation:
  ollama_url: "http://localhost:11434"
  model_name: "qwen2.5:7b-instruct-q4_K_M"
  temperature: 0.2
  max_retries: 2
  timeout_seconds: 90

typesetting:
  font_path: "fonts/anime_ace.ttf"
  max_font_size: 24
  min_font_size: 8
  text_color: "#000000"
  padding: 6

pipeline:
  skip_existing: true

paths:
  output_dir: "data/output"
```

## Environment Variable Overrides
Pydantic-settings allows nested overrides via env vars with the `MANGA_` prefix and `__` delimiter:
```
MANGA_TRANSLATION__TEMPERATURE=0.1
MANGA_DETECTION__DEVICE=cpu
```
This is useful for quick one-off overrides without editing config.yaml.

## Acceptance Criteria
- `load_config()` returns a `Config` instance with default values when `config.yaml` is absent
- `load_config(Path("config.yaml"))` correctly populates nested models from YAML
- Env var `MANGA_TRANSLATION__TEMPERATURE=0.1` overrides `config.translation.temperature`
- Invalid values (e.g., `temperature: "hot"`) raise a Pydantic `ValidationError` at load time, not at inference time

## Dependencies
- TICKET-001 (project skeleton)
- TICKET-002 (types.py, though config.py does not import from it)

## Estimated Effort
2 hours
