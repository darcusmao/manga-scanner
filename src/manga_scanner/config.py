from pathlib import Path
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict, YamlConfigSettingsSource


class DetectionConfig(BaseModel):
    model_path: str = "yolov8n.pt"
    confidence_threshold: float = 0.45
    device: str = "cuda"
    box_padding: int = 8
    row_threshold: int = 50


class InpaintingConfig(BaseModel):
    model_name: str = "lama"
    device: str = "cuda"


class OCRConfig(BaseModel):
    device: str = "cuda"


class TranslationConfig(BaseModel):
    backend: str = "ollama"  # "ollama" | "deepl" | "google"
    # Ollama settings
    ollama_url: str = "http://localhost:11434"
    model_name: str = "qwen2.5:7b-instruct-q4_K_M"
    temperature: float = 0.2
    max_retries: int = 2
    timeout_seconds: int = 300
    # API-based backends (prefer env vars: MANGA_TRANSLATION__DEEPL_API_KEY, etc.)
    deepl_api_key: str = ""
    google_api_key: str = ""


class TypesettingConfig(BaseModel):
    font_path: str = "fonts/Bangers-Regular.ttf"
    max_font_size: int = 24
    min_font_size: int = 8
    text_color: str = "#000000"
    padding: int = 6


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
    if not config_path.exists():
        return Config()

    # Explicit source order: init > env vars > yaml > defaults.
    # Passing YAML as **kwargs to Config() would make it init_settings
    # (highest priority), incorrectly outranking env vars.
    _path = config_path

    class _Config(Config):
        @classmethod
        def settings_customise_sources(cls, settings_cls, init_settings, env_settings, dotenv_settings, file_secret_settings):
            return (
                init_settings,
                env_settings,
                YamlConfigSettingsSource(settings_cls, yaml_file=_path),
            )

    return _Config()
