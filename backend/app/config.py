from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    database_url: str = Field(alias="DATABASE_URL")
    redis_url: str = Field(alias="REDIS_URL")
    storage_path: Path = Field(default=Path("/app/storage"), alias="STORAGE_PATH")
    max_upload_size: int = Field(default=52_428_800, alias="MAX_UPLOAD_SIZE")
    max_request_size: int = Field(default=209_715_200, alias="MAX_REQUEST_SIZE")
    blur_threshold: float = Field(default=100.0, alias="BLUR_THRESHOLD")
    yolo_confidence_threshold: float = Field(default=0.7, alias="YOLO_CONFIDENCE_THRESHOLD")
    segmentation_min_aspect_ratio: float = Field(default=0.5, alias="SEGMENTATION_MIN_ASPECT_RATIO")
    segmentation_max_aspect_ratio: float = Field(default=3.0, alias="SEGMENTATION_MAX_ASPECT_RATIO")
    yolo_model_path: Path = Field(default=Path("/app/models/yolov8-seg-album.onnx"), alias="YOLO_MODEL_PATH")
    paddleocr_model_path: Path = Field(default=Path("/app/models/paddleocr/"), alias="PADDLEOCR_MODEL_PATH")
    enable_llm_fallback: bool = Field(default=False, alias="ENABLE_LLM_FALLBACK")
    llm_model_path: Path = Field(default=Path("/app/models/llava-phi3/"), alias="LLM_MODEL_PATH")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    celery_task_always_eager: bool = Field(default=False, alias="CELERY_TASK_ALWAYS_EAGER")
    enable_classical_segmentation_fallback: bool = Field(default=True, alias="ENABLE_CLASSICAL_SEGMENTATION_FALLBACK")
    cors_origins: list[str] = Field(
        default_factory=lambda: [
            "http://localhost:5173",
            "http://127.0.0.1:5173",
            "http://localhost:8000",
            "http://127.0.0.1:8000",
        ],
        alias="CORS_ORIGINS",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
