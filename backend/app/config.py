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
    max_image_pixels: int = Field(default=40_000_000, alias="MAX_IMAGE_PIXELS")
    max_image_frames: int = Field(default=1, alias="MAX_IMAGE_FRAMES")
    pipeline_version: str = Field(default="2026-11-precise-crops-v15", alias="PIPELINE_VERSION")
    auth_required: bool = Field(default=False, alias="AUTH_REQUIRED")
    blur_threshold: float = Field(default=100.0, alias="BLUR_THRESHOLD")
    yolo_confidence_threshold: float = Field(default=0.7, alias="YOLO_CONFIDENCE_THRESHOLD")
    yolo_proposal_threshold: float = Field(default=0.25, alias="YOLO_PROPOSAL_THRESHOLD")
    segmentation_min_aspect_ratio: float = Field(default=0.5, alias="SEGMENTATION_MIN_ASPECT_RATIO")
    segmentation_max_aspect_ratio: float = Field(default=3.0, alias="SEGMENTATION_MAX_ASPECT_RATIO")
    yolo_model_path: Path = Field(default=Path("/app/models/yolov8-seg-album.onnx"), alias="YOLO_MODEL_PATH")
    visible_mask_model_path: Path | None = Field(default=None, alias="VISIBLE_MASK_MODEL_PATH")
    model_manifest_path: Path = Field(default=Path("/app/models/manifest.json"), alias="MODEL_MANIFEST_PATH")
    require_registered_models: bool = Field(default=True, alias="REQUIRE_REGISTERED_MODELS")
    segmentation_calibration_path: Path = Field(
        default=Path("/app/models/segmentation-calibration.json"), alias="SEGMENTATION_CALIBRATION_PATH"
    )
    require_confidence_calibration: bool = Field(default=True, alias="REQUIRE_CONFIDENCE_CALIBRATION")
    crop_validation_report_path: Path = Field(
        default=Path("/app/models/crop-validation.json"), alias="CROP_VALIDATION_REPORT_PATH"
    )
    camera_calibration_path: Path | None = Field(default=None, alias="CAMERA_CALIBRATION_PATH")
    realesrgan_model_path: Path = Field(
        default=Path("/app/models/RealESRGAN_x2plus.pth"), alias="REALESRGAN_MODEL_PATH"
    )
    gfpgan_model_path: Path = Field(default=Path("/app/models/GFPGANv1.3.pth"), alias="GFPGAN_MODEL_PATH")
    segmentation_training_workspace: Path = Field(
        default=Path("data/segmentation_training"), alias="SEGMENTATION_TRAINING_WORKSPACE"
    )
    enable_photo_orientation_correction: bool = Field(default=True, alias="ENABLE_PHOTO_ORIENTATION_CORRECTION")
    photo_orientation_model_path: Path = Field(
        default=Path("/app/models/photo-orientation.onnx"), alias="PHOTO_ORIENTATION_MODEL_PATH"
    )
    paddleocr_model_path: Path = Field(default=Path("/app/models/paddleocr/"), alias="PADDLEOCR_MODEL_PATH")
    paddleocr_language: str = Field(default="en", alias="PADDLEOCR_LANGUAGE")
    enable_llm_fallback: bool = Field(default=False, alias="ENABLE_LLM_FALLBACK")
    llm_model_path: Path = Field(default=Path("/app/models/llava-phi3/"), alias="LLM_MODEL_PATH")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    celery_task_always_eager: bool = Field(default=False, alias="CELERY_TASK_ALWAYS_EAGER")
    enable_classical_segmentation_fallback: bool = Field(default=True, alias="ENABLE_CLASSICAL_SEGMENTATION_FALLBACK")
    enable_precise_photo_crops: bool = Field(default=True, alias="ENABLE_PRECISE_PHOTO_CROPS")
    enable_border_trim: bool = Field(default=True, alias="ENABLE_BORDER_TRIM")
    enable_protected_region_guard: bool = Field(default=True, alias="ENABLE_PROTECTED_REGION_GUARD")
    enable_conservative_enhancement: bool = Field(default=True, alias="ENABLE_CONSERVATIVE_ENHANCEMENT")
    enable_deep_enhancement: bool = Field(default=False, alias="ENABLE_DEEP_ENHANCEMENT")
    enable_premium_enhancement: bool = Field(default=False, alias="ENABLE_PREMIUM_ENHANCEMENT")
    retention_derivative_days: int = Field(default=30, alias="RETENTION_DERIVATIVE_DAYS")
    retention_dlq_days: int = Field(default=14, alias="RETENTION_DLQ_DAYS")
    shadow_mode: bool = Field(default=True, alias="SHADOW_MODE")
    canary_percentage: int = Field(default=0, ge=0, le=100, alias="CANARY_PERCENTAGE")
    release_kill_switch: bool = Field(default=True, alias="RELEASE_KILL_SWITCH")
    auto_analyze_uploads: bool = Field(default=True, alias="AUTO_ANALYZE_UPLOADS")
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
