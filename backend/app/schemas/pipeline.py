from typing import Annotated
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from app.models import OCRTextType, PageStatus, PhotoStatus
from app.schemas.common import BoundingBox, SegmentationMask, StrictSchema


class AnalyzeRequest(StrictSchema):
    page_ids: list[Annotated[UUID, Field(strict=False)]] | None = None


class PageRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    album_id: UUID
    original_filename: str
    storage_path: str
    file_size_bytes: int | None
    file_hash_sha256: str
    blur_score: float | None
    status: PageStatus
    error_message: str | None
    processing_metadata: dict


class OCRRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    page_id: UUID
    photo_id: UUID | None
    text_content: str
    text_type: OCRTextType | None
    bounding_box: dict | None
    ocr_engine: str
    confidence: float | None
    spatial_classification_reason: str | None
    is_verified: bool


class PhotoRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    page_id: UUID
    album_id: UUID
    storage_path: str
    original_storage_path: str | None
    bounding_box: dict
    segmentation_mask: dict | None
    segmentation_confidence: float | None
    aspect_ratio: float | None
    geometry_valid: bool
    phash: str | None
    is_duplicate_of: UUID | None
    enhancement_applied: dict
    status: PhotoStatus


class PhotoDetail(PhotoRead):
    ocr_results: list[OCRRead] = []
    urls: dict[str, str | None] = Field(default_factory=dict)


class AlbumStats(BaseModel):
    id: UUID
    name: str
    description: str | None
    directory_path: str | None
    status: str
    total_pages: int
    processed_pages: int
    photos_extracted: int
    review_needed_count: int


class BoundingBoxUpdate(StrictSchema):
    bounding_box: BoundingBox
    segmentation_mask: SegmentationMask | None = None


class OCRUpdate(StrictSchema):
    text_content: str = Field(min_length=1, max_length=20_000)
    text_type: OCRTextType
    is_verified: bool


class ReviewItem(BaseModel):
    item_type: str
    id: UUID
    album_id: UUID
    page_id: UUID | None = None
    status: str
    reason: str | None = None
    preview_url: str | None = None

class SearchResult(BaseModel):
    photo_id: UUID | None
    page_id: UUID
    text_content: str
    text_type: OCRTextType | None
    confidence: float | None
    highlight: str
