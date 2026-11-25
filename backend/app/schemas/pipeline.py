from typing import Annotated, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator

from app.models import OCRTextType, PageStatus, PhotoStatus
from app.pipeline.decisions import motif_geometry_requires_review, photo_is_reviewed_non_print
from app.pipeline.variants import photo_image_url, variant_path_from_photo
from app.schemas.common import BoundingBox, NormalizedPoint, PrintSizeMM, SegmentationMask, StrictSchema


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

    @computed_field
    @property
    def reviewed_non_print(self) -> bool:
        return photo_is_reviewed_non_print(self)

    @computed_field
    @property
    def motif_review_pending(self) -> bool:
        return motif_geometry_requires_review(self.enhancement_applied.get("correction"))

    @computed_field
    @property
    def image_url(self) -> str:
        return photo_image_url(self)

    @computed_field
    @property
    def crop_download_urls(self) -> dict[str, str | None]:
        return {
            mode: photo_image_url(self, mode) if variant_path_from_photo(self, mode) else None
            for mode in ("full_print", "image_only")
        }


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


class ReviewSegmentationMask(SegmentationMask):
    """Reviewer geometry and an optional clockwise turn after rectification."""

    visible_polygon: list[NormalizedPoint] | None = Field(default=None, min_length=4)
    warp_quad: list[NormalizedPoint] | None = Field(default=None, min_length=4, max_length=4)
    image_quad: list[NormalizedPoint] | None = Field(default=None, min_length=4, max_length=4)
    print_size_mm: PrintSizeMM | None = None
    orientation_degrees: int | None = Field(default=None, ge=0, le=270, multiple_of=90)


class BoundingBoxUpdate(StrictSchema):
    bounding_box: BoundingBox
    segmentation_mask: ReviewSegmentationMask | None = None
    confirm_print_outline: bool = False


class AdjacentPrintSuggestion(StrictSchema):
    photo_id: Annotated[UUID, Field(strict=False)]
    polygon: list[NormalizedPoint] = Field(min_length=4, max_length=4)
    kind: Literal["nested"] | None = None


class ManualPhotoCreate(StrictSchema):
    request_id: Annotated[UUID, Field(strict=False)]
    warp_quad: list[NormalizedPoint] = Field(min_length=4, max_length=4)
    print_size_mm: PrintSizeMM | None = None
    adjacent_suggestion: AdjacentPrintSuggestion | None = None
    deferred_proposal_id: int | None = Field(default=None, ge=0)
    photo_count_revision: str | None = Field(default=None, min_length=1, max_length=64)

    @model_validator(mode="after")
    def check_suggestion(self) -> "ManualPhotoCreate":
        if self.adjacent_suggestion is not None and self.deferred_proposal_id is not None:
            raise ValueError("Choose one print suggestion.")
        return self


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
    payload: dict = Field(default_factory=dict)


class SearchResult(BaseModel):
    photo_id: UUID | None
    page_id: UUID
    text_content: str
    text_type: OCRTextType | None
    confidence: float | None
    highlight: str
