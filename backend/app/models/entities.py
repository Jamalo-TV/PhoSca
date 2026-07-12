from enum import Enum
from uuid import UUID, uuid4

from sqlalchemy import BigInteger, Boolean, Enum as SAEnum, Float, ForeignKey, Index, Integer, String, Text, UniqueConstraint, false, text, true
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.types import JSON

from app.database import GUID, Base, TimestampMixin


def enum_column(enum_cls: type[Enum], name: str) -> SAEnum:
    return SAEnum(
        enum_cls,
        name=name,
        values_callable=lambda enum_values: [member.value for member in enum_values],
        validate_strings=True,
    )


JSONType = JSON().with_variant(JSONB, "postgresql")


class AlbumStatus(str, Enum):
    pending = "pending"
    processing = "processing"
    completed = "completed"
    failed = "failed"


class PageStatus(str, Enum):
    uploaded = "uploaded"
    queued = "queued"
    processing = "processing"
    review_needed = "review_needed"
    completed = "completed"
    failed = "failed"


class PhotoStatus(str, Enum):
    pending = "pending"
    review_needed = "review_needed"
    completed = "completed"
    failed = "failed"


class RunStatus(str, Enum):
    pending = "pending"
    running = "running"
    review_needed = "review_needed"
    completed = "completed"
    failed = "failed"


class ApiRole(str, Enum):
    viewer = "viewer"
    editor = "editor"
    admin = "admin"


class OCRTextType(str, Enum):
    caption = "caption"
    directory_name = "directory_name"
    unknown = "unknown"


class Album(Base, TimestampMixin):
    __tablename__ = "albums"

    id: Mapped[UUID] = mapped_column(GUID(), primary_key=True, default=uuid4)
    tenant_id: Mapped[UUID | None] = mapped_column(GUID(), ForeignKey("tenants.id", ondelete="RESTRICT"))
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str | None] = mapped_column(Text)
    directory_path: Mapped[str | None] = mapped_column(String(512))
    status: Mapped[AlbumStatus] = mapped_column(
        enum_column(AlbumStatus, "album_status"),
        default=AlbumStatus.pending,
        server_default=AlbumStatus.pending.value,
        nullable=False,
    )
    total_pages: Mapped[int] = mapped_column(Integer, default=0, server_default=text("0"), nullable=False)
    processed_pages: Mapped[int] = mapped_column(Integer, default=0, server_default=text("0"), nullable=False)

    pages: Mapped[list["Page"]] = relationship(back_populates="album", cascade="all, delete-orphan")
    photos: Mapped[list["ExtractedPhoto"]] = relationship(back_populates="album", cascade="all, delete-orphan")
    tenant: Mapped["Tenant | None"] = relationship(back_populates="albums")


class Page(Base, TimestampMixin):
    __tablename__ = "pages"
    __table_args__ = (
        Index("ix_pages_album_id_status", "album_id", "status"),
        Index("ix_pages_file_hash_sha256", "file_hash_sha256"),
        UniqueConstraint("album_id", "file_hash_sha256", name="uq_pages_album_file_hash_sha256"),
    )

    id: Mapped[UUID] = mapped_column(GUID(), primary_key=True, default=uuid4)
    album_id: Mapped[UUID] = mapped_column(GUID(), ForeignKey("albums.id", ondelete="CASCADE"), nullable=False)
    original_filename: Mapped[str] = mapped_column(String(255), nullable=False)
    storage_path: Mapped[str] = mapped_column(String(512), nullable=False)
    file_size_bytes: Mapped[int | None] = mapped_column(BigInteger)
    file_hash_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    blur_score: Mapped[float | None] = mapped_column(Float)
    status: Mapped[PageStatus] = mapped_column(
        enum_column(PageStatus, "page_status"),
        default=PageStatus.uploaded,
        server_default=PageStatus.uploaded.value,
        nullable=False,
    )
    error_message: Mapped[str | None] = mapped_column(Text)
    processing_metadata: Mapped[dict] = mapped_column(JSONType, default=dict, server_default=text("'{}'"), nullable=False)

    album: Mapped[Album] = relationship(back_populates="pages")
    photos: Mapped[list["ExtractedPhoto"]] = relationship(back_populates="page", cascade="all, delete-orphan")
    ocr_results: Mapped[list["OCRResult"]] = relationship(back_populates="page", cascade="all, delete-orphan")
    processing_runs: Mapped[list["ProcessingRun"]] = relationship(back_populates="page", cascade="all, delete-orphan")


class ExtractedPhoto(Base, TimestampMixin):
    __tablename__ = "extracted_photos"
    __table_args__ = (
        Index("ix_extracted_photos_page_id", "page_id"),
        Index("ix_extracted_photos_phash", "phash"),
    )

    id: Mapped[UUID] = mapped_column(GUID(), primary_key=True, default=uuid4)
    page_id: Mapped[UUID] = mapped_column(GUID(), ForeignKey("pages.id", ondelete="CASCADE"), nullable=False)
    album_id: Mapped[UUID] = mapped_column(GUID(), ForeignKey("albums.id", ondelete="CASCADE"), nullable=False)
    storage_path: Mapped[str] = mapped_column(String(512), nullable=False)
    original_storage_path: Mapped[str | None] = mapped_column(String(512))
    bounding_box: Mapped[dict] = mapped_column(JSONType, nullable=False)
    segmentation_mask: Mapped[dict | None] = mapped_column(JSONType)
    segmentation_confidence: Mapped[float | None] = mapped_column(Float)
    aspect_ratio: Mapped[float | None] = mapped_column(Float)
    geometry_valid: Mapped[bool] = mapped_column(Boolean, default=True, server_default=true(), nullable=False)
    phash: Mapped[str | None] = mapped_column(String(64))
    is_duplicate_of: Mapped[UUID | None] = mapped_column(
        GUID(),
        ForeignKey("extracted_photos.id", ondelete="SET NULL"),
    )
    enhancement_applied: Mapped[dict] = mapped_column(JSONType, default=dict, server_default=text("'{}'"), nullable=False)
    status: Mapped[PhotoStatus] = mapped_column(
        enum_column(PhotoStatus, "photo_status"),
        default=PhotoStatus.pending,
        server_default=PhotoStatus.pending.value,
        nullable=False,
    )

    page: Mapped[Page] = relationship(back_populates="photos")
    album: Mapped[Album] = relationship(back_populates="photos")
    duplicate_source: Mapped["ExtractedPhoto | None"] = relationship(remote_side=[id])
    ocr_results: Mapped[list["OCRResult"]] = relationship(back_populates="photo")


class OCRResult(Base, TimestampMixin):
    __tablename__ = "ocr_results"
    __table_args__ = (
        Index("ix_ocr_results_page_id_photo_id", "page_id", "photo_id"),
    )

    id: Mapped[UUID] = mapped_column(GUID(), primary_key=True, default=uuid4)
    page_id: Mapped[UUID] = mapped_column(GUID(), ForeignKey("pages.id", ondelete="CASCADE"), nullable=False)
    photo_id: Mapped[UUID | None] = mapped_column(GUID(), ForeignKey("extracted_photos.id", ondelete="SET NULL"))
    text_content: Mapped[str] = mapped_column(Text, nullable=False)
    text_type: Mapped[OCRTextType | None] = mapped_column(enum_column(OCRTextType, "ocr_text_type"))
    bounding_box: Mapped[dict | None] = mapped_column(JSONType)
    ocr_engine: Mapped[str] = mapped_column(String(50), nullable=False)
    confidence: Mapped[float | None] = mapped_column(Float)
    spatial_classification_reason: Mapped[str | None] = mapped_column(Text)
    is_verified: Mapped[bool] = mapped_column(Boolean, default=False, server_default=false(), nullable=False)

    page: Mapped[Page] = relationship(back_populates="ocr_results")
    photo: Mapped[ExtractedPhoto | None] = relationship(back_populates="ocr_results")


class AuditLog(Base, TimestampMixin):
    __tablename__ = "audit_logs"
    __table_args__ = (
        Index("ix_audit_logs_entity_type_entity_id_created_at", "entity_type", "entity_id", "created_at"),
    )

    id: Mapped[int] = mapped_column(BigInteger().with_variant(Integer, "sqlite"), primary_key=True, autoincrement=True)
    entity_type: Mapped[str] = mapped_column(String(50), nullable=False)
    entity_id: Mapped[UUID] = mapped_column(GUID(), nullable=False)
    action: Mapped[str] = mapped_column(String(100), nullable=False)
    details: Mapped[dict] = mapped_column(JSONType, default=dict, server_default=text("'{}'"), nullable=False)


class Tenant(Base, TimestampMixin):
    __tablename__ = "tenants"

    id: Mapped[UUID] = mapped_column(GUID(), primary_key=True, default=uuid4)
    name: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, server_default=true(), nullable=False)

    albums: Mapped[list[Album]] = relationship(back_populates="tenant")
    api_keys: Mapped[list["ApiKey"]] = relationship(back_populates="tenant", cascade="all, delete-orphan")


class ApiKey(Base, TimestampMixin):
    __tablename__ = "api_keys"
    __table_args__ = (
        UniqueConstraint("key_hash", name="uq_api_keys_key_hash"),
        Index("ix_api_keys_tenant_id_active", "tenant_id", "is_active"),
    )

    id: Mapped[UUID] = mapped_column(GUID(), primary_key=True, default=uuid4)
    tenant_id: Mapped[UUID] = mapped_column(GUID(), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False)
    label: Mapped[str] = mapped_column(String(255), nullable=False)
    key_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    role: Mapped[ApiRole] = mapped_column(enum_column(ApiRole, "api_role"), nullable=False, default=ApiRole.viewer)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, server_default=true(), nullable=False)

    tenant: Mapped[Tenant] = relationship(back_populates="api_keys")


class ProcessingRun(Base, TimestampMixin):
    """Immutable configuration/input identity for one page-processing run."""

    __tablename__ = "processing_runs"
    __table_args__ = (
        UniqueConstraint("page_id", "input_sha256", "pipeline_version", "config_fingerprint", name="uq_processing_run_identity"),
        Index("ix_processing_runs_page_id_status", "page_id", "status"),
    )

    id: Mapped[UUID] = mapped_column(GUID(), primary_key=True, default=uuid4)
    page_id: Mapped[UUID] = mapped_column(GUID(), ForeignKey("pages.id", ondelete="CASCADE"), nullable=False)
    input_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    pipeline_version: Mapped[str] = mapped_column(String(120), nullable=False)
    config_fingerprint: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[RunStatus] = mapped_column(enum_column(RunStatus, "run_status"), default=RunStatus.pending, server_default=RunStatus.pending.value, nullable=False)
    manifest: Mapped[dict] = mapped_column(JSONType, default=dict, server_default=text("'{}'"), nullable=False)

    page: Mapped[Page] = relationship(back_populates="processing_runs")
    stages: Mapped[list["StageRun"]] = relationship(back_populates="run", cascade="all, delete-orphan")
    assets: Mapped[list["Asset"]] = relationship(back_populates="run", cascade="all, delete-orphan")


class StageRun(Base, TimestampMixin):
    """Append-only stage evidence; retries are represented by ``attempt``."""

    __tablename__ = "stage_runs"
    __table_args__ = (
        UniqueConstraint("processing_run_id", "stage", "attempt", name="uq_stage_run_attempt"),
        Index("ix_stage_runs_run_stage", "processing_run_id", "stage"),
    )

    id: Mapped[UUID] = mapped_column(GUID(), primary_key=True, default=uuid4)
    processing_run_id: Mapped[UUID] = mapped_column(GUID(), ForeignKey("processing_runs.id", ondelete="CASCADE"), nullable=False)
    stage: Mapped[str] = mapped_column(String(80), nullable=False)
    attempt: Mapped[int] = mapped_column(Integer, nullable=False, default=1, server_default=text("1"))
    status: Mapped[RunStatus] = mapped_column(enum_column(RunStatus, "stage_run_status"), nullable=False, default=RunStatus.pending)
    input_asset_sha256: Mapped[str | None] = mapped_column(String(64))
    output_asset_sha256: Mapped[str | None] = mapped_column(String(64))
    details: Mapped[dict] = mapped_column(JSONType, default=dict, server_default=text("'{}'"), nullable=False)

    run: Mapped[ProcessingRun] = relationship(back_populates="stages")
    assets: Mapped[list["Asset"]] = relationship(back_populates="stage", cascade="all, delete-orphan")


class Asset(Base, TimestampMixin):
    """Immutable file artifact produced or consumed by a processing run."""

    __tablename__ = "assets"
    __table_args__ = (
        UniqueConstraint("storage_path", name="uq_assets_storage_path"),
        Index("ix_assets_content_sha256", "content_sha256"),
        Index("ix_assets_processing_run_id", "processing_run_id"),
    )

    id: Mapped[UUID] = mapped_column(GUID(), primary_key=True, default=uuid4)
    processing_run_id: Mapped[UUID] = mapped_column(GUID(), ForeignKey("processing_runs.id", ondelete="CASCADE"), nullable=False)
    stage_run_id: Mapped[UUID | None] = mapped_column(GUID(), ForeignKey("stage_runs.id", ondelete="SET NULL"))
    parent_asset_id: Mapped[UUID | None] = mapped_column(GUID(), ForeignKey("assets.id", ondelete="SET NULL"))
    role: Mapped[str] = mapped_column(String(80), nullable=False)
    storage_path: Mapped[str] = mapped_column(String(1024), nullable=False)
    content_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    media_type: Mapped[str] = mapped_column(String(100), nullable=False, default="image/jpeg", server_default="image/jpeg")
    width: Mapped[int | None] = mapped_column(Integer)
    height: Mapped[int | None] = mapped_column(Integer)
    asset_metadata: Mapped[dict] = mapped_column("metadata", JSONType, default=dict, server_default=text("'{}'"), nullable=False)

    run: Mapped[ProcessingRun] = relationship(back_populates="assets")
    stage: Mapped[StageRun | None] = relationship(back_populates="assets")
    parent_asset: Mapped["Asset | None"] = relationship(remote_side=[id])


class ReviewEvent(Base, TimestampMixin):
    """Append-only mobile review decision; corrections create a new event."""

    __tablename__ = "review_events"
    __table_args__ = (
        UniqueConstraint("idempotency_key", name="uq_review_events_idempotency_key"),
        Index("ix_review_events_task_created", "task_type", "task_id", "created_at"),
    )

    id: Mapped[UUID] = mapped_column(GUID(), primary_key=True, default=uuid4)
    task_type: Mapped[str] = mapped_column(String(80), nullable=False)
    task_id: Mapped[UUID] = mapped_column(GUID(), nullable=False)
    answer: Mapped[str] = mapped_column(String(120), nullable=False)
    reason_code: Mapped[str | None] = mapped_column(String(120))
    idempotency_key: Mapped[str] = mapped_column(String(64), nullable=False)
    reviewer: Mapped[str] = mapped_column(String(120), default="local-mobile", nullable=False)
    event_metadata: Mapped[dict] = mapped_column("metadata", JSONType, default=dict, server_default=text("'{}'"), nullable=False)
