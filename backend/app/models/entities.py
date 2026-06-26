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


class OCRTextType(str, Enum):
    caption = "caption"
    directory_name = "directory_name"
    unknown = "unknown"


class Album(Base, TimestampMixin):
    __tablename__ = "albums"

    id: Mapped[UUID] = mapped_column(GUID(), primary_key=True, default=uuid4)
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
