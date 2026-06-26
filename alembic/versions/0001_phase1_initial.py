"""phase 1 initial schema

Revision ID: 0001_phase1_initial
Revises:
Create Date: 2026-06-20
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0001_phase1_initial"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


album_status = sa.Enum("pending", "processing", "completed", "failed", name="album_status")
page_status = sa.Enum("uploaded", "queued", "processing", "review_needed", "completed", "failed", name="page_status")
photo_status = sa.Enum("pending", "review_needed", "completed", "failed", name="photo_status")
ocr_text_type = sa.Enum("caption", "directory_name", "unknown", name="ocr_text_type")


def timestamps() -> list[sa.Column]:
    return [
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    ]


def upgrade() -> None:
    op.create_table(
        "albums",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("directory_path", sa.String(length=512), nullable=True),
        sa.Column("status", album_status, server_default="pending", nullable=False),
        sa.Column("total_pages", sa.Integer(), server_default=sa.text("0"), nullable=False),
        sa.Column("processed_pages", sa.Integer(), server_default=sa.text("0"), nullable=False),
        *timestamps(),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_table(
        "pages",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("album_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("original_filename", sa.String(length=255), nullable=False),
        sa.Column("storage_path", sa.String(length=512), nullable=False),
        sa.Column("file_size_bytes", sa.BigInteger(), nullable=True),
        sa.Column("file_hash_sha256", sa.String(length=64), nullable=False),
        sa.Column("blur_score", sa.Float(), nullable=True),
        sa.Column("status", page_status, server_default="uploaded", nullable=False),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("processing_metadata", postgresql.JSONB(astext_type=sa.Text()), server_default=sa.text("'{}'::jsonb"), nullable=False),
        *timestamps(),
        sa.ForeignKeyConstraint(["album_id"], ["albums.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("album_id", "file_hash_sha256", name="uq_pages_album_file_hash_sha256"),
    )
    op.create_index("ix_pages_album_id_status", "pages", ["album_id", "status"])
    op.create_index("ix_pages_file_hash_sha256", "pages", ["file_hash_sha256"])

    op.create_table(
        "extracted_photos",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("page_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("album_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("storage_path", sa.String(length=512), nullable=False),
        sa.Column("original_storage_path", sa.String(length=512), nullable=True),
        sa.Column("bounding_box", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("segmentation_mask", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("segmentation_confidence", sa.Float(), nullable=True),
        sa.Column("aspect_ratio", sa.Float(), nullable=True),
        sa.Column("geometry_valid", sa.Boolean(), server_default=sa.true(), nullable=False),
        sa.Column("phash", sa.String(length=64), nullable=True),
        sa.Column("is_duplicate_of", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("enhancement_applied", postgresql.JSONB(astext_type=sa.Text()), server_default=sa.text("'{}'::jsonb"), nullable=False),
        sa.Column("status", photo_status, server_default="pending", nullable=False),
        *timestamps(),
        sa.ForeignKeyConstraint(["album_id"], ["albums.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["is_duplicate_of"], ["extracted_photos.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["page_id"], ["pages.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_extracted_photos_page_id", "extracted_photos", ["page_id"])
    op.create_index("ix_extracted_photos_phash", "extracted_photos", ["phash"])

    op.create_table(
        "ocr_results",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("page_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("photo_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("text_content", sa.Text(), nullable=False),
        sa.Column("text_type", ocr_text_type, nullable=True),
        sa.Column("bounding_box", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("ocr_engine", sa.String(length=50), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=True),
        sa.Column("spatial_classification_reason", sa.Text(), nullable=True),
        sa.Column("is_verified", sa.Boolean(), server_default=sa.false(), nullable=False),
        *timestamps(),
        sa.ForeignKeyConstraint(["page_id"], ["pages.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["photo_id"], ["extracted_photos.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_ocr_results_page_id_photo_id", "ocr_results", ["page_id", "photo_id"])
    op.create_index(
        "ix_ocr_results_text_content_fts",
        "ocr_results",
        [sa.text("to_tsvector('english', text_content)")],
        postgresql_using="gin",
    )

    op.create_table(
        "audit_logs",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("entity_type", sa.String(length=50), nullable=False),
        sa.Column("entity_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("action", sa.String(length=100), nullable=False),
        sa.Column("details", postgresql.JSONB(astext_type=sa.Text()), server_default=sa.text("'{}'::jsonb"), nullable=False),
        *timestamps(),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_audit_logs_entity_type_entity_id_created_at",
        "audit_logs",
        ["entity_type", "entity_id", "created_at"],
    )


def downgrade() -> None:
    op.drop_index("ix_audit_logs_entity_type_entity_id_created_at", table_name="audit_logs")
    op.drop_table("audit_logs")
    op.drop_index("ix_ocr_results_text_content_fts", table_name="ocr_results")
    op.drop_index("ix_ocr_results_page_id_photo_id", table_name="ocr_results")
    op.drop_table("ocr_results")
    op.drop_index("ix_extracted_photos_phash", table_name="extracted_photos")
    op.drop_index("ix_extracted_photos_page_id", table_name="extracted_photos")
    op.drop_table("extracted_photos")
    op.drop_index("ix_pages_file_hash_sha256", table_name="pages")
    op.drop_index("ix_pages_album_id_status", table_name="pages")
    op.drop_table("pages")
    op.drop_table("albums")
    ocr_text_type.drop(op.get_bind(), checkfirst=True)
    photo_status.drop(op.get_bind(), checkfirst=True)
    page_status.drop(op.get_bind(), checkfirst=True)
    album_status.drop(op.get_bind(), checkfirst=True)
