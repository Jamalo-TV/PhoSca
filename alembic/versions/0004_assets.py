"""add immutable asset registry

Revision ID: 0004_assets
Revises: 0003_processing_runs
Create Date: 2026-07-11
"""

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = "0004_assets"
down_revision: str | None = "0003_processing_runs"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    json_type = sa.JSON().with_variant(postgresql.JSONB, "postgresql")
    op.create_table(
        "assets",
        sa.Column("id", sa.CHAR(length=36), primary_key=True, nullable=False),
        sa.Column("processing_run_id", sa.CHAR(length=36), sa.ForeignKey("processing_runs.id", ondelete="CASCADE"), nullable=False),
        sa.Column("stage_run_id", sa.CHAR(length=36), sa.ForeignKey("stage_runs.id", ondelete="SET NULL")),
        sa.Column("parent_asset_id", sa.CHAR(length=36), sa.ForeignKey("assets.id", ondelete="SET NULL")),
        sa.Column("role", sa.String(length=80), nullable=False),
        sa.Column("storage_path", sa.String(length=1024), nullable=False),
        sa.Column("content_sha256", sa.String(length=64), nullable=False),
        sa.Column("media_type", sa.String(length=100), nullable=False, server_default="image/jpeg"),
        sa.Column("width", sa.Integer()),
        sa.Column("height", sa.Integer()),
        sa.Column("metadata", json_type, nullable=False, server_default=sa.text("'{}'")),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.UniqueConstraint("storage_path", name="uq_assets_storage_path"),
    )
    op.create_index("ix_assets_content_sha256", "assets", ["content_sha256"])
    op.create_index("ix_assets_processing_run_id", "assets", ["processing_run_id"])


def downgrade() -> None:
    op.drop_index("ix_assets_processing_run_id", table_name="assets")
    op.drop_index("ix_assets_content_sha256", table_name="assets")
    op.drop_table("assets")
