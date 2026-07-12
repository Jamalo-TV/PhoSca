"""add immutable processing-run and stage evidence

Revision ID: 0003_processing_runs
Revises: 0002_album_scoped_page_hash
Create Date: 2026-07-11
"""

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = "0003_processing_runs"
down_revision: str | None = "0002_album_scoped_page_hash"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    run_status = sa.Enum("pending", "running", "review_needed", "completed", "failed", name="run_status")
    stage_status = sa.Enum("pending", "running", "review_needed", "completed", "failed", name="stage_run_status")
    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        run_status.create(bind, checkfirst=True)
        stage_status.create(bind, checkfirst=True)
    json_type = sa.JSON().with_variant(postgresql.JSONB, "postgresql")
    op.create_table(
        "processing_runs",
        sa.Column("id", sa.CHAR(length=36), primary_key=True, nullable=False),
        sa.Column("page_id", sa.CHAR(length=36), sa.ForeignKey("pages.id", ondelete="CASCADE"), nullable=False),
        sa.Column("input_sha256", sa.String(length=64), nullable=False),
        sa.Column("pipeline_version", sa.String(length=120), nullable=False),
        sa.Column("config_fingerprint", sa.String(length=64), nullable=False),
        sa.Column("status", run_status, nullable=False, server_default="pending"),
        sa.Column("manifest", json_type, nullable=False, server_default=sa.text("'{}'")),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.UniqueConstraint("page_id", "input_sha256", "pipeline_version", "config_fingerprint", name="uq_processing_run_identity"),
    )
    op.create_index("ix_processing_runs_page_id_status", "processing_runs", ["page_id", "status"])
    op.create_table(
        "stage_runs",
        sa.Column("id", sa.CHAR(length=36), primary_key=True, nullable=False),
        sa.Column("processing_run_id", sa.CHAR(length=36), sa.ForeignKey("processing_runs.id", ondelete="CASCADE"), nullable=False),
        sa.Column("stage", sa.String(length=80), nullable=False),
        sa.Column("attempt", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("status", stage_status, nullable=False, server_default="pending"),
        sa.Column("input_asset_sha256", sa.String(length=64)),
        sa.Column("output_asset_sha256", sa.String(length=64)),
        sa.Column("details", json_type, nullable=False, server_default=sa.text("'{}'")),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.UniqueConstraint("processing_run_id", "stage", "attempt", name="uq_stage_run_attempt"),
    )
    op.create_index("ix_stage_runs_run_stage", "stage_runs", ["processing_run_id", "stage"])


def downgrade() -> None:
    op.drop_index("ix_stage_runs_run_stage", table_name="stage_runs")
    op.drop_table("stage_runs")
    op.drop_index("ix_processing_runs_page_id_status", table_name="processing_runs")
    op.drop_table("processing_runs")
    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        sa.Enum(name="stage_run_status").drop(bind, checkfirst=True)
        sa.Enum(name="run_status").drop(bind, checkfirst=True)
