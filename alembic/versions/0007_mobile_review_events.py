"""Add append-only mobile review events.

Revision ID: 0007_mobile_review_events
Revises: 0006_sqlite_audit_log_identity
"""

from alembic import op
import sqlalchemy as sa


revision = "0007_mobile_review_events"
down_revision = "0006_sqlite_audit_log_identity"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "review_events",
        sa.Column("id", sa.CHAR(36), nullable=False),
        sa.Column("task_type", sa.String(80), nullable=False),
        sa.Column("task_id", sa.CHAR(36), nullable=False),
        sa.Column("answer", sa.String(120), nullable=False),
        sa.Column("reason_code", sa.String(120), nullable=True),
        sa.Column("idempotency_key", sa.String(64), nullable=False),
        sa.Column("reviewer", sa.String(120), nullable=False),
        sa.Column("metadata", sa.JSON(), nullable=False, server_default=sa.text("'{}'")),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("idempotency_key", name="uq_review_events_idempotency_key"),
    )
    op.create_index("ix_review_events_task_created", "review_events", ["task_type", "task_id", "created_at"])


def downgrade() -> None:
    op.drop_index("ix_review_events_task_created", table_name="review_events")
    op.drop_table("review_events")
