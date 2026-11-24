"""Match PostgreSQL review-event columns to the persisted model.

Revision ID: 0008_postgresql_review_types
Revises: 0007_mobile_review_events
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "0008_postgresql_review_types"
down_revision = "0007_mobile_review_events"
branch_labels = None
depends_on = None


def upgrade() -> None:
    if op.get_bind().dialect.name != "postgresql":
        return

    for column in ("id", "task_id"):
        op.alter_column(
            "review_events", column, existing_type=sa.CHAR(36),
            type_=postgresql.UUID(), postgresql_using=f"{column}::uuid",
        )
    op.alter_column(
        "review_events", "metadata", existing_type=sa.JSON(),
        type_=postgresql.JSONB(), postgresql_using="metadata::jsonb",
    )


def downgrade() -> None:
    if op.get_bind().dialect.name != "postgresql":
        return

    op.alter_column(
        "review_events", "metadata", existing_type=postgresql.JSONB(),
        type_=sa.JSON(), postgresql_using="metadata::json",
    )
    for column in ("id", "task_id"):
        op.alter_column(
            "review_events", column, existing_type=postgresql.UUID(),
            type_=sa.CHAR(36), postgresql_using=f"{column}::text",
        )
