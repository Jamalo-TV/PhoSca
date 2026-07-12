"""Use SQLite's native INTEGER primary key for audit log identities.

SQLite only auto-generates a primary-key value for a column declared exactly
as ``INTEGER PRIMARY KEY``.  The original BigInteger declaration therefore
failed for every mutation audit entry when the local SQLite setup was used.
PostgreSQL keeps its existing BIGINT schema.

Revision ID: 0006_sqlite_audit_log_identity
Revises: 0005_tenants_and_api_keys
"""

from alembic import op
import sqlalchemy as sa


revision = "0006_sqlite_audit_log_identity"
down_revision = "0005_tenants_and_api_keys"
branch_labels = None
depends_on = None


def upgrade() -> None:
    if op.get_bind().dialect.name != "sqlite":
        return

    # SQLite cannot alter a column type in place.  Alembic recreates the table
    # and preserves all existing audit records and indexes.
    with op.batch_alter_table("audit_logs", recreate="always") as batch_op:
        batch_op.alter_column(
            "id",
            existing_type=sa.BigInteger(),
            type_=sa.Integer(),
            existing_nullable=False,
            autoincrement=True,
        )


def downgrade() -> None:
    if op.get_bind().dialect.name != "sqlite":
        return

    with op.batch_alter_table("audit_logs", recreate="always") as batch_op:
        batch_op.alter_column(
            "id",
            existing_type=sa.Integer(),
            type_=sa.BigInteger(),
            existing_nullable=False,
            autoincrement=True,
        )
