"""add tenant ownership and API key roles

Revision ID: 0005_tenants_and_api_keys
Revises: 0004_assets
Create Date: 2026-07-11
"""

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa


revision: str = "0005_tenants_and_api_keys"
down_revision: str | None = "0004_assets"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

LEGACY_TENANT_ID = "00000000-0000-0000-0000-000000000001"


def upgrade() -> None:
    role = sa.Enum("viewer", "editor", "admin", name="api_role")
    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        role.create(bind, checkfirst=True)
    op.create_table(
        "tenants",
        sa.Column("id", sa.CHAR(length=36), primary_key=True, nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False, unique=True),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )
    op.execute(sa.text("INSERT INTO tenants (id, name, is_active) VALUES (:id, :name, :active)").bindparams(id=LEGACY_TENANT_ID, name="legacy-unscoped", active=True))
    with op.batch_alter_table("albums") as batch:
        batch.add_column(sa.Column("tenant_id", sa.CHAR(length=36), nullable=True))
    op.execute(sa.text("UPDATE albums SET tenant_id = :id WHERE tenant_id IS NULL").bindparams(id=LEGACY_TENANT_ID))
    with op.batch_alter_table("albums") as batch:
        batch.create_foreign_key("fk_albums_tenant_id", "tenants", ["tenant_id"], ["id"], ondelete="RESTRICT")
    op.create_index("ix_albums_tenant_id", "albums", ["tenant_id"])
    op.create_table(
        "api_keys",
        sa.Column("id", sa.CHAR(length=36), primary_key=True, nullable=False),
        sa.Column("tenant_id", sa.CHAR(length=36), sa.ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False),
        sa.Column("label", sa.String(length=255), nullable=False),
        sa.Column("key_hash", sa.String(length=64), nullable=False),
        sa.Column("role", role, nullable=False, server_default="viewer"),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.UniqueConstraint("key_hash", name="uq_api_keys_key_hash"),
    )
    op.create_index("ix_api_keys_tenant_id_active", "api_keys", ["tenant_id", "is_active"])


def downgrade() -> None:
    op.drop_index("ix_api_keys_tenant_id_active", table_name="api_keys")
    op.drop_table("api_keys")
    op.drop_index("ix_albums_tenant_id", table_name="albums")
    with op.batch_alter_table("albums") as batch:
        batch.drop_constraint("fk_albums_tenant_id", type_="foreignkey")
        batch.drop_column("tenant_id")
    op.drop_table("tenants")
    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        sa.Enum(name="api_role").drop(bind, checkfirst=True)
