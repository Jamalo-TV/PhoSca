"""scope page hash uniqueness to albums

Revision ID: 0002_album_scoped_page_hash
Revises: 0001_phase1_initial
Create Date: 2026-06-26
"""

from collections.abc import Sequence

from alembic import op

revision: str = "0002_album_scoped_page_hash"
down_revision: str | None = "0001_phase1_initial"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.drop_constraint("pages_file_hash_sha256_key", "pages", type_="unique")
    op.create_unique_constraint("uq_pages_album_file_hash_sha256", "pages", ["album_id", "file_hash_sha256"])


def downgrade() -> None:
    op.drop_constraint("uq_pages_album_file_hash_sha256", "pages", type_="unique")
    op.create_unique_constraint("pages_file_hash_sha256_key", "pages", ["file_hash_sha256"])
