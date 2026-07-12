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
    # The initial migration already defines the album-scoped constraint.  The
    # historical migration is retained as a no-op so both SQLite and existing
    # PostgreSQL deployments can advance deterministically.
    return None


def downgrade() -> None:
    return None
