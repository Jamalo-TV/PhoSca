"""Verified deletion of an album and every locally managed derivative."""
from __future__ import annotations

import shutil
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings
from app.models import Album, Page


def _inside(root: Path, target: Path) -> bool:
    try:
        target.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _remove_tree(root: Path, target: Path) -> None:
    if target.exists() and _inside(root, target):
        shutil.rmtree(target)


async def delete_album_with_assets(session: AsyncSession, album: Album, settings: Settings) -> dict:
    """Delete database rows and all album-scoped managed storage trees.

    Only paths deterministically derived from the configured storage root are
    removed; arbitrary database paths are never passed to recursive deletion.
    """
    root = settings.storage_path.resolve()
    pages = list(await session.scalars(select(Page).where(Page.album_id == album.id)))
    targets = [
        root / "uploads" / str(album.id),
        root / "processed" / str(album.id),
        *(root / "failed" / str(page.id) for page in pages),
    ]
    await session.delete(album)
    await session.commit()
    deleted: list[str] = []
    for target in targets:
        if target.exists() and _inside(root, target):
            _remove_tree(root, target)
            deleted.append(str(target))
    return {"album_id": str(album.id), "deleted_paths": deleted, "pages": len(pages)}
