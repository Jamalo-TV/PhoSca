from __future__ import annotations

import asyncio
import hashlib
import os
import shutil
from pathlib import Path

from sqlalchemy import func, select

os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///./real_data_smoke.db")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/0")
os.environ.setdefault("STORAGE_PATH", str(Path("storage").resolve()))
os.environ.setdefault("ENABLE_CLASSICAL_SEGMENTATION_FALLBACK", "true")

import sys

sys.path.insert(0, str(Path("backend").resolve()))

from app.config import get_settings
from app.database import Base, get_engine, get_session_factory
from app.models import Album, ExtractedPhoto, Page
from app.services.pipeline import process_page_pipeline


async def main() -> None:
    settings = get_settings()
    engine = get_engine()
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.drop_all)
        await connection.run_sync(Base.metadata.create_all)

    source_dir = Path(os.environ.get("SMOKE_SOURCE_DIR", "data/raw_album_pages"))
    if not source_dir.exists() or not any(source_dir.glob("*.jpg")):
        source_dir = Path("PHOTOALBUM")
    source_images = sorted(source_dir.glob("*.jpg"))
    limit = int(os.environ.get("SMOKE_LIMIT", "0"))
    if limit > 0:
        source_images = source_images[:limit]
    async with get_session_factory()() as session:
        album = Album(name="real-data-smoke", description="Local smoke run over supplied album pages")
        session.add(album)
        await session.flush()
        upload_dir = settings.storage_path / "uploads" / str(album.id)
        upload_dir.mkdir(parents=True, exist_ok=True)
        page_ids = []
        for image in source_images:
            destination = upload_dir / image.name
            shutil.copy2(image, destination)
            page = Page(
                album_id=album.id,
                original_filename=image.name,
                storage_path=str(destination),
                file_size_bytes=destination.stat().st_size,
                file_hash_sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),
            )
            session.add(page)
            await session.flush()
            page_ids.append(page.id)
        album.total_pages = len(page_ids)
        await session.commit()

    results = []
    for page_id in page_ids:
        results.append(await process_page_pipeline(page_id, settings))

    async with get_session_factory()() as session:
        photo_count = await session.scalar(select(func.count(ExtractedPhoto.id)))

    completed = sum(1 for result in results if result["status"] == "completed")
    review = sum(1 for result in results if result["status"] == "review_needed")
    failed = [result for result in results if result["status"] == "failed"]
    print(
        {
            "album": "real-data-smoke",
            "source_dir": str(source_dir),
            "pages": len(results),
            "completed": completed,
            "review_needed": review,
            "failed": failed,
            "photos_extracted": photo_count,
        }
    )


if __name__ == "__main__":
    asyncio.run(main())
