from __future__ import annotations

import hashlib

import cv2
import numpy as np


async def test_perceptual_dedup_never_crosses_tenant_boundary(client) -> None:
    from app.config import get_settings
    from app.database import get_session_factory
    from app.models import Album, ExtractedPhoto, Page, Tenant
    from app.pipeline.dedup import compute_phash
    from app.services.pipeline import deduplication_task_logic

    settings = get_settings()
    image_path = settings.storage_path / "dedup" / "same.jpg"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image = np.full((96, 128, 3), 40, dtype=np.uint8)
    cv2.circle(image, (64, 48), 28, (220, 160, 80), -1)
    assert cv2.imwrite(str(image_path), image)
    digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
    box = {"x1": 0.0, "y1": 0.0, "x2": 1.0, "y2": 1.0}

    async with get_session_factory()() as session:
        tenant_a, tenant_b = Tenant(name="dedup-a"), Tenant(name="dedup-b")
        session.add_all([tenant_a, tenant_b])
        await session.flush()
        album_a, album_b = Album(name="A", tenant_id=tenant_a.id), Album(name="B", tenant_id=tenant_b.id)
        session.add_all([album_a, album_b])
        await session.flush()
        page_a = Page(
            album_id=album_a.id, original_filename="same.jpg", storage_path=str(image_path), file_hash_sha256=digest
        )
        page_b = Page(
            album_id=album_b.id, original_filename="same.jpg", storage_path=str(image_path), file_hash_sha256=digest
        )
        session.add_all([page_a, page_b])
        await session.flush()
        existing = ExtractedPhoto(
            page_id=page_a.id,
            album_id=album_a.id,
            storage_path=str(image_path),
            original_storage_path=str(image_path),
            bounding_box=box,
            phash=compute_phash(image_path),
        )
        target = ExtractedPhoto(
            page_id=page_b.id,
            album_id=album_b.id,
            storage_path=str(image_path),
            original_storage_path=str(image_path),
            bounding_box=box,
        )
        session.add_all([existing, target])
        await session.commit()
        target_id = target.id

        result = await deduplication_task_logic(session, target_id, settings)
        await session.refresh(target)

    assert result["duplicate_of"] is None
    assert target.is_duplicate_of is None
