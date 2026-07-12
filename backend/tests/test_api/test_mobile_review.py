from __future__ import annotations

import hashlib
from uuid import uuid4

import cv2
import numpy as np


async def _seed_review_data():
    from app.config import get_settings
    from app.database import get_session_factory
    from app.models import Album, ExtractedPhoto, Page, PageStatus, PhotoStatus

    settings = get_settings()
    image = np.full((120, 160, 3), 120, dtype=np.uint8)
    page_path = settings.storage_path / "uploads" / "review-page.jpg"
    original_path = settings.storage_path / "processed" / "review-original.jpg"
    enhanced_path = settings.storage_path / "processed" / "review-enhanced.jpg"
    for path in (page_path, original_path, enhanced_path):
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), image)
    album = Album(name="Mobile review")
    async with get_session_factory()() as session:
        session.add(album)
        await session.flush()
        page = Page(
            album_id=album.id,
            original_filename=page_path.name,
            storage_path=str(page_path),
            file_hash_sha256=hashlib.sha256(page_path.read_bytes()).hexdigest(),
            status=PageStatus.review_needed,
        )
        session.add(page)
        await session.flush()
        photo = ExtractedPhoto(
            page_id=page.id,
            album_id=album.id,
            storage_path=str(original_path),
            original_storage_path=str(original_path),
            bounding_box={"x1": 0.1, "y1": 0.2, "x2": 0.8, "y2": 0.9},
            segmentation_mask={"polygon": [{"x": 0.1, "y": 0.2}, {"x": 0.8, "y": 0.2}, {"x": 0.8, "y": 0.9}, {"x": 0.1, "y": 0.9}]},
            segmentation_confidence=0.72,
            aspect_ratio=1.2,
            geometry_valid=True,
            status=PhotoStatus.review_needed,
            enhancement_applied={
                "variants": {
                    "original": {"path": str(original_path)},
                    "enhanced": {"path": str(enhanced_path)},
                }
            },
        )
        session.add(photo)
        await session.commit()
        return album.id, page.id, photo.id


async def test_mobile_queues_expose_overlays_and_blind_pair(client) -> None:
    album_id, page_id, photo_id = await _seed_review_data()
    segmentation = (await client.get(f"/api/v1/review/queue?album_id={album_id}&type=segmentation")).json()
    geometry = (await client.get(f"/api/v1/review/queue?album_id={album_id}&type=geometry")).json()
    enhancement = (await client.get(f"/api/v1/review/queue?album_id={album_id}&type=enhancement")).json()
    assert segmentation[0]["id"] == str(page_id)
    assert segmentation[0]["payload"]["detection_count"] == 1
    assert geometry[0]["payload"]["photo_id"] == str(photo_id)
    assert enhancement[0]["payload"]["candidate_name"] == "enhanced"
    assert "candidate_url" in enhancement[0]["payload"]


async def test_mobile_review_event_is_idempotent_and_removes_task(client) -> None:
    album_id, page_id, _ = await _seed_review_data()
    key = uuid4().hex
    payload = {"task_type": "segmentation", "task_id": str(page_id), "answer": "correct", "idempotency_key": key, "metadata": {"ui": "test"}}
    first = await client.post("/api/v1/review/events", json=payload)
    second = await client.post("/api/v1/review/events", json=payload)
    assert first.status_code == 201
    assert second.status_code == 201
    assert second.json()["duplicate"] is True
    remaining = (await client.get(f"/api/v1/review/queue?album_id={album_id}&type=segmentation")).json()
    assert all(item["id"] != str(page_id) for item in remaining)


async def test_mobile_review_rejects_answer_from_wrong_task_contract(client) -> None:
    _, page_id, _ = await _seed_review_data()
    response = await client.post("/api/v1/review/events", json={
        "task_type": "segmentation",
        "task_id": str(page_id),
        "answer": "left",
        "idempotency_key": uuid4().hex,
    })
    assert response.status_code == 422
