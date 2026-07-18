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
            segmentation_mask={
                "source": "classical_boundary",
                "polygon": [
                    {"x": 0.1, "y": 0.2},
                    {"x": 0.45, "y": 0.18},
                    {"x": 0.8, "y": 0.2},
                    {"x": 0.82, "y": 0.55},
                    {"x": 0.8, "y": 0.9},
                    {"x": 0.1, "y": 0.9},
                ],
                "warp_quad": [{"x": 0.1, "y": 0.2}, {"x": 0.8, "y": 0.2}, {"x": 0.8, "y": 0.9}, {"x": 0.1, "y": 0.9}],
            },
            segmentation_confidence=0.72,
            aspect_ratio=1.2,
            geometry_valid=True,
            status=PhotoStatus.review_needed,
            enhancement_applied={
                "variants": {
                    "original": {"path": str(original_path)},
                    "enhanced": {"path": str(enhanced_path), "metrics": {"source_similarity": 0.91}},
                },
                "correction": {"orientation": {"rotation_degrees": 0}, "perspective_confidence": 0.84},
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
    assert len(geometry[0]["payload"]["visible_polygon"]) == 6
    assert len(geometry[0]["payload"]["warp_quad"]) == 4
    assert geometry[0]["payload"]["confidence_components"]["perspective_confidence"] == 0.84
    assert geometry[0]["payload"]["variants"]["enhanced"]["metrics"]["source_similarity"] == 0.91
    assert enhancement[0]["payload"]["candidate_name"] == "enhanced"
    assert "candidate_url" in enhancement[0]["payload"]


async def test_enhancement_review_persists_semantic_variant_selection(client) -> None:
    _, _, photo_id = await _seed_review_data()
    response = await client.post(
        "/api/v1/review/events",
        json={
            "task_type": "enhancement",
            "task_id": str(photo_id),
            "answer": "left",
            "idempotency_key": uuid4().hex,
            "metadata": {"blind_choice": "candidate", "candidate_name": "enhanced"},
        },
    )
    assert response.status_code == 201

    from app.database import get_session_factory
    from app.models import ExtractedPhoto
    from app.pipeline.variants import released_variant_from_photo

    async with get_session_factory()() as session:
        photo = await session.get(ExtractedPhoto, photo_id)
        decision = photo.enhancement_applied["review_decisions"]["enhancement"][-1]
        assert decision["blind_choice"] == "candidate"
        assert photo.enhancement_applied["selected_variant"] == "enhanced"
        assert released_variant_from_photo(photo) == "enhanced"


async def test_enhancement_review_rejects_invalid_semantic_metadata(client) -> None:
    _, _, photo_id = await _seed_review_data()
    missing_choice = await client.post(
        "/api/v1/review/events",
        json={
            "task_type": "enhancement",
            "task_id": str(photo_id),
            "answer": "left",
            "idempotency_key": uuid4().hex,
            "metadata": {"candidate_name": "enhanced"},
        },
    )
    source_as_candidate = await client.post(
        "/api/v1/review/events",
        json={
            "task_type": "enhancement",
            "task_id": str(photo_id),
            "answer": "right",
            "idempotency_key": uuid4().hex,
            "metadata": {"blind_choice": "candidate", "candidate_name": "original"},
        },
    )
    assert missing_choice.status_code == 422
    assert source_as_candidate.status_code == 422


async def test_enhancement_idempotency_key_covers_semantic_metadata(client) -> None:
    _, _, photo_id = await _seed_review_data()
    key = uuid4().hex
    payload = {
        "task_type": "enhancement",
        "task_id": str(photo_id),
        "answer": "left",
        "idempotency_key": key,
        "metadata": {"blind_choice": "candidate", "candidate_name": "enhanced"},
    }
    accepted = await client.post("/api/v1/review/events", json=payload)
    changed_replay = await client.post(
        "/api/v1/review/events",
        json=payload | {"metadata": {"blind_choice": "original", "candidate_name": "enhanced"}},
    )
    assert accepted.status_code == 201
    assert changed_replay.status_code == 409


async def test_mobile_review_event_is_idempotent_and_removes_task(client) -> None:
    album_id, page_id, _ = await _seed_review_data()
    key = uuid4().hex
    payload = {
        "task_type": "segmentation",
        "task_id": str(page_id),
        "answer": "correct",
        "idempotency_key": key,
        "metadata": {"ui": "test"},
    }
    first = await client.post("/api/v1/review/events", json=payload)
    second = await client.post("/api/v1/review/events", json=payload)
    assert first.status_code == 201
    assert second.status_code == 201
    assert second.json()["duplicate"] is True
    remaining = (await client.get(f"/api/v1/review/queue?album_id={album_id}&type=segmentation")).json()
    assert all(item["id"] != str(page_id) for item in remaining)


async def test_mobile_review_rejects_answer_from_wrong_task_contract(client) -> None:
    _, page_id, _ = await _seed_review_data()
    response = await client.post(
        "/api/v1/review/events",
        json={
            "task_type": "segmentation",
            "task_id": str(page_id),
            "answer": "left",
            "idempotency_key": uuid4().hex,
        },
    )
    assert response.status_code == 422


async def test_unsure_geometry_stays_queued_and_event_snapshots_prediction(client) -> None:
    album_id, _, photo_id = await _seed_review_data()
    response = await client.post(
        "/api/v1/review/events",
        json={
            "task_type": "geometry",
            "task_id": str(photo_id),
            "answer": "unsure",
            "reason_code": "ambiguous_border",
            "idempotency_key": uuid4().hex,
            "metadata": {"ui": "test", "difficulty_tags": ["low_contrast"]},
        },
    )
    assert response.status_code == 201
    queued = (await client.get(f"/api/v1/review/queue?album_id={album_id}&type=geometry")).json()
    assert any(item["id"] == str(photo_id) for item in queued)

    from app.database import get_session_factory
    from app.models import ExtractedPhoto, ReviewEvent
    from sqlalchemy import select

    async with get_session_factory()() as session:
        event = await session.scalar(select(ReviewEvent).where(ReviewEvent.id == response.json()["id"]))
        photo = await session.get(ExtractedPhoto, photo_id)
        snapshot = event.event_metadata["snapshot"]
        assert snapshot["page"]["sha256"]
        assert snapshot["photos"][0]["original_prediction"]["bounding_box"] == photo.bounding_box
        assert snapshot["photos"][0]["current_correction"]["perspective_confidence"] == 0.84
        assert event.event_metadata["client"]["difficulty_tags"] == ["low_contrast"]
        assert photo.status.value == "review_needed"
        assert "GEOMETRY_REVIEW_UNSURE" in photo.enhancement_applied["review_reasons"]


async def test_review_event_validates_entity_type_and_idempotency_payload(client) -> None:
    _, page_id, photo_id = await _seed_review_data()
    wrong_entity = await client.post(
        "/api/v1/review/events",
        json={
            "task_type": "geometry",
            "task_id": str(page_id),
            "answer": "correct",
            "idempotency_key": uuid4().hex,
        },
    )
    assert wrong_entity.status_code == 404

    key = uuid4().hex
    accepted = await client.post(
        "/api/v1/review/events",
        json={
            "task_type": "geometry",
            "task_id": str(photo_id),
            "answer": "correct",
            "idempotency_key": key,
        },
    )
    conflict = await client.post(
        "/api/v1/review/events",
        json={
            "task_type": "geometry",
            "task_id": str(photo_id),
            "answer": "unsure",
            "idempotency_key": key,
        },
    )
    assert accepted.status_code == 201
    assert conflict.status_code == 409


async def test_bounding_correction_preserves_dense_boundary_and_audit_provenance(client, monkeypatch) -> None:
    _, _, photo_id = await _seed_review_data()

    async def correction_stub(session, target_photo_id, settings):
        return {"review_required": True}

    monkeypatch.setattr("app.routers.photos.perspective_correction_task_logic", correction_stub)
    visible = [
        {"x": 0.12, "y": 0.22},
        {"x": 0.42, "y": 0.19},
        {"x": 0.76, "y": 0.23},
        {"x": 0.79, "y": 0.58},
        {"x": 0.75, "y": 0.86},
        {"x": 0.13, "y": 0.88},
    ]
    warp = [visible[0], visible[2], visible[4], visible[5]]
    response = await client.patch(
        f"/api/v1/photos/{photo_id}/bounding-box",
        json={
            "bounding_box": {"x1": 0.12, "y1": 0.19, "x2": 0.79, "y2": 0.88},
            "segmentation_mask": {
                "polygon": warp,
                "visible_polygon": visible,
                "warp_quad": warp,
                "source": "manual_quad",
            },
        },
    )
    assert response.status_code == 200
    assert response.json()["segmentation_mask"]["visible_polygon"] == visible

    from app.database import get_session_factory
    from app.models import AuditLog, ExtractedPhoto
    from sqlalchemy import select

    async with get_session_factory()() as session:
        photo = await session.get(ExtractedPhoto, photo_id)
        audit = await session.scalar(
            select(AuditLog).where(AuditLog.entity_id == photo_id, AuditLog.action == "bounding_box_updated")
        )
        provenance = photo.enhancement_applied["review_provenance"]
        assert provenance["original_prediction"]["bounding_box"]["x1"] == 0.1
        assert provenance["correction_revisions"][-1]["after"]["segmentation_mask"]["visible_polygon"] == visible
        assert audit.details["before"]["bounding_box"]["x1"] == 0.1
        assert audit.details["after"]["bounding_box"]["x1"] == 0.12
        assert audit.details["prediction_provenance"]["page_sha256"]
