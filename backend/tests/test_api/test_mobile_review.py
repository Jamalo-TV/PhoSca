from __future__ import annotations

import hashlib
import json
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np
import pytest


async def _seed_review_data(image=None):
    from app.config import get_settings
    from app.database import get_session_factory
    from app.models import Album, ExtractedPhoto, Page, PageStatus, PhotoStatus

    settings = get_settings()
    if image is None:
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


async def test_legacy_photo_serves_trim_and_keeps_untrimmed_variant(client, monkeypatch) -> None:
    from app.config import get_settings

    monkeypatch.setenv("ENABLE_PRECISE_PHOTO_CROPS", "false")
    get_settings.cache_clear()
    image = np.full((120, 160, 3), 240, dtype=np.uint8)
    image[9:-9, 10:-10] = (50, 85, 150)
    _, _, photo_id = await _seed_review_data(image)
    photo_url = f"/api/v1/photos/{photo_id}"
    correction = await client.patch(
        f"{photo_url}/bounding-box",
        json={"bounding_box": {"x1": 0.0, "y1": 0.0, "x2": 1.0, "y2": 1.0}},
    )
    assert correction.status_code == 200, correction.text
    metadata = correction.json()["enhancement_applied"]
    assert metadata["correction"]["border_trim"]["applied"]
    assert metadata["variants"]["source_corrected_trimmed"]["path"] == correction.json()["original_storage_path"]

    image_url = correction.json()["image_url"]
    displayed = await client.get(image_url)
    archived = await client.get(f"{photo_url}/image?variant=source_corrected_untrimmed")
    assert displayed.status_code == archived.status_code == 200
    crop = cv2.imdecode(np.frombuffer(displayed.content, np.uint8), cv2.IMREAD_COLOR)
    untrimmed = cv2.imdecode(np.frombuffer(archived.content, np.uint8), cv2.IMREAD_COLOR)
    assert crop.shape[0] < untrimmed.shape[0] and crop.shape[1] < untrimmed.shape[1]
    source_shape = metadata["variants"]["source_corrected"]["metrics"]["shape"]
    assert crop.shape[:2] == (source_shape["height"], source_shape["width"])

    monkeypatch.setenv("ENABLE_BORDER_TRIM", "false")
    get_settings.cache_clear()
    retry = await client.post(f"{photo_url}/reprocess")
    assert retry.status_code == 200, retry.text
    assert "source_corrected_trimmed" not in retry.json()["enhancement_applied"]["variants"]
    new_url = retry.json()["image_url"]
    assert new_url != image_url
    variants = (await client.get(f"{photo_url}/variants")).json()["urls"]
    assert new_url.split("?")[1] in variants["original"]
    displayed = await client.get(new_url)
    assert cv2.imdecode(np.frombuffer(displayed.content, np.uint8), cv2.IMREAD_COLOR).shape == untrimmed.shape


async def test_resaved_model_corners_stay_within_the_reviewers_selection(client, monkeypatch) -> None:
    from app.config import get_settings
    from app.pipeline import perspective

    monkeypatch.setenv("ENABLE_PRECISE_PHOTO_CROPS", "false")
    get_settings.cache_clear()
    _, _, photo_id = await _seed_review_data()
    url = f"/api/v1/photos/{photo_id}"
    reference_boxes = []
    find_border = perspective.find_physical_border

    def check_reference(analysis, *, reference_box, protected_regions):
        reference_boxes.append((reference_box, analysis.shape[:2]))
        return find_border(analysis, reference_box=reference_box, protected_regions=protected_regions)

    monkeypatch.setattr(perspective, "find_physical_border", check_reference)
    polygon = [{"x": x, "y": y} for x, y in [(0.1, 0.2), (0.8, 0.2), (0.8, 0.9), (0.1, 0.9)]]
    saved = await client.patch(
        f"{url}/bounding-box",
        json={
            "bounding_box": {"x1": 0.1, "y1": 0.2, "x2": 0.8, "y2": 0.9},
            "segmentation_mask": {"source": "yolo_seg_mask", "polygon": polygon},
        },
    )
    assert saved.status_code == 200, saved.text
    assert (await client.post(f"{url}/reprocess")).status_code == 200
    assert len(reference_boxes) == 2
    assert all(box == (0, 0, w, h) for box, (h, w) in reference_boxes)


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

    async def correction_stub(session, target_photo_id, settings, **kwargs):
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


async def test_precise_crop_variants_follow_explicit_corners_and_preserve_last_good_export(client, monkeypatch) -> None:
    from app.config import get_settings

    monkeypatch.setenv("ENABLE_BORDER_TRIM", "false")
    monkeypatch.setenv("ENABLE_PHOTO_ORIENTATION_CORRECTION", "false")
    get_settings.cache_clear()
    image = np.full((120, 160, 3), 30, dtype=np.uint8)
    image[12:108, 15:145] = 240
    image[20:101, 23:138] = (50, 90, 150)
    _, _, photo_id = await _seed_review_data(image)
    url = f"/api/v1/photos/{photo_id}"
    outer = [{"x": x / 159, "y": y / 119} for x, y in [(15, 12), (144, 12), (144, 107), (15, 107)]]
    inner = [{"x": x / 159, "y": y / 119} for x, y in [(23, 20), (137, 20), (137, 100), (23, 100)]]
    body = {
        "bounding_box": {"x1": 0.0, "y1": 0.0, "x2": 1.0, "y2": 1.0},
        "segmentation_mask": {"polygon": inner, "visible_polygon": outer, "warp_quad": outer, "source": "manual_quad"},
    }
    response = await client.patch(f"{url}/bounding-box", json=body)
    assert response.status_code == 200, response.text
    unavailable = (await client.get(url)).json()
    assert unavailable["urls"]["full_print"]
    assert not unavailable["urls"]["image_only"]
    assert (await client.get(f"{url}/image?variant=image_only")).status_code == 404

    body["segmentation_mask"]["image_quad"] = inner
    response = await client.patch(f"{url}/bounding-box", json=body)
    assert response.status_code == 200, response.text
    payload = (await client.get(url)).json()
    for mode, size in (("full_print", (96, 130)), ("image_only", (81, 115))):
        artifact = await client.get(payload["urls"][mode])
        assert artifact.status_code == 200
        crop = cv2.imdecode(np.frombuffer(artifact.content, np.uint8), cv2.IMREAD_COLOR)
        assert crop.shape[:2] == size
        assert payload["enhancement_applied"]["correction"]["crop_modes"][mode]["status"] == "ready"
        if mode == "full_print":
            assert crop[2, 60].min() >= 225
        else:
            np.testing.assert_allclose(crop[4, 60], (50, 90, 150), atol=5)

    body["segmentation_mask"]["image_quad"] = [
        {"x": 0.0, "y": 0.0}, *inner[1:]
    ]
    failure = await client.patch(f"{url}/bounding-box", json=body)
    assert failure.status_code == 422, failure.text
    persisted = (await client.get(url)).json()
    assert persisted["segmentation_mask"]["image_quad"] == inner
    assert persisted["urls"]["full_print"] == payload["urls"]["full_print"]
    assert persisted["urls"]["image_only"] == payload["urls"]["image_only"]


async def test_measured_print_size_recovers_both_oblique_crops_and_survives_reprocessing(client, monkeypatch) -> None:
    from app.config import get_settings

    monkeypatch.setenv("ENABLE_PRECISE_PHOTO_CROPS", "true")
    monkeypatch.setenv("ENABLE_BORDER_TRIM", "false")
    monkeypatch.setenv("ENABLE_PHOTO_ORIENTATION_CORRECTION", "false")
    monkeypatch.setenv("ENABLE_PROTECTED_REGION_GUARD", "false")
    get_settings.cache_clear()
    width, height = 960, 780
    camera = np.float64([[900, 0, width / 2], [0, 900, height / 2], [0, 0, 1]])
    physical = np.float64([[-75, -50, 0], [75, -50, 0], [75, 50, 0], [-75, 50, 0]])
    projected, _ = cv2.projectPoints(physical, np.float64([0, 0.65, 0]), np.float64([0, 0, 350]), camera, None)
    print_corners = np.float32([[0, 0], [150, 0], [150, 100], [0, 100]])
    motif_corners = np.float32([[11, 17], [139, 17], [139, 89], [11, 89]])
    capture = cv2.getPerspectiveTransform(print_corners, projected.reshape(4, 2).astype(np.float32))
    original = np.full((101, 151, 3), 235, dtype=np.uint8)
    original[17:90, 11:140] = (35, 100, 166)
    image = cv2.warpPerspective(original, capture, (width, height), borderValue=(55, 55, 55))

    def vertices(points):
        captured = cv2.perspectiveTransform(points[None], capture)[0] / [width - 1, height - 1]
        return [{"x": float(x), "y": float(y)} for x, y in captured]

    outer, inner = vertices(print_corners), vertices(motif_corners)
    _, _, photo_id = await _seed_review_data(image)
    url = f"/api/v1/photos/{photo_id}"
    mask = {"polygon": outer, "warp_quad": outer, "image_quad": inner, "source": "manual_quad"}
    body = {
        "bounding_box": {"x1": 0.0, "y1": 0.0, "x2": 1.0, "y2": 1.0},
        "segmentation_mask": mask,
    }
    # Four correct corners still leave the metric shape of this single-axis view underconstrained.
    unmeasured = await client.patch(f"{url}/bounding-box", json=body)
    assert unmeasured.status_code == 200, unmeasured.text
    correction = unmeasured.json()["enhancement_applied"]["correction"]
    assert correction["crop_modes"]["full_print"]["reason_codes"] == ["PRINT_ASPECT_UNCERTAIN"]
    assert (await client.get(f"{url}/image?variant=full_print")).status_code == 404

    size = {"width": 150, "height": 100}
    mask["print_size_mm"] = size
    measured = await client.patch(f"{url}/bounding-box", json=body)
    assert measured.status_code == 200, measured.text
    assert measured.json()["aspect_ratio"] == pytest.approx(1.5)
    assert measured.json()["segmentation_mask"]["print_size_mm"] == size
    aspect = measured.json()["enhancement_applied"]["correction"]["aspect_ratio_estimation"]
    assert aspect == {"method": "known_print_aspect", "estimated_ratio": pytest.approx(1.5)}
    queue = await client.get("/api/v1/review/queue?type=geometry")
    assert queue.status_code == 200
    item = next(item for item in queue.json() if item["id"] == str(photo_id))
    assert item["payload"]["print_size_mm"] == size

    invalid = await client.patch(f"{url}/bounding-box", json={**body, "segmentation_mask": {
        **mask, "print_size_mm": {"width": 150, "height": 0},
    }})
    assert invalid.status_code == 422, invalid.text
    mask.pop("print_size_mm")
    mask["warp_quad"] = [outer[-1], *outer[:-1]]
    legacy_edit = await client.patch(f"{url}/bounding-box", json=body)
    assert legacy_edit.status_code == 200, legacy_edit.text
    assert legacy_edit.json()["segmentation_mask"]["print_size_mm"] == size
    assert legacy_edit.json()["segmentation_mask"]["warp_quad"] == outer
    reprocessed = await client.post(f"{url}/reprocess")
    assert reprocessed.status_code == 200, reprocessed.text
    payload = (await client.get(url)).json()
    assert payload["segmentation_mask"]["print_size_mm"] == size
    for mode, expected in (("full_print", 1.5), ("image_only", 128 / 72)):
        response = await client.get(payload["urls"][mode])
        assert response.status_code == 200
        crop = cv2.imdecode(np.frombuffer(response.content, np.uint8), cv2.IMREAD_COLOR)
        assert (crop.shape[1] - 1) / (crop.shape[0] - 1) == pytest.approx(expected, rel=0.007)
        expected_color = (235, 235, 235) if mode == "full_print" else (35, 100, 166)
        np.testing.assert_allclose(crop[5, crop.shape[1] // 2], expected_color, atol=6)

    # Numbering from another edge must rotate the result, with its measured axes intact.
    mask["warp_quad"] = [outer[-1], *outer[:-1]]
    mask["print_size_mm"] = {"width": 100, "height": 150}
    size_edit = {key: value for key, value in mask.items() if key != "image_quad"}
    renumbered = await client.patch(f"{url}/bounding-box", json={**body, "segmentation_mask": size_edit})
    assert renumbered.status_code == 200, renumbered.text
    assert renumbered.json()["segmentation_mask"]["image_quad"] == inner
    for mode, expected in (("full_print", 100 / 150), ("image_only", 72 / 128)):
        geometry = renumbered.json()["enhancement_applied"]["correction"]["crop_modes"][mode]
        assert geometry["status"] == "ready"
        assert (geometry["shape"]["width"] - 1) / (geometry["shape"]["height"] - 1) == pytest.approx(expected, rel=0.007)
    transformed = renumbered.json()["enhancement_applied"]["correction"]["warp_quad"]
    assert transformed[0] == pytest.approx(outer[-1], abs=1e-6)

    motif_edit = await client.patch(f"{url}/bounding-box", json={**body, "segmentation_mask": {
        **mask, "image_quad": vertices(np.float32([[11, 17], [139, 17], [116, 89], [38, 89]])),
    }})
    assert motif_edit.status_code == 200, motif_edit.text
    modes = motif_edit.json()["enhancement_applied"]["correction"]["crop_modes"]
    assert modes["full_print"]["status"] == "ready"
    assert modes["image_only"]["reason_codes"] == ["IMAGE_RECTANGLE_INCONSISTENT"]
    assert (await client.get(f"{url}/image?variant=image_only")).status_code == 404

    monkeypatch.setenv("ENABLE_PRECISE_PHOTO_CROPS", "false")
    get_settings.cache_clear()
    assert (await client.post(f"{url}/reprocess")).status_code == 200
    original = await client.get(f"{url}/image?variant=source_corrected_untrimmed")
    crop = cv2.imdecode(np.frombuffer(original.content, np.uint8), cv2.IMREAD_COLOR)
    assert original.status_code == 200
    assert (crop.shape[1] - 1) / (crop.shape[0] - 1) == pytest.approx(100 / 150, rel=0.007)

    monkeypatch.setenv("ENABLE_PRECISE_PHOTO_CROPS", "true")
    get_settings.cache_clear()
    mask["print_size_mm"] = None
    cleared = await client.patch(f"{url}/bounding-box", json=body)
    assert cleared.status_code == 200, cleared.text
    assert "print_size_mm" not in cleared.json()["segmentation_mask"]
    assert (await client.get(f"{url}/image?variant=full_print")).status_code == 404


async def test_unvalidated_automatic_crop_is_available_only_as_a_review_preview(client, monkeypatch) -> None:
    from app.config import get_settings

    monkeypatch.setenv("ENABLE_BORDER_TRIM", "false")
    monkeypatch.setenv("ENABLE_PHOTO_ORIENTATION_CORRECTION", "false")
    monkeypatch.setenv("REQUIRE_CONFIDENCE_CALIBRATION", "true")
    get_settings.cache_clear()
    _, _, photo_id = await _seed_review_data()
    url = f"/api/v1/photos/{photo_id}"
    result = await client.post(f"{url}/reprocess")
    assert result.status_code == 200, result.text
    payload = (await client.get(url)).json()
    full_mode = payload["enhancement_applied"]["correction"]["crop_modes"]["full_print"]
    assert full_mode["geometry_status"] == "ready"
    assert full_mode["status"] == "review_needed"
    assert payload["urls"]["full_print"] is None
    assert (await client.get(payload["urls"]["source_corrected"])).status_code == 200

    confirmed = await client.patch(
        f"{url}/bounding-box",
        json={
            "bounding_box": payload["bounding_box"],
            "segmentation_mask": {
                "polygon": full_mode["warp_quad"],
                "warp_quad": full_mode["warp_quad"],
                "source": "manual_quad",
            },
        },
    )
    assert confirmed.status_code == 200, confirmed.text
    available = (await client.get(url)).json()
    assert (await client.get(available["urls"]["full_print"])).status_code == 200


async def test_review_can_remove_a_false_nested_candidate_from_occlusion_evidence(client, monkeypatch) -> None:
    from app.config import get_settings
    from app.database import get_session_factory
    from app.models import ExtractedPhoto

    monkeypatch.setenv("ENABLE_PRECISE_PHOTO_CROPS", "true")
    monkeypatch.setenv("ENABLE_BORDER_TRIM", "false")
    monkeypatch.setenv("ENABLE_PROTECTED_REGION_GUARD", "false")
    monkeypatch.setenv("ENABLE_PHOTO_ORIENTATION_CORRECTION", "false")
    get_settings.cache_clear()
    image = np.full((200, 290, 3), 60, dtype=np.uint8)
    image[18:182, 21:272] = (99, 140, 190)
    image[51:148, 98:195] = (179, 50, 75)
    album_id, page_id, whole_id = await _seed_review_data(image)

    def edit(x1, y1, x2, y2):
        quad = [{"x": x / 289, "y": y / 199} for x, y in [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]]
        return {
            "bounding_box": {"x1": x1 / 289, "y1": y1 / 199, "x2": x2 / 289, "y2": y2 / 199},
            "segmentation_mask": {"source": "manual_quad", "polygon": quad, "warp_quad": quad},
        }

    nested_id = uuid4()
    nested_edit = edit(98, 51, 194, 147)
    async with get_session_factory()() as session:
        session.add(ExtractedPhoto(id=nested_id, page_id=page_id, album_id=album_id, storage_path="", **nested_edit))
        await session.commit()

    whole = f"/api/v1/photos/{whole_id}"
    nested = f"/api/v1/photos/{nested_id}"
    reviewed = await client.patch(f"{whole}/bounding-box", json=edit(21, 18, 271, 181))
    assert reviewed.status_code == 200, reviewed.text
    full_mode = reviewed.json()["enhancement_applied"]["correction"]["crop_modes"]["full_print"]
    assert full_mode["reason_codes"] == ["PRINT_POSSIBLY_OCCLUDED"]
    assert (await client.patch(f"{nested}/bounding-box", json=nested_edit)).status_code == 200
    assert (await client.get(f"{nested}/image?variant=full_print")).status_code == 200

    rejection = await client.post("/api/v1/review/events", json={
        "task_type": "geometry", "task_id": str(nested_id), "answer": "wrong_object", "idempotency_key": str(uuid4()),
    })
    assert rejection.status_code == 201, rejection.text
    assert (await client.get(f"{nested}/image?variant=full_print")).status_code == 404
    assert (await client.get(f"{nested}/image?variant=source_corrected")).status_code == 200
    result = await client.post(f"{whole}/reprocess")
    assert result.status_code == 200, result.text
    assert (await client.get(f"{whole}/image?variant=full_print")).status_code == 200

    # A new explicit edit supersedes the rejection, without ignoring the new candidate.
    assert (await client.patch(f"{nested}/bounding-box", json=nested_edit)).status_code == 200
    assert (await client.post(f"{whole}/reprocess")).status_code == 200
    assert (await client.get(f"{whole}/image?variant=full_print")).status_code == 404


@pytest.mark.parametrize("corner_mount", [False, True], ids=["full-print-visible", "motif-only-visible"])
@pytest.mark.parametrize("with_refiner", [False, True], ids=["primary-detector", "visible-mask-refiner"])
async def test_automatic_crop_download_requires_current_album_validation(
    client, monkeypatch, tmp_path, corner_mount, with_refiner,
) -> None:
    from app.config import get_settings
    from app.database import get_session_factory
    from app.models import ExtractedPhoto, Page
    from app.pipeline.confidence import calibrate_confidence
    from app.pipeline.crop_validation import CROP_VALIDATION_SCHEMA, crop_validation_profile
    from app.services.pipeline import perspective_correction_task_logic

    model_path = tmp_path / "registered.onnx"
    model_path.write_bytes(b"registered-checkpoint-for-api-test")
    digest = hashlib.sha256(model_path.read_bytes()).hexdigest()
    registry_path = tmp_path / "manifest.json"
    registry = {"models": {model_path.name: {"sha256": digest, "license": "test", "input": {"width": 640}}}}
    refiner_metadata = {}
    refiner = tmp_path / "visible-mask.onnx"
    refiner_bytes = b"registered-visible-mask-for-api-test"
    if with_refiner:
        refiner.write_bytes(refiner_bytes)
        refiner_digest = hashlib.sha256(refiner_bytes).hexdigest()
        registry["models"][refiner.name] = {"sha256": refiner_digest, "license": "test"}
        monkeypatch.setenv("VISIBLE_MASK_MODEL_PATH", str(refiner))
        refiner_metadata = {"visible_mask_refinement": {"model_sha256": refiner_digest, "refined_detections": 1}}
    registry_path.write_text(json.dumps(registry), encoding="utf-8")
    calibration_path = tmp_path / "calibration.json"
    calibration_path.write_text(json.dumps({
        "schema_version": 1,
        "independent_holdout_id": "fixture-confidence-holdout",
        "strata": {"default": {"raw": [0, 1], "calibrated": [0, 1]}},
    }), encoding="utf-8")
    validation_path = tmp_path / "crop-validation.json"
    monkeypatch.setenv("ENABLE_PRECISE_PHOTO_CROPS", "true")
    monkeypatch.setenv("ENABLE_BORDER_TRIM", "false")
    monkeypatch.setenv("ENABLE_PROTECTED_REGION_GUARD", "true")
    monkeypatch.setenv("ENABLE_PHOTO_ORIENTATION_CORRECTION", "false")
    monkeypatch.setenv("REQUIRE_CONFIDENCE_CALIBRATION", "true")
    monkeypatch.setenv("YOLO_MODEL_PATH", str(model_path))
    monkeypatch.setenv("MODEL_MANIFEST_PATH", str(registry_path))
    monkeypatch.setenv("REQUIRE_REGISTERED_MODELS", "true")
    monkeypatch.setenv("SEGMENTATION_CALIBRATION_PATH", str(calibration_path))
    monkeypatch.setenv("CROP_VALIDATION_REPORT_PATH", str(validation_path))
    get_settings.cache_clear()
    profile = crop_validation_profile(get_settings())
    image = np.full((120, 160, 3), 105, dtype=np.uint8)
    image[24:108, 16:128] = 240
    margin = 13 if corner_mount else 6
    left, right, top, bottom = 16 + margin, 127 - margin, 24 + margin, 107 - margin
    image[top:bottom + 1, left:right + 1] = (50, 90, 150)
    if corner_mount:
        cv2.fillConvexPoly(image, np.int32([[13, 21], [36, 21], [13, 44]]), (48, 55, 61))
    album_id, page_id, photo_id = await _seed_review_data(image)
    outer = [{"x": x / 159, "y": y / 119} for x, y in [(16, 24), (127, 24), (127, 107), (16, 107)]]
    inner = [{"x": x / 159, "y": y / 119} for x, y in [(left, top), (right, top), (right, bottom), (left, bottom)]]
    async with get_session_factory()() as session:
        page = await session.get(Page, page_id)
        page.processing_metadata = {"steps": {"segmentation": {
            "model_artifact": {"sha256": digest}, "validation_profile": profile,
            **refiner_metadata,
        }}}
        photo = await session.get(ExtractedPhoto, photo_id)
        _, calibration = calibrate_confidence(photo.segmentation_confidence, "default", calibration_path)
        assert calibration["calibrated"] is True
        photo.enhancement_applied = {**photo.enhancement_applied, "segmentation_confidence": calibration}
        photo.segmentation_mask = {"source": "classical_boundary", "polygon": outer, "warp_quad": outer, "image_quad": inner}
        await session.commit()

    url = f"/api/v1/photos/{photo_id}"
    crop_variants = ("image_only",) if corner_mount else ("full_print", "image_only")
    variants = (*crop_variants, "source_corrected_trimmed")
    result = await client.post(f"{url}/reprocess")
    assert result.status_code == 200, result.text
    payload = (await client.get(url)).json()
    crop_modes = payload["enhancement_applied"]["correction"]["crop_modes"]
    for mode in crop_variants:
        assert crop_modes[mode]["geometry_status"] == "ready"
        assert crop_modes[mode]["reason_codes"] == ["CROP_VALIDATION_UNAVAILABLE"]
    assert all(payload["urls"][name] is None for name in ("full_print", "image_only", "source_corrected_trimmed"))
    assert (await client.get(payload["urls"]["source_corrected"])).status_code == 200

    report = {
        "schema_version": CROP_VALIDATION_SCHEMA,
        "model_sha256": digest,
        "validation_profile": profile,
        "legacy_crop": False,
        "geometry_only": False,
        "split": {"album_independent": True},
        "dataset_manifest_sha256": "d" * 64,
        "validation_gates": {
            "locked_real_album_manifest": True, "locked_reference_artifacts": True, "calibration_holdout_disjoint": True,
            "standard_detection_thresholds": True,
            "detection": True, "final_instance_segmentation": True,
            "rendered_crop_modes": {"full_print": True, "image_only": True},
        },
    }
    validation_path.write_text(json.dumps(report | {"schema_version": CROP_VALIDATION_SCHEMA - 1}), encoding="utf-8")
    result = await client.post(f"{url}/reprocess")
    assert result.status_code == 200, result.text
    stale = (await client.get(url)).json()["urls"]
    assert all(stale[mode] is None for mode in crop_variants)

    validation_path.write_text(json.dumps(report), encoding="utf-8")
    if with_refiner:
        async with get_session_factory()() as session:
            page = await session.get(Page, page_id)
            steps = page.processing_metadata["steps"]
            page.processing_metadata = {"steps": {**steps, "segmentation": {
                **steps["segmentation"], "visible_mask_refinement": {"error": "failed to load"},
            }}}
            await session.commit()
        result = await client.post(f"{url}/reprocess")
        assert result.status_code == 200, result.text
        unavailable = (await client.get(url)).json()["urls"]
        assert all(unavailable[mode] is None for mode in crop_variants)
        async with get_session_factory()() as session:
            page = await session.get(Page, page_id)
            steps = page.processing_metadata["steps"]
            page.processing_metadata = {"steps": {
                **steps, "segmentation": {**steps["segmentation"], **refiner_metadata},
            }}
            await session.commit()
    result = await client.post(f"{url}/reprocess")
    assert result.status_code == 200, result.text
    payload = (await client.get(url)).json()
    released = payload["urls"]
    for name in variants:
        assert (await client.get(released[name])).status_code == 200
    album_photos = (await client.get(f"/api/v1/albums/{album_id}/photos")).json()
    assert album_photos[0]["crop_download_urls"] == {mode: released[mode] for mode in ("full_print", "image_only")}

    if corner_mount:
        correction = payload["enhancement_applied"]["correction"]
        assert correction["crop_modes"]["image_only"]["independent_of_full_print"] is True
        assert payload["status"] == "review_needed"
        assert released["full_print"] is None
        assert (await client.get(f"{url}/image?variant=full_print")).status_code == 404
        assert (await client.get(released["source_corrected_untrimmed"])).status_code == 200
        approval = await client.post("/api/v1/review/events", json={
            "task_type": "geometry", "task_id": str(photo_id), "answer": "correct", "idempotency_key": uuid4().hex,
        })
        assert approval.status_code == 409
        crop = cv2.imdecode(np.frombuffer((await client.get(released["image_only"])).content, np.uint8), cv2.IMREAD_COLOR)
        assert crop.shape[:2] == (bottom - top + 1, right - left + 1)
        assert np.max(np.abs(crop[8:-8, 8:-8].astype(np.int16) - [50, 90, 150])) <= 5
        async with get_session_factory()() as session:
            assert (await perspective_correction_task_logic(session, photo_id, get_settings(), skip_if_exists=True))["idempotent_replay"]
            Path(correction["trimmed_path"]).unlink()
            repaired = await perspective_correction_task_logic(session, photo_id, get_settings(), skip_if_exists=True)
            assert repaired["review_required"] and Path(repaired["trimmed_path"]).is_file()
        released = (await client.get(url)).json()["urls"]

    monkeypatch.setenv("ENABLE_BORDER_TRIM", "true")
    get_settings.cache_clear()
    assert (await client.get(released["image_only"])).status_code == 404
    monkeypatch.setenv("ENABLE_BORDER_TRIM", "false")
    get_settings.cache_clear()
    model_path.write_bytes(b"another-checkpoint")
    checkpoint_variant = "image_only" if corner_mount else "full_print"
    assert (await client.get(released[checkpoint_variant])).status_code == 404
    model_path.write_bytes(b"registered-checkpoint-for-api-test")
    assert (await client.get(released[checkpoint_variant])).status_code == 200
    registry_path.write_text("{}", encoding="utf-8")
    assert (await client.get(released["image_only"])).status_code == 404
    registry_path.write_text(json.dumps(registry), encoding="utf-8")
    assert (await client.get(released["image_only"])).status_code == 200
    if with_refiner:
        refiner.write_bytes(b"replacement-mask-model")
        assert (await client.get(released["image_only"])).status_code == 404
        refiner.write_bytes(refiner_bytes)
        assert (await client.get(released["image_only"])).status_code == 200
    validation_path.unlink()
    revoked = (await client.get(url)).json()["urls"]
    for name in variants:
        assert revoked[name] is None
        assert (await client.get(released[name])).status_code == 404
    assert (await client.get(revoked["source_corrected"])).status_code == 200
    approval = await client.post("/api/v1/review/events", json={
        "task_type": "geometry", "task_id": str(photo_id), "answer": "correct", "idempotency_key": uuid4().hex,
    })
    assert approval.status_code == 409

    async with get_session_factory()() as session:
        page = await session.get(Page, page_id)
        steps = page.processing_metadata["steps"] | {"protected_regions": {"regions": [{
            "type": "face", "source": "manual", "box": {"x1": 0.05, "y1": 0.39, "x2": 0.15, "y2": 0.60},
        }]}}
        page.processing_metadata = {"steps": steps}
        await session.commit()
    result = await client.post(f"{url}/reprocess")
    assert result.status_code == 200, result.text
    payload = (await client.get(url)).json()
    assert payload["enhancement_applied"]["correction"]["status"] == "rejected"
    assert payload["enhancement_applied"]["correction"]["crop_modes"]["image_only"]["status"] == "review_needed"
    assert (await client.get(released["source_corrected_trimmed"])).status_code == 404


async def test_reprocessing_keeps_the_review_preview_for_an_automatic_face_hint(client, monkeypatch) -> None:
    from app.config import get_settings
    from app.database import get_session_factory
    from app.models import Page

    monkeypatch.setenv("ENABLE_PRECISE_PHOTO_CROPS", "true")
    monkeypatch.setenv("ENABLE_PROTECTED_REGION_GUARD", "true")
    monkeypatch.setenv("ENABLE_PHOTO_ORIENTATION_CORRECTION", "false")
    get_settings.cache_clear()
    _, page_id, photo_id = await _seed_review_data()
    async with get_session_factory()() as session:
        page = await session.get(Page, page_id)
        page.processing_metadata = {"steps": {"protected_regions": {"regions": [{
            "type": "face", "source": "opencv_haarcascade_frontalface_default",
            "box": {"x1": 0.05, "y1": 0.39, "x2": 0.15, "y2": 0.60},
        }]}}}
        await session.commit()
    url = f"/api/v1/photos/{photo_id}"
    result = await client.post(f"{url}/reprocess")
    assert result.status_code == 200, result.text
    payload = (await client.get(url)).json()
    full_mode = payload["enhancement_applied"]["correction"]["crop_modes"]["full_print"]
    assert "POSSIBLE_CONTENT_CUTOFF" in full_mode["reason_codes"]
    assert payload["urls"]["full_print"] is None
    assert (await client.get(payload["urls"]["source_corrected_untrimmed"])).status_code == 200


async def test_invalid_neighbor_mask_can_still_show_an_occlusion(client, monkeypatch) -> None:
    from app.config import get_settings
    from app.database import get_session_factory
    from app.models import ExtractedPhoto

    monkeypatch.setenv("ENABLE_PRECISE_PHOTO_CROPS", "true")
    monkeypatch.setenv("ENABLE_PHOTO_ORIENTATION_CORRECTION", "false")
    get_settings.cache_clear()
    album_id, page_id, photo_id = await _seed_review_data()

    def vertices(points):
        return [{"x": x / 159, "y": y / 119} for x, y in points]

    outer = vertices([(16, 24), (127, 24), (127, 107), (16, 107)])
    visible = vertices([(16, 24), (63, 24), (63, 43), (81, 43), (81, 24), (127, 24), (127, 107), (16, 107)])
    neighbor = vertices([(63, 2), (81, 2), (81, 43), (63, 43)])
    async with get_session_factory()() as session:
        photo = await session.get(ExtractedPhoto, photo_id)
        photo.segmentation_mask = {"source": "yolo_seg_mask", "warp_quad": outer, "polygon": visible}
        session.add(ExtractedPhoto(
            page_id=page_id, album_id=album_id, storage_path="", geometry_valid=False,
            bounding_box={"x1": 63 / 159, "y1": 2 / 119, "x2": 81 / 159, "y2": 43 / 119},
            segmentation_mask={"source": "yolo_seg_mask", "polygon": neighbor},
        ))
        await session.commit()
    url = f"/api/v1/photos/{photo_id}"
    result = await client.post(f"{url}/reprocess")
    assert result.status_code == 200, result.text
    modes = (await client.get(url)).json()["enhancement_applied"]["correction"]["crop_modes"]
    assert "PRINT_POSSIBLY_OCCLUDED" in modes["full_print"]["reason_codes"]


async def test_visible_rim_is_persisted_without_releasing_a_curved_print(client, monkeypatch) -> None:
    from app.config import get_settings
    from app.database import get_session_factory
    from app.evaluation.crop_fixtures import _page
    from app.evaluation.cropping import evaluate_page_crops
    from app.models import ExtractedPhoto

    monkeypatch.setenv("ENABLE_PHOTO_ORIENTATION_CORRECTION", "false")
    get_settings.cache_clear()
    fixture = _page("curl", np.random.default_rng(31523))
    height, width = fixture.image.shape[:2]
    instance = fixture.prints[0]
    contours, _ = cv2.findContours(instance.reference_modes["image_only"].mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    points = cv2.approxPolyDP(max(contours, key=cv2.contourArea), 0.45, True).reshape(-1, 2) / [width - 1, height - 1]
    raw = [{"x": float(x), "y": float(y)} for x, y in points]
    _, _, photo_id = await _seed_review_data(fixture.image)
    async with get_session_factory()() as session:
        photo = await session.get(ExtractedPhoto, photo_id)
        photo.segmentation_mask = {"polygon": raw, "visible_polygon": raw, "source": "yolo_seg_mask"}
        await session.commit()

    url = f"/api/v1/photos/{photo_id}"
    repro = await client.post(f"{url}/reprocess")
    assert repro.status_code == 200, repro.text
    first = (await client.get(url)).json()
    assert first["urls"]["full_print"] is None and first["urls"]["image_only"] is None
    assert first["urls"]["source_corrected"]
    visible = first["segmentation_mask"]
    assert visible["visible_polygon_source"] == "visible_paper_rim"
    assert visible["polygon"] == visible["proposal_visible_polygon"] == raw
    reference = [instance.visible_polygon]
    refined = [[(point["x"], point["y"]) for point in visible["visible_polygon"]]]
    before = evaluate_page_crops(reference, [points.tolist()], width=width, height=height)
    after = evaluate_page_crops(reference, refined, width=width, height=height)
    assert after.mask_iou >= 0.95 and after.mask_iou > before.mask_iou + 0.10

    assert (await client.post(f"{url}/reprocess")).status_code == 200
    assert (await client.get(url)).json()["segmentation_mask"] == visible
