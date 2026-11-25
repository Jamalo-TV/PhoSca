from uuid import UUID, uuid4

import cv2
import numpy as np
import pytest


@pytest.mark.parametrize("add_second", [False, True])
async def test_fusion_proposals_stay_uncounted_until_manually_added_or_dismissed(client, monkeypatch, add_second):
    from app.config import get_settings
    from app.database import get_session_factory
    from app.models import ReviewEvent
    from app.pipeline.segmentation import SegmentationDetection, SegmentationResult
    from app.routers import photos
    from app.services import pipeline

    monkeypatch.setenv("AUTO_ANALYZE_UPLOADS", "true")
    monkeypatch.setenv("ENABLE_PHOTO_ORIENTATION_CORRECTION", "false")
    monkeypatch.setenv("ENABLE_PROTECTED_REGION_GUARD", "false")
    monkeypatch.setattr(pipeline, "run_paddle_ocr", lambda *_: [])
    get_settings.cache_clear()

    image = np.full((180, 300, 3), 210, np.uint8)
    image[31:129, 10:99] = (72, 92, 178)
    image[30:151, 113:213] = (238, 239, 240)
    image[36:145, 119:207] = (113, 61, 91)
    if add_second:
        image[32:137, 230:287] = (135, 98, 48)

    def vertices(left, top, right, bottom):
        return [{"x": x / 299, "y": y / 179} for x, y in [(left, top), (right, top), (right, bottom), (left, bottom)]]

    first = vertices(10, 31, 98, 128)
    detected = SegmentationDetection(
        bounding_box={"x1": first[0]["x"], "y1": first[0]["y"], "x2": first[2]["x"], "y2": first[2]["y"]},
        mask={"polygon": first, "source": "yolo_seg_mask", "detector_source": "yolo_onnx"},
        confidence=.95,
        aspect_ratio=.91,
        geometry_valid=True,
        review_reasons=[],
    )
    pending = [
        {"id": 0, "polygon": vertices(114, 30, 212, 150), "reason": "unclear_overlap"},
        {"id": 4, "polygon": vertices(230, 32, 286, 136), "reason": "unclear_overlap"},
    ]
    monkeypatch.setattr(pipeline, "segment_album_page", lambda *_, **__: SegmentationResult(
        [detected], {"fusion": {"deferred_print_proposals": pending, "selected_instances": 1}},
    ))
    encoded, png = cv2.imencode(".png", image)
    assert encoded
    created_album = await client.post("/api/v1/albums", json={"name": "Count review"})
    assert created_album.status_code == 201, created_album.text
    album_id = created_album.json()["id"]
    album_url = f"/api/v1/albums/{album_id}"
    uploaded = await client.post(f"{album_url}/pages/upload", files={"files": ("page.png", png.tobytes(), "image/png")})
    assert uploaded.status_code == 201, uploaded.text
    page, = (await client.get(f"{album_url}/pages")).json()
    assert page["status"] == "review_needed"
    assert page["processing_metadata"]["steps"]["segmentation"]["fusion"]["deferred_print_proposals"] == pending
    page_id = page["id"]
    manual_url = f"/api/v1/pages/{page_id}/manual-photos"
    photo_list = f"{album_url}/photos"
    review_url = f"/api/v1/review/queue?album_id={album_id}&type=segmentation"
    review, = (await client.get(review_url)).json()
    before = review["payload"]
    assert before["detection_count"] == len((await client.get(photo_list)).json()) == 1
    assert before["deferred_print_proposals"] == pending

    confirmation = {
        "task_type": "segmentation", "task_id": page_id, "answer": "correct", "idempotency_key": "page-confirm-1",
        "metadata": {"photo_count_revision": before["photo_count_revision"]},
    }
    unacknowledged = await client.post("/api/v1/review/events", json=confirmation)
    assert unacknowledged.status_code == 409, unacknowledged.text

    # Preserve both hints until a correction has rendered and the transaction commits.
    addition = {
        "request_id": str(uuid4()), "warp_quad": vertices(113, 30, 212, 150), "deferred_proposal_id": 0,
        "photo_count_revision": before["photo_count_revision"],
    }
    reused_cluster = await client.post(manual_url, json={**addition, "photo_count_revision": str(uuid4())})
    assert reused_cluster.status_code == 409 and reused_cluster.json()["detail"] == "deferred_print_proposal_stale"

    async def reject_render(*_, **__):
        raise ValueError("simulated invalid crop")

    with monkeypatch.context() as faults:
        faults.setattr(photos, "perspective_correction_task_logic", reject_render)
        failed = await client.post(manual_url, json=addition)
    assert failed.status_code == 422, failed.text
    assert len((await client.get(photo_list)).json()) == 1
    assert (await client.get(review_url)).json()[0]["payload"]["deferred_print_proposals"] == pending

    added = await client.post(manual_url, json=addition)
    assert added.status_code == 201, added.text
    added_photo = added.json()
    provenance = added_photo["enhancement_applied"]["review_provenance"]
    assert provenance["deferred_print_proposal"] == pending[0]
    replay = await client.post(manual_url, json=addition)
    assert replay.status_code == 200 and replay.json()["id"] == added_photo["id"]
    duplicate = await client.post(manual_url, json={**addition, "request_id": str(uuid4())})
    assert duplicate.status_code == 409 and duplicate.json()["detail"] == "deferred_print_proposal_saved"
    review, = (await client.get(review_url)).json()
    after = review["payload"]
    assert after["detection_count"] == len((await client.get(photo_list)).json()) == 2
    assert after["deferred_print_proposals"] == [pending[1]]
    assert after["photo_count_revision"] != before["photo_count_revision"]
    second = {**addition, "request_id": str(uuid4()), "deferred_proposal_id": 4, "warp_quad": pending[1]["polygon"]}
    stale_addition = await client.post(manual_url, json=second)
    assert stale_addition.status_code == 409 and stale_addition.json()["detail"] == "deferred_print_proposal_stale"

    remaining = [pending[1]]
    if add_second:
        next_added = await client.post(manual_url, json={**second, "photo_count_revision": after["photo_count_revision"]})
        assert next_added.status_code == 201, next_added.text
        assert next_added.json()["enhancement_applied"]["review_provenance"]["deferred_print_proposal"] == pending[1]
        review, = (await client.get(review_url)).json()
        after = review["payload"]
        assert after["detection_count"] == len((await client.get(photo_list)).json()) == 3
        remaining = []
        assert after["deferred_print_proposals"] == remaining

    confirmation["metadata"].update(deferred_proposal_ids=[candidate["id"] for candidate in remaining])
    stale = await client.post("/api/v1/review/events", json=confirmation)
    assert stale.status_code == 409, stale.text
    confirmation["metadata"].update(photo_count_revision=after["photo_count_revision"])
    accepted = await client.post("/api/v1/review/events", json=confirmation)
    assert accepted.status_code == 201, accepted.text
    assert (await client.get(review_url)).json() == []
    page, = (await client.get(f"{album_url}/pages")).json()
    assert "deferred_print_proposals" not in page["processing_metadata"]["steps"]["segmentation"]["fusion"]
    dismissed = await client.post(manual_url, json={**second, "request_id": str(uuid4()),
                                                   "photo_count_revision": after["photo_count_revision"]})
    assert dismissed.status_code == 409 and dismissed.json()["detail"] == "deferred_print_proposal_stale"
    async with get_session_factory()() as session:
        event = await session.get(ReviewEvent, UUID(accepted.json()["id"]))
        assert event.event_metadata["snapshot"]["provenance"]["segmentation"]["fusion"]["deferred_print_proposals"] == remaining
