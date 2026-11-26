from __future__ import annotations

from contextlib import asynccontextmanager
from copy import deepcopy
from uuid import UUID, uuid4

import cv2
import numpy as np


WIDTH, HEIGHT = 440, 280
DOWNLOADS = ("full_print", "image_only", "source_corrected_trimmed")
EVENT_URL = "/api/v1/review/events"


def _point(x, y):
    return {"x": x / (WIDTH - 1), "y": y / (HEIGHT - 1)}


def _quad(left, top, right, bottom):
    return [_point(x, y) for x, y in ((left, top), (right, top), (right, bottom), (left, bottom))]


@asynccontextmanager
async def _temporary_photo_fields(photo_id, **changes):
    from app.database import get_session_factory
    from app.models import ExtractedPhoto

    async with get_session_factory()() as session:
        photo = await session.get(ExtractedPhoto, UUID(photo_id))
        original = {name: deepcopy(getattr(photo, name)) for name in changes}
        for name, value in changes.items():
            setattr(photo, name, value)
        await session.commit()
    try:
        yield
    finally:
        async with get_session_factory()() as session:
            photo = await session.get(ExtractedPhoto, UUID(photo_id))
            for name, value in original.items():
                setattr(photo, name, value)
            await session.commit()


async def test_album_leaf_exclusion_requires_the_current_held_leaf_and_its_unique_automatic_child(client, monkeypatch):
    from app.config import get_settings
    from app.database import get_session_factory
    from app.models import ReviewEvent
    from app.pipeline.segmentation import SegmentationDetection, SegmentationResult
    from app.services import pipeline

    monkeypatch.setenv("AUTO_ANALYZE_UPLOADS", "true")
    monkeypatch.setenv("MAX_UPLOAD_SIZE", "1048576")
    monkeypatch.setenv("MAX_REQUEST_SIZE", "2097152")
    monkeypatch.setenv("ENABLE_PRECISE_PHOTO_CROPS", "true")
    monkeypatch.setenv("ENABLE_PHOTO_ORIENTATION_CORRECTION", "false")
    monkeypatch.setenv("ENABLE_PROTECTED_REGION_GUARD", "false")
    monkeypatch.setattr(pipeline, "run_paddle_ocr", lambda *_: [])
    get_settings.cache_clear()

    leaf = _quad(20, 23, 220, 253)
    child = _quad(44, 40, 200, 236)
    separate = _quad(250, 38, 408, 225)
    traces = [[_point(20, 30), _point(20, 235)], [_point(32, 23), _point(210, 23)]]
    detections = []
    for polygon in (leaf, child, separate):
        bounds = {"x1": polygon[0]["x"], "y1": polygon[0]["y"], "x2": polygon[2]["x"], "y2": polygon[2]["y"]}
        if polygon == leaf:
            mask = {"source": "classical_border_quad", "detector_source": "contrast_region", "polygon": polygon,
                    "album_leaf_review_traces": traces, "album_leaf_review_child": child}
            detections.append(SegmentationDetection(bounds, mask, 0.61, 0.87, True, ["ALBUM_LEAF_COUNT_REVIEW"]))
        else:
            mask = {"source": "yolo_seg_mask", "detector_source": "yolo_onnx", "polygon": polygon}
            detections.append(SegmentationDetection(bounds, mask, 0.91, 0.81, True, []))
    monkeypatch.setattr(pipeline, "segment_album_page", lambda *_, **__: SegmentationResult(detections, {}))

    image = np.full((HEIGHT, WIDTH, 3), (89, 103, 123), np.uint8)
    image[23:254, 20:221] = (220, 230, 235)
    image[40:237, 44:201] = (53, 82, 110)
    image[38:226, 250:409] = (89, 137, 186)
    ok, png = cv2.imencode(".png", image)
    assert ok
    created = await client.post("/api/v1/albums", json={"name": "One leaf and two prints"})
    assert created.status_code == 201, created.text
    album_id = created.json()["id"]
    album_url = f"/api/v1/albums/{album_id}"
    uploaded = await client.post(f"{album_url}/pages/upload", files={"files": ("leaf.png", png.tobytes(), "image/png")})
    assert uploaded.status_code == 201, uploaded.text

    queue_url = f"/api/v1/review/queue?album_id={album_id}&type=segmentation"
    task, = (await client.get(queue_url)).json()
    before = task["payload"]
    assert before["detection_count"] == 3
    leaf_row, = [row for row in before["detections"] if row["segmentation_mask"].get("album_leaf_review_traces")]
    child_row, = [row for row in before["detections"] if row["segmentation_mask"]["polygon"] == child]
    other_row, = [row for row in before["detections"] if row["segmentation_mask"]["polygon"] == separate]
    assert leaf_row["segmentation_mask"]["album_leaf_review_traces"] == traces
    assert leaf_row["segmentation_mask"]["album_leaf_review_child"] == child_row["segmentation_mask"]["polygon"]
    leaf_id, child_id, other_id = (row["photo_id"] for row in (leaf_row, child_row, other_row))
    leaf_url, child_url, other_url = [f"/api/v1/photos/{photo_id}" for photo_id in (leaf_id, child_id, other_id)]
    leaf_before, child_before, other_before = [(await client.get(url)).json() for url in (leaf_url, child_url, other_url)]
    enhancement = leaf_before["enhancement_applied"]
    assert "ALBUM_LEAF_COUNT_REVIEW" in enhancement["segmentation_review_reasons"]
    assert all(enhancement["correction"]["crop_modes"][mode]["status"] == "review_needed" for mode in DOWNLOADS[:2])

    async def assert_held():
        for variant in DOWNLOADS:
            assert (await client.get(f"{leaf_url}/image", params={"variant": variant})).status_code == 404
        assert (await client.get(f"{leaf_url}/image?variant=source_corrected_untrimmed")).status_code == 200

    await assert_held()
    untouched = [await client.get(f"{url}/image?variant=source_corrected_untrimmed") for url in (child_url, other_url)]
    assert all(response.status_code == 200 for response in untouched)

    metadata = {"photo_count_revision": before["photo_count_revision"], "album_leaf_group_photo_ids": [leaf_id]}
    body = {
        "task_type": "segmentation", "task_id": task["id"], "answer": "exclude_album_leaf_group",
        "reason_code": "non_photo_object", "idempotency_key": uuid4().hex, "metadata": metadata,
    }
    for reason_code in (None, "other"):
        no_confirmation = await client.post(EVENT_URL, json={**body, "reason_code": reason_code})
        assert no_confirmation.status_code == 422, no_confirmation.text
        assert no_confirmation.json()["detail"] == "album_leaf_confirmation_required"
    for proposed_ids in ([], [leaf_id, leaf_id]):
        bad = await client.post(EVENT_URL, json={**body, "metadata": {**metadata, "album_leaf_group_photo_ids": proposed_ids}})
        assert bad.status_code == 422, bad.text
        assert bad.json()["detail"] == "album_leaf_group_selection_invalid"
    wrong_row = await client.post(
        EVENT_URL, json={**body, "metadata": {**metadata, "album_leaf_group_photo_ids": [child_id]}},
    )
    assert wrong_row.status_code == 409, wrong_row.text
    assert wrong_row.json()["detail"] == "album_leaf_review_group_stale"
    stale = await client.post(EVENT_URL, json={**body, "metadata": {**metadata, "photo_count_revision": uuid4().hex}})
    assert stale.status_code == 409, stale.text

    moved_child = {**child_before["segmentation_mask"], "polygon": _quad(46, 40, 200, 236)}
    rejected_child = {**child_before["enhancement_applied"], "review_rejected_detection": {
        "bounding_box": child_before["bounding_box"], "segmentation_mask": child_before["segmentation_mask"],
    }}
    ready_leaf = deepcopy(enhancement)
    ready_leaf["correction"]["crop_modes"]["image_only"]["status"] = "ready"
    changed_children = [
        (child_id, {"segmentation_mask": moved_child}),
        (child_id, {"segmentation_confidence": None}),
        (child_id, {"enhancement_applied": rejected_child}),
        (other_id, {"segmentation_mask": child_before["segmentation_mask"], "bounding_box": child_before["bounding_box"]}),
        (leaf_id, {"segmentation_mask": {**leaf_before["segmentation_mask"], "album_leaf_review_traces": traces[:1]}}),
        (leaf_id, {"enhancement_applied": ready_leaf}),
    ]
    for photo_id, changes in changed_children:
        async with _temporary_photo_fields(photo_id, **changes):
            veto = await client.post(EVENT_URL, json=body)
            assert veto.status_code == 409, veto.text
            assert veto.json()["detail"] == "album_leaf_review_group_stale"
    assert (await client.get(queue_url)).json() == [task]
    assert (await client.get(leaf_url)).json() == leaf_before

    count_answer = {
        "task_type": "segmentation", "task_id": task["id"], "answer": "correct",
        "idempotency_key": uuid4().hex, "metadata": {"photo_count_revision": before["photo_count_revision"]},
    }
    counted = await client.post(EVENT_URL, json=count_answer)
    assert counted.status_code == 201, counted.text
    assert (await client.get(queue_url)).json() == []
    closed_dialog = await client.post(EVENT_URL, json=body)
    assert closed_dialog.status_code == 409, closed_dialog.text
    assert closed_dialog.json()["detail"] == "album_leaf_review_group_stale"
    reopened = await client.post(EVENT_URL, json={**count_answer, "answer": "unsure", "idempotency_key": uuid4().hex})
    assert reopened.status_code == 201, reopened.text
    open_task, = (await client.get(queue_url)).json()
    assert open_task["payload"]["photo_count_revision"] == before["photo_count_revision"]

    confirmed = await client.post(EVENT_URL, json=body)
    assert confirmed.status_code == 201, confirmed.text
    event_id = confirmed.json()["id"]
    retry = await client.post(EVENT_URL, json=body)
    assert retry.status_code == 201, retry.text
    assert retry.json() == {"id": event_id, "saved": True, "duplicate": True}
    stale = await client.post(EVENT_URL, json={**body, "idempotency_key": uuid4().hex})
    assert stale.status_code == 409, stale.text

    open_task, = (await client.get(queue_url)).json()
    after = open_task["payload"]
    assert after["detection_count"] == 2 and after["photo_count_revision"] == event_id
    assert after["detections"] == [row for row in before["detections"] if row["photo_id"] != leaf_id]
    assert {row["id"] for row in (await client.get(f"{album_url}/photos")).json()} == {child_id, other_id}
    assert [(await client.get(url)).json() for url in (child_url, other_url)] == [child_before, other_before]
    remaining = [await client.get(f"{url}/image?variant=source_corrected_untrimmed") for url in (child_url, other_url)]
    assert [(r.status_code, r.content) for r in remaining] == [(200, r.content) for r in untouched]
    excluded = (await client.get(leaf_url)).json()
    assert excluded["reviewed_non_print"] is True
    assert excluded["enhancement_applied"]["review_rejected_detection"]["segmentation_mask"] == leaf_before["segmentation_mask"]
    assert excluded["enhancement_applied"]["review_decisions"]["geometry"][-1]["event_id"] == event_id
    await assert_held()

    async with get_session_factory()() as session:
        event = await session.get(ReviewEvent, UUID(event_id))
        assert event.event_metadata["client"]["album_leaf_group_photo_ids"] == [leaf_id]
        assert len(event.event_metadata["snapshot"]["photos"]) == 3
