from pathlib import Path
from uuid import UUID, uuid4

import cv2
import numpy as np
import pytest


async def test_recompressed_page_recovers_an_unconfirmed_print_draft_and_neighbor_traces(client, monkeypatch, tmp_path):
    from app.config import get_settings
    from app.pipeline.image_ops import load_image
    from app.pipeline.projective_prints import PROJECTIVE_PRINT_OUTLINE_REVIEW, PROJECTIVE_PRINT_PROPOSAL
    from app.services import pipeline

    monkeypatch.setenv("AUTO_ANALYZE_UPLOADS", "true")
    monkeypatch.setenv("MAX_UPLOAD_SIZE", "2000000")
    monkeypatch.setenv("MAX_REQUEST_SIZE", "2100000")
    monkeypatch.setenv("ENABLE_PHOTO_ORIENTATION_CORRECTION", "false")
    monkeypatch.setenv("YOLO_MODEL_PATH", str(tmp_path / "absent.onnx"))
    monkeypatch.setattr(pipeline, "run_paddle_ocr", lambda *_: [])
    get_settings.cache_clear()
    source = Path(__file__).resolve().parents[3] / "data" / "golden_fixtures" / "page_11.jpg"
    ok, jpeg = cv2.imencode(".jpg", load_image(source), [cv2.IMWRITE_JPEG_QUALITY, 75])
    assert ok
    album_id = (await client.post("/api/v1/albums", json={"name": "Unconfirmed source outline"})).json()["id"]
    album = f"/api/v1/albums/{album_id}"
    uploaded = await client.post(f"{album}/pages/upload", files={"files": ("page-75.jpg", jpeg.tobytes(), "image/jpeg")})
    assert uploaded.status_code == 201, uploaded.text
    review_url = f"/api/v1/review/queue?album_id={album_id}&type=segmentation"
    task, = (await client.get(review_url)).json()
    initial = task["payload"]
    assert initial["detection_count"] == 0
    assert not task["reason"]
    proposal, = initial["deferred_print_proposals"]
    assert proposal["reason"] == PROJECTIVE_PRINT_PROPOSAL
    assert len(initial["missing_print_edge_traces"]) >= 3

    manual_url = f"/api/v1/pages/{task['page_id']}/manual-photos"
    form = {"request_id": str(uuid4()), "warp_quad": proposal["polygon"], "deferred_proposal_id": proposal["id"],
            "photo_count_revision": initial["photo_count_revision"], "print_size_mm": {"width": 70, "height": 100}}
    saved = await client.post(manual_url, json=form)
    assert saved.status_code == 201, saved.text
    first = saved.json()
    full = first["enhancement_applied"]["correction"]["crop_modes"]["full_print"]
    assert PROJECTIVE_PRINT_OUTLINE_REVIEW in full["reason_codes"]
    assert first["segmentation_mask"]["projective_outline_review"] is True
    first_url = f"/api/v1/photos/{first['id']}"
    checked = await client.patch(f"{first_url}/bounding-box", json={
        "bounding_box": first["bounding_box"], "confirm_print_outline": True,
        "segmentation_mask": {"source": "manual_quad", "warp_quad": form["warp_quad"], "polygon": form["warp_quad"]},
    })
    assert checked.status_code == 200, checked.text
    assert checked.json()["segmentation_mask"]["projective_outline_confirmed"] is True
    assert PROJECTIVE_PRINT_OUTLINE_REVIEW in checked.json()["enhancement_applied"]["correction"]["crop_modes"]["full_print"]["reason_codes"]
    after = (await client.get(review_url)).json()[0]["payload"]
    assert after["detection_count"] == 1 and after["deferred_print_proposals"] == []
    assert after["missing_print_edge_traces"] == initial["missing_print_edge_traces"]

    right = [{"x": x / 2559, "y": y / 1919} for x, y in [(1539, 765), (2311, 749), (2271, 1835), (1552, 1859)]]
    added = await client.post(manual_url, json={
        "request_id": str(uuid4()), "warp_quad": right, "print_size_mm": {"width": 70, "height": 100},
    })
    assert added.status_code == 201, added.text
    second = added.json()
    assert second["enhancement_applied"]["review_provenance"]["paper_trace_review"]["sides"] == [1]
    final = (await client.get(review_url)).json()[0]["payload"]
    assert final["detection_count"] == 2 and final["missing_print_edge_traces"] == []
    for url in (first_url, f"/api/v1/photos/{second['id']}"):
        assert (await client.post(f"{url}/reprocess")).status_code == 200
        for variant in ("full_print", "image_only", "source_corrected_trimmed"):
            assert (await client.get(f"{url}/image?variant={variant}")).status_code == 404
        assert (await client.get(f"{url}/image?variant=source_corrected_untrimmed")).status_code == 200


async def test_compressed_quarter_turn_only_offers_the_measured_neighbor_edges(client, monkeypatch, tmp_path):
    from app.config import get_settings
    from app.pipeline.crossing_prints import _quarter_turn
    from app.pipeline.geometry import order_quad_cyclic
    from app.pipeline.image_ops import load_image
    from app.services import pipeline

    monkeypatch.setenv("AUTO_ANALYZE_UPLOADS", "true")
    monkeypatch.setenv("MAX_UPLOAD_SIZE", "2000000")
    monkeypatch.setenv("MAX_REQUEST_SIZE", "2100000")
    monkeypatch.setenv("ENABLE_PHOTO_ORIENTATION_CORRECTION", "false")
    monkeypatch.setenv("ENABLE_PRECISE_PHOTO_CROPS", "true")
    monkeypatch.setenv("ENABLE_CLASSICAL_SEGMENTATION_FALLBACK", "true")
    monkeypatch.setenv("YOLO_MODEL_PATH", str(tmp_path / "absent.onnx"))
    monkeypatch.setenv("CROP_VALIDATION_REPORT_PATH", str(tmp_path / "validation-unavailable.json"))
    monkeypatch.setattr(pipeline, "run_paddle_ocr", lambda *_: [])
    get_settings.cache_clear()

    album_id = (await client.post("/api/v1/albums", json={"name": "Measured neighbor without a draft"})).json()["id"]
    album = f"/api/v1/albums/{album_id}"
    path = Path(__file__).resolve().parents[3] / "data" / "golden_fixtures" / "page_11.jpg"
    ok, jpeg = cv2.imencode(".jpg", np.rot90(load_image(path)), [cv2.IMWRITE_JPEG_QUALITY, 55])
    assert ok
    source = jpeg.tobytes()
    uploaded = await client.post(f"{album}/pages/upload", files={"files": ("quarter-turn.jpg", source, "image/jpeg")})
    assert uploaded.status_code == 201, uploaded.text
    assert (await client.get(f"{album}/photos")).json() == []
    review_url = f"/api/v1/review/queue?album_id={album_id}&type=segmentation"
    task, = (await client.get(review_url)).json()
    initial = task["payload"]
    traces = initial["missing_print_edge_traces"]
    assert task["reason"] == "No photos detected; manual review required."
    assert initial["detection_count"] == 0 and initial["deferred_print_proposals"] == []
    assert initial["photo_count_revision"] is None
    assert len(traces) >= 3
    assert all(.38 < point["x"] <= .975 and .09 < point["y"] < .41 for run in traces for point in run)

    page_url = f"/api/v1/pages/{task['page_id']}"
    original = await client.get(f"{page_url}/image")
    assert original.status_code == 200 and original.content == source
    right = np.float32([[1539, 765], [2311, 749], [2271, 1835], [1552, 1859]])
    rotated = order_quad_cyclic(_quarter_turn(right, 1, 2560, 1920))
    manual = await client.post(f"{page_url}/manual-photos", json={
        "request_id": str(uuid4()),
        "warp_quad": [{"x": float(x / 1919), "y": float(y / 2559)} for x, y in rotated],
        "print_size_mm": {"width": 100, "height": 70},
    })
    assert manual.status_code == 201, manual.text
    stored = manual.json()
    after = (await client.get(review_url)).json()[0]["payload"]
    assert after["detection_count"] == 1 and after["deferred_print_proposals"] == []
    assert after["missing_print_edge_traces"] == []
    assert stored["enhancement_applied"]["review_provenance"]["paper_trace_review"]["sides"]
    url = f"/api/v1/photos/{stored['id']}"
    assert (await client.post(f"{url}/reprocess")).status_code == 200
    for variant in ("full_print", "image_only", "source_corrected_trimmed"):
        assert (await client.get(f"{url}/image?variant={variant}")).status_code == 404
    assert (await client.get(f"{url}/image?variant=source_corrected_untrimmed")).status_code == 200


async def test_adjacent_stock_draft_requires_explicit_paper_confirmation(client, monkeypatch):
    from app.config import get_settings
    from app.evaluation.crop_fixtures import synthetic_crop_pages
    from app.pipeline.segmentation import SegmentationDetection, SegmentationResult
    from app.services import pipeline

    monkeypatch.setenv("AUTO_ANALYZE_UPLOADS", "true")
    monkeypatch.setenv("MAX_UPLOAD_SIZE", "1048576")
    monkeypatch.setenv("MAX_REQUEST_SIZE", "2097152")
    monkeypatch.setenv("ENABLE_PRECISE_PHOTO_CROPS", "true")
    monkeypatch.setenv("ENABLE_CLASSICAL_SEGMENTATION_FALLBACK", "true")
    monkeypatch.setenv("ENABLE_PROTECTED_REGION_GUARD", "false")
    monkeypatch.setenv("ENABLE_PHOTO_ORIENTATION_CORRECTION", "false")
    monkeypatch.setattr(pipeline, "run_paddle_ocr", lambda *_: [])
    get_settings.cache_clear()

    fixture = next(page for page in synthetic_crop_pages("locked") if page.name == "touching")
    image = fixture.image
    height, width = image.shape[:2]
    seeds = (((86, 196), (413, 196), (414, 420), (86, 420)),
             ((419, 230), (697, 230), (697, 425), (419, 425)))
    detections = []
    for seed in seeds:
        polygon = [{"x": x / (width - 1), "y": y / (height - 1)} for x, y in seed]
        xs, ys = zip(*seed, strict=True)
        bounds = {"x1": min(xs) / width, "y1": min(ys) / height,
                  "x2": max(xs) / width, "y2": max(ys) / height}
        detections.append(SegmentationDetection(bounds, {"source": "yolo_seg_mask", "polygon": polygon}, .9, 1., True, []))
    monkeypatch.setattr(pipeline, "segment_album_page", lambda *_, **__: SegmentationResult(detections, {}))
    ok, png = cv2.imencode(".png", image)
    assert ok
    album_id = (await client.post("/api/v1/albums", json={"name": "Two neighboring paper stocks"})).json()["id"]
    album = f"/api/v1/albums/{album_id}"
    uploaded = await client.post(f"{album}/pages/upload", files={"files": ("touching.png", png.tobytes(), "image/png")})
    assert uploaded.status_code == 201, uploaded.text
    left, right = sorted((await client.get(f"{album}/photos")).json(), key=lambda row: row["bounding_box"]["x1"])
    left_url = f"/api/v1/photos/{left['id']}"
    right_url = f"/api/v1/photos/{right['id']}"
    other = (await client.get(right_url)).json()
    full = left["enhancement_applied"]["correction"]["crop_modes"]["full_print"]
    assert full["suggested_outline_source"] == "adjacent_paper_stock"
    proposal = full["suggested_warp_quad"]
    assert proposal != full["warp_quad"]
    for url in (left_url, right_url):
        for variant in ("full_print", "image_only", "source_corrected_trimmed"):
            assert (await client.get(f"{url}/image?variant={variant}")).status_code == 404
    update = {"bounding_box": left["bounding_box"], "segmentation_mask": {
        "source": "manual_quad", "warp_quad": proposal, "polygon": proposal, "print_size_mm": {"width": 166, "height": 114},
    }}
    implicit = await client.patch(f"{left_url}/bounding-box", json=update)
    assert implicit.status_code == 409 and implicit.json()["detail"] == "print_outline_review_required"
    saved = await client.patch(f"{left_url}/bounding-box", json={**update, "confirm_print_outline": True})
    assert saved.status_code == 200, saved.text
    for variant in ("full_print", "image_only"):
        assert (await client.get(f"{left_url}/image?variant={variant}")).status_code == 200
    assert (await client.get(right_url)).json() == other


@pytest.mark.parametrize("kind", ["film", "paper"])
async def test_review_excludes_only_the_confirmed_group(client, monkeypatch, kind):
    from app.config import get_settings
    from app.database import get_session_factory
    from app.models import ReviewEvent
    from app.pipeline.multipanel_prints import FILM_GRID_REVIEW, PAPER_GRID_REVIEW, PAPER_JOIN_REVIEW
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

    def vertices(left, top, right, bottom):
        return [{"x": x / 399, "y": y / 259} for x, y in ((left, top), (right, top), (right, bottom), (left, bottom))]

    linked = [vertices(25, 30, 179, 121), vertices(26, 131, 181, 228)]
    if kind == "film":
        linked.insert(0, vertices(20, 24, 184, 233))
    paper = [vertices(206, 21, 375, 234), vertices(218, 132, 362, 232)]
    seam = [{"x": 0.08, "y": 0.48}, {"x": 0.45, "y": 0.49}]
    traces = [[{"x": x, "y": .49}, {"x": x, "y": .54}] for x in (.05, .46)]
    mask_key = "film_grid_review_seam" if kind == "film" else "paper_join_review_traces"
    reason = FILM_GRID_REVIEW if kind == "film" else PAPER_JOIN_REVIEW
    detections = []
    for polygon in [*linked, *paper]:
        bounds = {"x1": polygon[0]["x"], "y1": polygon[0]["y"], "x2": polygon[2]["x"], "y2": polygon[2]["y"]}
        mask = {"source": "yolo_seg_mask", "polygon": polygon}
        if polygon in linked:
            mask[mask_key] = seam if kind == "film" else traces
        detections.append(SegmentationDetection(
            bounds, mask, 0.81, 1.0, False, [reason if polygon in linked else PAPER_GRID_REVIEW],
        ))
    monkeypatch.setattr(pipeline, "segment_album_page", lambda *_, **__: SegmentationResult(detections, {}))
    image = np.full((260, 400, 3), 42, np.uint8)
    image[24:234, 20:185] = (53, 82, 110)
    image[21:235, 206:376] = (89, 137, 186)
    ok, png = cv2.imencode(".png", image)
    assert ok

    album_id = (await client.post("/api/v1/albums", json={"name": "Linked proposal review"})).json()["id"]
    album_url = f"/api/v1/albums/{album_id}"
    uploaded = await client.post(f"{album_url}/pages/upload", files={"files": ("page.png", png.tobytes(), "image/png")})
    assert uploaded.status_code == 201, uploaded.text
    queue_url = f"/api/v1/review/queue?album_id={album_id}&type=segmentation"
    task, = (await client.get(queue_url)).json()
    before = task["payload"]
    assert before["detection_count"] == len(detections)
    group_ids = [row["photo_id"] for row in before["detections"] if row["segmentation_mask"].get(mask_key)]
    paper_before = [row for row in before["detections"] if row["photo_id"] not in group_ids]
    assert len(group_ids) == len(linked) and len(paper_before) == 2

    event_url = "/api/v1/review/events"
    ids_key = f"{kind}_group_photo_ids"
    metadata = {"ui": "desktop", "photo_count_revision": before["photo_count_revision"], ids_key: group_ids}
    answer = {"task_type": "segmentation", "task_id": task["id"], "answer": f"exclude_{kind}_group",
              "reason_code": "film_carrier" if kind == "film" else "non_photo_object",
              "idempotency_key": uuid4().hex, "metadata": metadata}
    incomplete = await client.post(event_url, json={**answer, "metadata": {**metadata, ids_key: group_ids[:-1]}})
    assert incomplete.status_code == 422
    for selection in ([*group_ids[:-1], paper_before[0]["photo_id"]], [group_ids[0]] * len(group_ids)):
        wrong = await client.post(event_url, json={**answer, "metadata": {**metadata, ids_key: selection}})
        assert wrong.status_code == 409 and wrong.json()["detail"] == f"{kind}_review_group_stale"
    assert (await client.get(queue_url)).json() == [task]

    confirmed = await client.post(event_url, json=answer)
    assert confirmed.status_code == 201, confirmed.text
    event_id = confirmed.json()["id"]
    retry = await client.post(event_url, json=answer)
    assert retry.json() == {"id": event_id, "saved": True, "duplicate": True}
    stale = await client.post(event_url, json={**answer, "idempotency_key": uuid4().hex})
    assert stale.status_code == 409

    current, = (await client.get(queue_url)).json()
    after = current["payload"]
    assert after["detection_count"] == 2 and after["photo_count_revision"] == event_id
    assert after["detections"] == paper_before
    rows = (await client.get(f"{album_url}/photos")).json()
    assert {row["id"] for row in rows} == {row["photo_id"] for row in paper_before}
    for photo_id in group_ids:
        url = f"/api/v1/photos/{photo_id}"
        photo = (await client.get(url)).json()
        assert photo["reviewed_non_print"] is True
        saved = photo["enhancement_applied"]
        assert saved["review_rejected_detection"]["segmentation_mask"] == photo["segmentation_mask"]
        assert saved["review_decisions"]["geometry"][-1]["event_id"] == event_id
        for mode in ("full_print", "image_only", "source_corrected_trimmed"):
            assert (await client.get(f"{url}/image?variant={mode}")).status_code == 404
        assert (await client.get(f"{url}/image?variant=source_corrected_untrimmed")).status_code == 200

    async with get_session_factory()() as session:
        event = await session.get(ReviewEvent, UUID(event_id))
        assert event.event_metadata["client"][ids_key] == group_ids
        assert len(event.event_metadata["snapshot"]["photos"]) == len(detections)

    count = await client.post(event_url, json={
        **answer, "answer": "correct", "idempotency_key": uuid4().hex, "metadata": {"photo_count_revision": event_id},
    })
    assert count.status_code == 201, count.text
    assert (await client.get(queue_url)).json() == []

    original = next(row for row in before["detections"] if row["photo_id"] == group_ids[0])
    restored = await client.patch(f"/api/v1/photos/{original['photo_id']}/bounding-box", json={
        "bounding_box": original["bounding_box"],
        "segmentation_mask": {"source": "manual_quad", "polygon": original["polygon"], "warp_quad": original["polygon"]},
    })
    assert restored.status_code == 200, restored.text
    assert restored.json()["reviewed_non_print"] is False
    reopened, = (await client.get(queue_url)).json()
    assert reopened["payload"]["detection_count"] == 3


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


async def test_review_can_replace_mounted_fragments_with_one_print(client, monkeypatch):
    from app.config import get_settings
    from app.pipeline.segmentation import SegmentationDetection, SegmentationResult
    from app.routers import photos
    from app.services import pipeline

    monkeypatch.setenv("AUTO_ANALYZE_UPLOADS", "true")
    monkeypatch.setenv("MAX_UPLOAD_SIZE", "1048576")
    monkeypatch.setenv("MAX_REQUEST_SIZE", "2097152")
    monkeypatch.setenv("ENABLE_PRECISE_PHOTO_CROPS", "true")
    monkeypatch.setenv("ENABLE_PHOTO_ORIENTATION_CORRECTION", "false")
    monkeypatch.setenv("ENABLE_PROTECTED_REGION_GUARD", "false")
    monkeypatch.setattr(pipeline, "run_paddle_ocr", lambda *_: [])
    get_settings.cache_clear()
    image = np.full((260, 400, 3), 34, np.uint8)
    image[30:191, 28:236] = (232, 238, 239)
    image[42:178, 41:222] = (105, 95, 138)
    image[90:220, 267:375] = (67, 102, 148)

    def outline(x1, y1, x2, y2):
        return [{"x": x / 399, "y": y / 259} for x, y in ((x1, y1), (x2, y1), (x2, y2), (x1, y2))]

    def bounds(quad):
        return {"x1": quad[0]["x"], "x2": quad[2]["x"], "y1": quad[0]["y"], "y2": quad[2]["y"]}

    outer = outline(28, 30, 235, 190)
    fragments = [outline(41, 42, 98, 177), outline(102, 42, 156, 177), outline(160, 42, 221, 177)]
    independent = outline(267, 90, 374, 219)
    detections = [
        SegmentationDetection(bounds(quad), {"source": "yolo_seg_mask", "polygon": quad}, .88, 1.0, True, [])
        for quad in [*fragments, independent]
    ]
    pending = [{"id": 3, "reason": "possible_mounted_print", "polygon": outer, "fragment_polygons": fragments}]
    monkeypatch.setattr(pipeline, "segment_album_page", lambda *_, **__: SegmentationResult(
        detections, {"fusion": {"deferred_print_proposals": pending}},
    ))
    ok, png = cv2.imencode(".png", image)
    assert ok
    album_id = (await client.post("/api/v1/albums", json={"name": "Mounted print review"})).json()["id"]
    base = f"/api/v1/albums/{album_id}"
    upload = await client.post(f"{base}/pages/upload", files={"files": ("page.png", png.tobytes(), "image/png")})
    assert upload.status_code == 201
    page, = (await client.get(f"{base}/pages")).json()
    manual_url = f"/api/v1/pages/{page['id']}/manual-photos"
    photo_list = f"{base}/photos"
    review_url = f"/api/v1/review/queue?album_id={album_id}&type=segmentation"
    before = (await client.get(review_url)).json()[0]["payload"]
    initial = (await client.get(photo_list)).json()
    fragment_ids = {row["id"] for row in initial if row["segmentation_mask"]["polygon"] in fragments}
    independent_id, = {row["id"] for row in initial} - fragment_ids
    assert before["detection_count"] == 4 and len(fragment_ids) == 3
    body = {
        "request_id": str(uuid4()), "deferred_proposal_id": 3, "warp_quad": outer,
        "photo_count_revision": before["photo_count_revision"], "replace_mounted_fragments": True,
        "print_size_mm": {"width": 132, "height": 102},
    }
    excluded = await client.post(manual_url, json={**body, "warp_quad": outline(28, 30, 166, 190)})
    assert excluded.status_code == 422 and excluded.json()["detail"] == "mounted_print_outline_excludes_fragments"

    async def fail_render(*_, **__):
        raise ValueError("simulated invalid crop")

    with monkeypatch.context() as faults:
        faults.setattr(photos, "perspective_correction_task_logic", fail_render)
        failed = await client.post(manual_url, json=body)
    assert failed.status_code == 422
    assert (await client.get(photo_list)).json() == initial
    assert (await client.get(review_url)).json()[0]["payload"] == before

    saved = await client.post(manual_url, json=body)
    assert saved.status_code == 201, saved.text
    replacement = saved.json()
    replacement_id = replacement["id"]
    assert replacement["crop_download_urls"] == {"full_print": None, "image_only": None}
    assert "MOUNTED_PRINT_OUTLINE_REVIEW" in replacement["enhancement_applied"]["correction"]["crop_modes"]["full_print"]["reason_codes"]
    creation = replacement["enhancement_applied"]["review_provenance"]["manual_creation"]
    assert set(creation["replaced_fragment_ids"]) == fragment_ids
    assert {row["id"] for row in (await client.get(photo_list)).json()} == {replacement_id, independent_id}
    after = (await client.get(review_url)).json()[0]["payload"]
    assert after["detection_count"] == 2 and after["deferred_print_proposals"] == []
    for fragment in fragment_ids:
        row = (await client.get(f"/api/v1/photos/{fragment}")).json()
        assert row["reviewed_non_print"]
        assert row["segmentation_mask"]["polygon"] in fragments
    repeated = await client.post(manual_url, json=body)
    assert repeated.status_code == 200 and repeated.json()["id"] == replacement_id
    changed = await client.post(manual_url, json={**body, "replace_mounted_fragments": False})
    assert changed.status_code == 409

    motif = outline(41, 42, 221, 177)
    photo_url = f"/api/v1/photos/{replacement_id}"
    edit = {
        "bounding_box": bounds(outer),
        "segmentation_mask": {"source": "manual_quad", "polygon": outer, "warp_quad": outer, "image_quad": motif},
    }
    unconfirmed = await client.patch(f"{photo_url}/bounding-box", json=edit)
    assert unconfirmed.status_code == 200 and not unconfirmed.json()["crop_download_urls"]["full_print"]
    corrected = await client.patch(f"{photo_url}/bounding-box", json={**edit, "confirm_print_outline": True})
    assert corrected.status_code == 200, corrected.text
    for mode in ("full_print", "image_only"):
        assert (await client.get(f"{photo_url}/image", params={"variant": mode})).status_code == 200

    fragment_id = next(row["id"] for row in initial if row["segmentation_mask"]["polygon"] == fragments[0])
    fragment_url = f"/api/v1/photos/{fragment_id}"
    restored = await client.patch(f"{fragment_url}/bounding-box", json={
        "bounding_box": bounds(fragments[0]),
        "segmentation_mask": {"source": "manual_quad", "polygon": fragments[0], "warp_quad": fragments[0]},
    })
    assert restored.status_code == 200 and not restored.json()["reviewed_non_print"]
    assert {row["id"] for row in (await client.get(photo_list)).json()} == {replacement_id, independent_id, fragment_id}
    for mode in ("full_print", "image_only"):
        assert (await client.get(f"{photo_url}/image", params={"variant": mode})).status_code == 404
