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


async def test_source_corner_preview_is_held_and_retires_for_a_new_claim(client, monkeypatch, tmp_path):
    from app.config import get_settings
    from app.services import pipeline

    monkeypatch.setenv("AUTO_ANALYZE_UPLOADS", "true")
    monkeypatch.setenv("MAX_UPLOAD_SIZE", "2000000")
    monkeypatch.setenv("MAX_REQUEST_SIZE", "2100000")
    monkeypatch.setenv("ENABLE_PHOTO_ORIENTATION_CORRECTION", "false")
    monkeypatch.setenv("ENABLE_PRECISE_PHOTO_CROPS", "true")
    monkeypatch.setenv("ENABLE_CLASSICAL_SEGMENTATION_FALLBACK", "true")
    monkeypatch.setenv("ENABLE_PROTECTED_REGION_GUARD", "true")
    monkeypatch.setenv("YOLO_MODEL_PATH", str(tmp_path / "absent.onnx"))
    monkeypatch.setattr(pipeline, "run_paddle_ocr", lambda *_: [])
    get_settings.cache_clear()

    source = Path(__file__).resolve().parents[3] / "data" / "golden_fixtures" / "page_05.jpg"
    album_id = (await client.post("/api/v1/albums", json={"name": "Partial contour review"})).json()["id"]
    album = f"/api/v1/albums/{album_id}"
    uploaded = await client.post(f"{album}/pages/upload", files={"files": (source.name, source.read_bytes(), "image/jpeg")})
    assert uploaded.status_code == 201, uploaded.text
    photos = (await client.get(f"{album}/photos")).json()
    assert len(photos) == 5
    row, = [photo for photo in photos if photo["enhancement_applied"]["correction"]["crop_modes"]["full_print"]
            .get("suggested_outline_source") == "possible_inward_print_corner"]
    url = f"/api/v1/photos/{row['id']}"
    detail = (await client.get(url)).json()
    modes = detail["enhancement_applied"]["correction"]["crop_modes"]
    full = modes["full_print"]
    assert full["suggested_edge_source"] == "competing_print_edges"
    assert len(full["suggested_edge_traces"]) == 2
    assert full["warp_quad"][2]["y"] < .26
    proposal = full["suggested_warp_quad"]
    assert np.linalg.norm([2559 * proposal[2]["x"] - 1265.86, 1919 * proposal[2]["y"] - 759.85]) < 3.
    assert full["outline_preview"]["source_outline"] == proposal
    preview_url = detail["urls"]["full_print_preview"]
    preview = await client.get(preview_url)
    assert preview.status_code == 200 and preview.headers["content-type"] == "image/png"
    assert (await client.get(detail["urls"]["source_corrected_untrimmed"])).status_code == 200
    for variant in ("full_print", "image_only", "source_corrected_trimmed"):
        assert (await client.get(f"{url}/image?variant={variant}")).status_code == 404
    implicit = await client.patch(f"{url}/bounding-box", json={
        "bounding_box": row["bounding_box"],
        "segmentation_mask": {"source": "manual_quad", "polygon": proposal, "warp_quad": proposal},
    })
    assert implicit.status_code == 409 and implicit.json()["detail"] == "print_outline_review_required"
    queue = f"/api/v1/review/queue?album_id={album_id}&type=segmentation"
    pending, = (await client.get(queue)).json()[0]["payload"]["deferred_print_proposals"]
    manual_url = f"/api/v1/pages/{row['page_id']}/manual-photos"

    in_added_region = [{"x": x / 2559, "y": y / 1919}
                       for x, y in ((1050, 657), (1190, 657), (1190, 729), (1050, 729))]
    manual = await client.post(manual_url, json={
        "request_id": str(uuid4()), "warp_quad": in_added_region,
    })
    assert manual.status_code == 201, manual.text
    for reprocess in (False, True):
        if reprocess:
            assert (await client.post(f"{url}/reprocess")).status_code == 200
        current = (await client.get(url)).json()
        corrected = current["enhancement_applied"]["correction"]["crop_modes"]
        assert "suggested_warp_quad" not in corrected["full_print"]
        assert "outline_preview" not in corrected["full_print"]
        assert corrected["full_print"]["suggested_edge_traces"] == full["suggested_edge_traces"]
        assert corrected["full_print"]["warp_quad"] == full["warp_quad"]
        assert corrected["image_only"] == modes["image_only"]
        assert (await client.get(preview_url)).status_code == 404
        assert (await client.get(f"{url}/image?variant=full_print")).status_code == 404
    latest = (await client.get(queue)).json()[0]["payload"]
    assert latest["deferred_print_proposals"] == [pending]
    intersecting_claim = [{"x": x / 2559, "y": y / 1919}
                          for x, y in ((2240, 1725), (2300, 1725), (2300, 1810), (2240, 1810))]
    overlap = await client.post(manual_url, json={"request_id": str(uuid4()), "warp_quad": intersecting_claim})
    assert overlap.status_code == 201, overlap.text
    invalidated = (await client.get(queue)).json()[0]["payload"]
    assert invalidated["deferred_print_proposals"] == []
    assert invalidated["detection_count"] == latest["detection_count"] + 1
    stale = await client.post(manual_url, json={
        "request_id": str(uuid4()), "warp_quad": pending["polygon"], "deferred_proposal_id": pending["id"],
        "photo_count_revision": latest["photo_count_revision"], "replace_mounted_fragments": True,
    })
    assert stale.status_code == 409 and stale.json()["detail"] == "deferred_print_proposal_stale"


async def _upload_source_fragment_review_page(client, monkeypatch, tmp_path):
    from app.config import get_settings
    from app.services import pipeline

    for name, value in {
        "AUTO_ANALYZE_UPLOADS": "true", "MAX_UPLOAD_SIZE": "2000000", "MAX_REQUEST_SIZE": "2100000",
        "ENABLE_PHOTO_ORIENTATION_CORRECTION": "false", "ENABLE_PRECISE_PHOTO_CROPS": "true",
        "ENABLE_CLASSICAL_SEGMENTATION_FALLBACK": "true", "ENABLE_PROTECTED_REGION_GUARD": "true",
        "YOLO_MODEL_PATH": str(tmp_path / "absent.onnx"),
    }.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(pipeline, "run_paddle_ocr", lambda *_: [])
    get_settings.cache_clear()
    source = Path(__file__).resolve().parents[3] / "data" / "golden_fixtures" / "page_05.jpg"
    album_id = (await client.post("/api/v1/albums", json={"name": "One fragmented print"})).json()["id"]
    album = f"/api/v1/albums/{album_id}"
    uploaded = await client.post(f"{album}/pages/upload", files={"files": (source.name, source.read_bytes(), "image/jpeg")})
    assert uploaded.status_code == 201, uploaded.text
    queue = f"/api/v1/review/queue?album_id={album_id}&type=segmentation"
    task, = (await client.get(queue)).json()
    return album, queue, task


async def test_source_fragment_group_replaces_only_its_two_held_rows(client, monkeypatch, tmp_path):
    from app.pipeline.fragmented_prints import SOURCE_FRAGMENT_GROUP_REVIEW, SOURCE_FRAGMENT_PROPOSAL
    from app.routers import photos

    album, queue, task = await _upload_source_fragment_review_page(client, monkeypatch, tmp_path)
    initial = task["payload"]
    proposal, = initial["deferred_print_proposals"]
    assert proposal["reason"] == SOURCE_FRAGMENT_PROPOSAL and proposal["uncertain_edge"] == 1
    assert initial["detection_count"] == 5
    rows = (await client.get(f"{album}/photos")).json()
    fragments = [row for row in rows if row["segmentation_mask"].get("source_fragment_group_review") == proposal["id"]]
    assert [row["segmentation_mask"]["polygon"] for row in fragments] == proposal["fragment_polygons"]
    for row in fragments:
        for mode in ("full_print", "image_only"):
            result = row["enhancement_applied"]["correction"]["crop_modes"][mode]
            assert result["status"] == "review_needed" and SOURCE_FRAGMENT_GROUP_REVIEW in result["reason_codes"]

    manual = f"/api/v1/pages/{task['page_id']}/manual-photos"
    form = {"request_id": str(uuid4()), "warp_quad": proposal["polygon"], "deferred_proposal_id": proposal["id"],
            "photo_count_revision": initial["photo_count_revision"]}
    for attempt in (form, {"request_id": form["request_id"], "warp_quad": form["warp_quad"]}):
        response = await client.post(manual, json=attempt)
        assert response.status_code == 409 and response.json()["detail"] == "source_print_fragments_confirmation_required"
    confirmed = {**form, "replace_mounted_fragments": True}
    wrong_outline = await client.post(manual, json={**confirmed, "warp_quad": proposal["fragment_polygons"][0]})
    assert wrong_outline.status_code == 422 and wrong_outline.json()["detail"] == "source_print_outline_excludes_fragments"

    async def fail_render(*args, **kwargs):
        raise ValueError("test_render_failure")

    with monkeypatch.context() as failure:
        failure.setattr(photos, "perspective_correction_task_logic", fail_render)
        failed = await client.post(manual, json=confirmed)
        assert failed.status_code == 422 and failed.json()["detail"] == "test_render_failure"
    assert (await client.get(queue)).json() == [task]
    assert (await client.get(f"{album}/photos")).json() == rows

    saved = await client.post(manual, json=confirmed)
    assert saved.status_code == 201, saved.text
    replacement = saved.json()
    retry = await client.post(manual, json=confirmed)
    assert retry.status_code == 200 and retry.json() == replacement
    assert (await client.post(manual, json={**confirmed, "request_id": str(uuid4())})).status_code == 409
    after, = (await client.get(queue)).json()
    assert after["payload"]["detection_count"] == 4 and after["payload"]["deferred_print_proposals"] == []
    excluded = {row["id"] for row in fragments}
    kept = [row for row in (await client.get(f"{album}/photos")).json() if row["id"] != replacement["id"]]
    assert kept == [row for row in rows if row["id"] not in excluded]
    for photo_id in excluded | {replacement["id"]}:
        url = f"/api/v1/photos/{photo_id}"
        for variant in ("full_print", "image_only", "source_corrected_trimmed"):
            assert (await client.get(f"{url}/image?variant={variant}")).status_code == 404
    replacement_url = f"/api/v1/photos/{replacement['id']}"
    assert (await client.post(f"{replacement_url}/reprocess")).status_code == 200
    assert (await client.get(f"{replacement_url}/image?variant=source_corrected_untrimmed")).status_code == 200
    assert (await client.post("/api/v1/review/events", json={
        "task_type": "geometry", "task_id": replacement["id"], "answer": "correct", "idempotency_key": uuid4().hex,
    })).status_code == 409


async def test_overwriting_a_source_fragment_retires_the_shared_offer(client, monkeypatch, tmp_path):
    _, queue, task = await _upload_source_fragment_review_page(client, monkeypatch, tmp_path)
    initial = task["payload"]
    proposal, = initial["deferred_print_proposals"]
    carrier = next(row for row in initial["detections"] if row["polygon"] == proposal["fragment_polygons"][0])
    quad = proposal["polygon"]
    box = {"x1": min(p["x"] for p in quad), "y1": min(p["y"] for p in quad),
           "x2": max(p["x"] for p in quad), "y2": max(p["y"] for p in quad)}
    updated = await client.patch(f"/api/v1/photos/{carrier['photo_id']}/bounding-box", json={
        "bounding_box": box, "segmentation_mask": {"source": "manual_quad", "warp_quad": quad, "polygon": quad,
                                                   "print_size_mm": {"width": 150, "height": 100}},
    })
    assert updated.status_code == 200, updated.text
    for variant in ("full_print", "image_only", "source_corrected_trimmed"):
        url = f"/api/v1/photos/{carrier['photo_id']}/image?variant={variant}"
        assert (await client.get(url)).status_code == 404
    current = (await client.get(queue)).json()[0]["payload"]
    assert current["detection_count"] == 5 and current["deferred_print_proposals"] == []
    assert current["photo_count_revision"] != initial["photo_count_revision"]
    stale = await client.post(f"/api/v1/pages/{task['page_id']}/manual-photos", json={
        "request_id": str(uuid4()), "warp_quad": quad, "deferred_proposal_id": proposal["id"],
        "photo_count_revision": initial["photo_count_revision"], "replace_mounted_fragments": True,
    })
    assert stale.status_code == 409 and stale.json()["detail"] == "deferred_print_proposal_stale"


async def test_dark_album_count_offer_needs_confirmation_and_then_a_separate_paper_check(client, monkeypatch):
    from app.config import get_settings
    from app.pipeline.dark_album_review import DARK_ALBUM_PROPOSAL, DARK_BACKING_OUTLINE_REVIEW
    from app.pipeline.segmentation import SegmentationDetection, SegmentationResult, _defer_dark_album_claims
    from app.services import pipeline

    for name, value in {
        "AUTO_ANALYZE_UPLOADS": "true", "ENABLE_PRECISE_PHOTO_CROPS": "true",
        "ENABLE_PROTECTED_REGION_GUARD": "false", "ENABLE_PHOTO_ORIENTATION_CORRECTION": "false",
        "MAX_UPLOAD_SIZE": "1048576", "MAX_REQUEST_SIZE": "2097152",
    }.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(pipeline, "run_paddle_ocr", lambda *_: [])
    get_settings.cache_clear()
    image = np.full((1000, 1200, 3), 25, np.uint8)
    image[140:570, 110:445] = (213, 220, 234)
    image[570:900, 445:980] = (180, 211, 228)

    def detection(x1, y1, x2, y2, cluster_id, *, corroborated):
        vertices = [{"x": x / 1199, "y": y / 999} for x, y in ((x1, y1), (x2, y1), (x2, y2), (x1, y2))]
        box = dict(x1=x1 / 1199, x2=x2 / 1199, y1=y1 / 999, y2=y2 / 999)
        return SegmentationDetection(box, {
            "source": "classical_border_quad" if corroborated else "yolo_seg_mask",
            "detector_source": "edge_contour" if corroborated else "yolo_onnx", "polygon": vertices,
            "fusion": {"proposal_families": ["classical", "model"] if corroborated else ["model"],
                       "proposal_count": 2 if corroborated else 1, "cluster_id": cluster_id},
        }, .83 if corroborated else .52, 1., True, [])

    neighbors = [detection(110, 140, 445, 570, 0, corroborated=True),
                 detection(445, 570, 980, 900, 1, corroborated=True)]
    suspect = detection(445, 140, 980, 570, 2, corroborated=False)
    source_result = SegmentationResult([*neighbors, suspect], {
        "fallback_used": True, "fusion": {"model_proposals": 3, "classical_proposals": 2},
    })
    crop_source = image.copy()
    crop_source[140:570, 445:980] = 46
    assert _defer_dark_album_claims(crop_source, source_result, .70) == source_result
    assert _defer_dark_album_claims(image, source_result, .50) == source_result
    monkeypatch.setattr(pipeline, "segment_album_page", lambda _, __, source_image, **_kwargs:
                        _defer_dark_album_claims(source_image, source_result, .70))

    album_id = (await client.post("/api/v1/albums", json={"name": "Dark backing count review"})).json()["id"]
    album = f"/api/v1/albums/{album_id}"
    ok, png = cv2.imencode(".png", image)
    assert ok
    uploaded = await client.post(f"{album}/pages/upload", files={"files": ("dark-stock.png", png.tobytes(), "image/png")})
    assert uploaded.status_code == 201, uploaded.text
    queue_url = f"/api/v1/review/queue?album_id={album_id}&type=segmentation"
    task, = (await client.get(queue_url)).json()
    initial = task["payload"]
    assert initial["detection_count"] == 2
    proposal, = initial["deferred_print_proposals"]
    assert proposal["reason"] == DARK_ALBUM_PROPOSAL and proposal["fusion_cluster_id"] == 2
    assert len(proposal["dark_backing_traces"]) == len(proposal["borrowed_edge_traces"]) == 2
    original = (await client.get(f"{album}/photos")).json()

    quad = proposal["polygon"]
    manual_url = f"/api/v1/pages/{task['page_id']}/manual-photos"
    payload = {"request_id": str(uuid4()), "warp_quad": quad, "deferred_proposal_id": proposal["id"],
               "photo_count_revision": initial["photo_count_revision"], "print_size_mm": {"width": 90, "height": 70}}
    without_link = {key: payload[key] for key in ("request_id", "warp_quad")}
    for request in (payload, without_link):
        blocked = await client.post(manual_url, json=request)
        assert blocked.status_code == 409 and blocked.json()["detail"] == "dark_backing_confirmation_required"
    wrong_outline = await client.post(manual_url, json={**payload, "confirm_dark_backing": True,
                                                       "warp_quad": neighbors[0].mask["polygon"]})
    assert wrong_outline.status_code == 422 and wrong_outline.json()["detail"] == "dark_backing_outline_outside_proposal"
    assert (await client.get(queue_url)).json() == [task]

    confirmed = {**payload, "confirm_dark_backing": True}
    saved = await client.post(manual_url, json=confirmed)
    assert saved.status_code == 201, saved.text
    restored = saved.json()
    retry = await client.post(manual_url, json=confirmed)
    assert retry.status_code == 200 and retry.json() == restored
    remaining = (await client.get(queue_url)).json()[0]["payload"]
    assert remaining["detection_count"] == 3 and remaining["deferred_print_proposals"] == []
    assert [row for row in (await client.get(f"{album}/photos")).json() if row["id"] != restored["id"]] == original
    url = f"/api/v1/photos/{restored['id']}"
    edited = await client.patch(f"{url}/bounding-box", json={
        "bounding_box": restored["bounding_box"], "confirm_print_outline": True,
        "segmentation_mask": {"source": "manual_quad", "warp_quad": quad, "polygon": quad},
    })
    assert edited.status_code == 200 and edited.json()["segmentation_mask"]["dark_backing_outline_confirmed"] is True
    assert (await client.post(f"{url}/reprocess")).status_code == 200
    photo = (await client.get(url)).json()
    modes = photo["enhancement_applied"]["correction"]["crop_modes"]
    assert all(mode["status"] == "review_needed" and DARK_BACKING_OUTLINE_REVIEW in mode["reason_codes"]
               for mode in modes.values())
    for variant in ("full_print", "image_only", "source_corrected_trimmed"):
        assert (await client.get(f"{url}/image?variant={variant}")).status_code == 404
    assert (await client.get(f"{url}/image?variant=source_corrected_untrimmed")).status_code == 200


async def test_shared_paper_grid_count_confirmation_preserves_the_outer_crop_for_review(client, monkeypatch):
    from app.config import get_settings
    from app.database import get_session_factory
    from app.models import ReviewEvent
    from app.pipeline.multipanel_prints import PAPER_GRID_REVIEW
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

    def bounds(quad):
        return {"x1": quad[0]["x"], "y1": quad[0]["y"], "x2": quad[2]["x"], "y2": quad[2]["y"]}

    outer = vertices(196, 21, 374, 234)
    parent = vertices(208, 31, 366, 233)
    fragments = [vertices(211, 111, 360, 226), vertices(211, 151, 360, 226)]
    neighbor = vertices(24, 32, 192, 228)
    detections = []
    for index, polygon in enumerate([parent, *fragments, neighbor]):
        mask = {"source": "classical_border_quad" if index == 0 else "yolo_seg_mask", "polygon": polygon}
        reasons = []
        if polygon != neighbor:
            mask["paper_grid_review"] = {"outline": outer}
            if polygon in fragments:
                mask["paper_grid_review"]["cut"] = polygon[:2]
            reasons.append(PAPER_GRID_REVIEW)
        detections.append(SegmentationDetection(bounds(polygon), mask, .81, 1.0, True, reasons))
    monkeypatch.setattr(pipeline, "segment_album_page", lambda *_, **__: SegmentationResult(detections, {}))

    image = np.full((260, 400, 3), 42, np.uint8)
    image[32:229, 24:193] = (62, 123, 151)
    image[21:235, 196:375] = (220, 225, 228)
    image[32:224, 207:364] = (67, 100, 143)
    image[41:214:4, 218:354] = (76, 111, 150)
    ok, png = cv2.imencode(".png", image)
    assert ok
    album_id = (await client.post("/api/v1/albums", json={"name": "A shared photo-paper grid"})).json()["id"]
    album_url = f"/api/v1/albums/{album_id}"
    uploaded = await client.post(f"{album_url}/pages/upload", files={"files": ("page.png", png.tobytes(), "image/png")})
    assert uploaded.status_code == 201, uploaded.text

    queue_url = f"/api/v1/review/queue?album_id={album_id}&type=segmentation"
    task, = (await client.get(queue_url)).json()
    before = task["payload"]
    assert before["detection_count"] == 4
    linked = [row for row in before["detections"] if row["segmentation_mask"].get("paper_grid_review")]
    parent_row, = [row for row in linked if "cut" not in row["segmentation_mask"]["paper_grid_review"]]
    assert len(parent_row["crop_modes"]["full_print"]["suggested_warp_quad"]) == 4
    parent_id = parent_row["photo_id"]
    fragment_ids = [row["photo_id"] for row in linked if row["photo_id"] != parent_id]
    neighbor_row, = [row for row in before["detections"] if row not in linked]
    parent_url = f"/api/v1/photos/{parent_id}"
    preview_url = f"{parent_url}/image?variant=source_corrected_untrimmed"
    preview = await client.get(preview_url)
    assert preview.status_code == 200
    variants = ("full_print", "image_only", "source_corrected_trimmed")
    for variant in variants:
        assert (await client.get(f"{parent_url}/image?variant={variant}")).status_code == 404

    event_url = "/api/v1/review/events"
    answer = {"task_type": "segmentation", "task_id": task["id"], "answer": "confirm_shared_paper_grid",
              "reason_code": "same_physical_print", "idempotency_key": uuid4().hex, "metadata": {
                  "photo_count_revision": before["photo_count_revision"], "paper_grid_parent_id": parent_id,
                  "paper_grid_fragment_ids": fragment_ids,
              }}
    reason_required = await client.post(event_url, json={**answer, "reason_code": None})
    assert reason_required.status_code == 422 and reason_required.json()["detail"] == "paper_grid_confirmation_required"
    for selection in ([fragment_ids[0]], [fragment_ids[0], neighbor_row["photo_id"]]):
        wrong = await client.post(event_url, json={**answer, "metadata": {**answer["metadata"],
                                  "paper_grid_fragment_ids": selection}})
        assert wrong.status_code == 409 and wrong.json()["detail"] == "paper_grid_review_group_stale"
    swapped = await client.post(event_url, json={**answer, "metadata": {**answer["metadata"],
                                "paper_grid_parent_id": fragment_ids[0],
                                "paper_grid_fragment_ids": [parent_id, fragment_ids[1]]}})
    assert swapped.status_code == 409 and swapped.json()["detail"] == "paper_grid_review_group_stale"
    assert (await client.get(queue_url)).json() == [task]

    count_answer = {"task_type": "segmentation", "task_id": task["id"], "answer": "correct",
                    "idempotency_key": uuid4().hex, "metadata": {"photo_count_revision": before["photo_count_revision"]}}
    assert (await client.post(event_url, json=count_answer)).status_code == 201
    assert (await client.get(queue_url)).json() == []
    competing = await client.post(event_url, json=answer)
    assert competing.status_code == 409 and competing.json()["detail"] == "paper_grid_review_group_stale"
    unsure = await client.post(event_url, json={**count_answer, "answer": "unsure", "idempotency_key": uuid4().hex})
    assert unsure.status_code == 201 and (await client.get(queue_url)).json() == [task]

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
    assert {row["photo_id"]: row for row in after["detections"]} == {
        parent_id: parent_row, neighbor_row["photo_id"]: neighbor_row,
    }
    assert (await client.get(preview_url)).content == preview.content
    for variant in variants:
        assert (await client.get(f"{parent_url}/image?variant={variant}")).status_code == 404
    for fragment_id in fragment_ids:
        fragment = (await client.get(f"/api/v1/photos/{fragment_id}")).json()
        assert fragment["reviewed_non_print"] is True
        rejected = fragment["enhancement_applied"]["review_rejected_detection"]
        assert rejected["segmentation_mask"] == fragment["segmentation_mask"]
    async with get_session_factory()() as session:
        event = await session.get(ReviewEvent, UUID(event_id))
        assert event.event_metadata["client"] == answer["metadata"]
        assert len(event.event_metadata["snapshot"]["photos"]) == 4

    approve = await client.post(event_url, json={"task_type": "geometry", "task_id": parent_id,
                                "answer": "correct", "idempotency_key": uuid4().hex})
    assert approve.status_code == 409
    crop = await client.patch(f"{parent_url}/bounding-box", json={
        "bounding_box": bounds(outer), "confirm_print_outline": True,
        "segmentation_mask": {"source": "manual_quad", "polygon": outer, "warp_quad": outer,
                              "print_size_mm": {"width": 178, "height": 213},
                              "image_quad": vertices(207, 32, 363, 223)},
    })
    assert crop.status_code == 200, crop.text
    for variant in variants:
        exported = await client.get(f"{parent_url}/image?variant={variant}")
        assert exported.status_code == 200, (variant, crop.json()["enhancement_applied"]["correction"]["crop_modes"])

    restored_quad = next(row["polygon"] for row in linked if row["photo_id"] == fragment_ids[0])
    restored = await client.patch(f"/api/v1/photos/{fragment_ids[0]}/bounding-box", json={
        "bounding_box": bounds(restored_quad),
        "segmentation_mask": {"source": "manual_quad", "polygon": restored_quad, "warp_quad": restored_quad},
    })
    assert restored.status_code == 200 and restored.json()["reviewed_non_print"] is False
    reopened, = (await client.get(queue_url)).json()
    assert reopened["payload"]["detection_count"] == 3
    assert reopened["payload"]["photo_count_revision"] != event_id
    for variant in variants:
        assert (await client.get(f"{parent_url}/image?variant={variant}")).status_code == 404


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
