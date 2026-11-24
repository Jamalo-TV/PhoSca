from dataclasses import replace
from types import SimpleNamespace
from uuid import uuid4

import cv2
import numpy as np
import pytest

from app.config import Settings
from app.pipeline import segmentation
from app.pipeline.edge_prints import CAPTURE_EDGE_HOLD, CAPTURE_EDGE_SOURCE, SCANNER_BACKING_REVIEW
from app.pipeline.variants import variant_path_from_photo


def scan_with_clipped_prints(settings, turns=0, *, blank=False):
    width, height = 940, 640
    page = np.full((height, width, 3), (22, 27, 31), np.uint8)
    rng = np.random.default_rng(427)
    outlines = []
    guesses = []
    for left, top, bottom in ((392, 45, 268), (449, 322, 585)):
        page[top:bottom + 1, left:] = (221, 231, 235)
        if not blank:
            texture = rng.integers(95, 243, size=(34, 41, 3), dtype=np.uint8)
            page[top + 8:bottom - 7, left + 8:] = cv2.resize(
                texture, (width - left - 8, bottom - top - 15), interpolation=cv2.INTER_CUBIC,
            )
        outlines.append(np.float32([[left, top], [939, top], [939, bottom], [left, bottom]]))
        guesses.append(np.float32([[left + 25, top + 12], [939, top + 14],
                                   [939, bottom - 12], [left + 54, bottom - 16]]))
    for _ in range(turns):
        rotation = np.float32([[0, 1, 0], [-1, 0, width - 1]])
        outlines = [cv2.transform(outline[None], rotation)[0] for outline in outlines]
        guesses = [cv2.transform(guess[None], rotation)[0] for guess in guesses]
        width, height = height, width
    page = np.ascontiguousarray(np.rot90(page, turns))
    detections = [
        segmentation._make_detection(
            box={}, points=guess, confidence=.94, image_width=width, image_height=height, settings=settings,
            source="yolo_seg_mask", detector_source="yolo_onnx",
        )
        for guess in guesses
    ]
    return page, outlines, segmentation.SegmentationResult(detections, {})


@pytest.mark.parametrize("turns", [0, 1, 2, 3])
def test_recovers_visible_edges_in_source_coordinates(monkeypatch, turns):
    settings = Settings(DATABASE_URL="sqlite://", REDIS_URL="redis://")
    page, expected, original = scan_with_clipped_prints(settings, turns)
    monkeypatch.setattr(segmentation, "_segment_album_page", lambda *_: original)
    prepared = cv2.convertScaleAbs(page, alpha=.70, beta=87)

    result = segmentation.segment_album_page(prepared, settings, source_image=page, protected_regions=[])

    assert result.metadata["capture_edge_outlines_recovered"] == 2
    for detection, truth in zip(result.detections, expected, strict=True):
        height, width = page.shape[:2]
        outline = segmentation.normalized_vertices_to_pixels(
            segmentation._detection_polygon(detection), width, height,
        )
        assert min(np.max(np.linalg.norm(outline - np.roll(truth, i, axis=0), axis=1)) for i in range(4)) < 1
        assert detection.mask["detector_source"] == CAPTURE_EDGE_SOURCE
        assert detection.mask["capture_edge_clipped"] is True
        assert CAPTURE_EDGE_HOLD in detection.review_reasons


@pytest.mark.parametrize("reason", ["blank", "dark_last_column", "unseeded"])
def test_capture_edge_recovery_abstains_without_source_or_detector_support(reason):
    settings = Settings(DATABASE_URL="sqlite://", REDIS_URL="redis://")
    page, _, original = scan_with_clipped_prints(settings, blank=reason == "blank")
    if reason == "dark_last_column":
        # Scale the source so the final native pixel can vanish in discovery.
        page = cv2.resize(page, None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST)
        page[:, -1] = (22, 27, 31)
    if reason == "unseeded":
        original = segmentation.SegmentationResult(original.detections[:1], {})

    assert segmentation._recover_capture_edge_prints(page, original, settings) is original


def scan_with_backing_rails(settings, case):
    height, width = 1320, 960
    page = np.full((height, width, 3), 27, np.uint8)
    page[45:1144, 275:] = 43
    page[45:1144, 332:] = 51
    if case != "single_rail":
        page[925:1144, 275:] -= 10
    backing = np.float32([[275, 40], [739, 393], [453, 1144], [275, 1144]])
    if case == "dark_edge":
        print_mask = np.zeros((height, width), np.uint8)
        cv2.fillConvexPoly(print_mask, backing.astype(np.int32), 1)
        page[print_mask != 0] += 3
    rng = np.random.default_rng(223)
    if case == "dark_detail":
        texture = rng.integers(29, 55, (24, 26, 3), dtype=np.uint8)
        page[520:707, 350:495] = cv2.resize(texture, (145, 187), interpolation=cv2.INTER_CUBIC)
    if case == "dark_tone":
        patch = np.zeros(page.shape[:2], np.float32)
        patch[545:640, 394:490] = 10
        page -= np.rint(cv2.GaussianBlur(patch, (0, 0), 4)).astype(np.uint8)[..., None]
    detections = []
    for left, top, bottom in ((454, 0, 392), (587, 404, 941), (454, 949, 1319)):
        page[top:bottom + 1, left:] = 222
        texture = rng.integers(99, 233, (31, 37, 3), dtype=np.uint8)
        page[top + 8:bottom - 7, left + 8:] = cv2.resize(texture, (width - left - 8, bottom - top - 15))
        quad = np.float32([[left + 15, top + 12], [width - 1, top + 10],
                           [width - 1, bottom - 12], [left + 15, bottom - 8]])
        detections.append(segmentation._make_detection(
            box={}, points=quad, confidence=.9, image_width=width, image_height=height,
            settings=settings, source="yolo_seg_mask", detector_source="yolo_onnx",
        ))
    extra = segmentation._make_detection(
        box={}, points=backing, confidence=.77, image_width=width, image_height=height,
        settings=settings, source="classical_border_quad", detector_source="edge_contour",
    )
    if case == "model_support":
        extra = replace(extra, mask={**extra.mask, "fusion": {"proposal_families": ["classical", "model"]}})
    return page, segmentation.SegmentationResult([*detections, extra], {})


@pytest.mark.parametrize("case", ["scanner", "single_rail", "dark_edge", "dark_detail", "dark_tone", "model_support"])
def test_scanner_backing_requires_native_evidence_without_an_independent_photo(case):
    settings = Settings(DATABASE_URL="sqlite://", REDIS_URL="redis://")
    page, original = scan_with_backing_rails(settings, case)
    result = segmentation._recover_capture_edge_prints(page, original, settings)

    expected_rejections = int(case == "scanner")
    assert result.metadata["capture_edge_outlines_recovered"] == 3
    assert result.metadata.get("scanner_backing_outlines_rejected", 0) == expected_rejections
    assert sum(item.mask.get("capture_edge_clipped") is True for item in result.detections) == 3
    if expected_rejections:
        assert len(result.detections) == 3
    else:
        assert result.detections[-1].mask == original.detections[-1].mask
        assert (SCANNER_BACKING_REVIEW in result.detections[-1].review_reasons) == (case in {"dark_detail", "dark_tone"})


def test_capture_edge_hold_applies_to_stale_download_metadata():
    modes = ("full_print", "image_only", "source_corrected_trimmed")
    photo = SimpleNamespace(
        segmentation_mask={"capture_edge_clipped": True},
        enhancement_applied={
            "correction": {"crop_modes": {mode: {"status": "ready"} for mode in modes[:2]}},
            "variants": {mode: {"path": "/previous-crop.jpg"}
                         for mode in (*modes, "source_corrected_untrimmed")},
        },
    )

    assert all(variant_path_from_photo(photo, mode) is None for mode in modes)
    assert variant_path_from_photo(photo, "source_corrected_untrimmed") == "/previous-crop.jpg"


async def test_dark_content_on_scanner_backing_is_kept_for_review(client, monkeypatch):
    from app.config import get_settings
    from app.services import pipeline

    monkeypatch.setenv("AUTO_ANALYZE_UPLOADS", "true")
    monkeypatch.setenv("MAX_UPLOAD_SIZE", "1048576")
    monkeypatch.setenv("MAX_REQUEST_SIZE", "2097152")
    monkeypatch.setenv("ENABLE_PHOTO_ORIENTATION_CORRECTION", "false")
    monkeypatch.setenv("ENABLE_PROTECTED_REGION_GUARD", "false")
    get_settings.cache_clear()
    page, original = scan_with_backing_rails(get_settings(), "dark_tone")
    monkeypatch.setattr(segmentation, "_segment_album_page", lambda *_: original)
    monkeypatch.setattr(pipeline, "run_paddle_ocr", lambda *_: [])
    album = await client.post("/api/v1/albums", json={"name": "Dunkler Abzug im Scanner"})
    assert album.status_code == 201, album.text
    url = f"/api/v1/albums/{album.json()['id']}"
    ok, image = cv2.imencode(".jpg", page, [cv2.IMWRITE_JPEG_QUALITY, 97])
    assert ok
    upload = await client.post(f"{url}/pages/upload", files=[("files", ("scan.jpg", image.tobytes(), "image/jpeg"))])
    assert upload.status_code == 201, upload.text
    photos = (await client.get(f"{url}/photos")).json()
    assert len(photos) == 4
    candidate, = [photo for photo in photos if photo["segmentation_mask"]["detector_source"] == "edge_contour"]
    metadata = candidate["enhancement_applied"]
    assert SCANNER_BACKING_REVIEW in metadata["segmentation_review_reasons"]
    for mode in ("full_print", "image_only"):
        assert metadata["correction"]["crop_modes"][mode]["status"] == "review_needed"
        assert candidate["crop_download_urls"][mode] is None
    preview = await client.get(f"/api/v1/photos/{candidate['id']}/image?variant=source_corrected_untrimmed")
    assert preview.status_code == 200


async def test_capture_edge_hold_survives_manual_corners_and_review(client, monkeypatch):
    from app.config import get_settings
    from app.services import pipeline

    monkeypatch.setenv("AUTO_ANALYZE_UPLOADS", "true")
    monkeypatch.setenv("MAX_UPLOAD_SIZE", "1048576")
    monkeypatch.setenv("MAX_REQUEST_SIZE", "2097152")
    monkeypatch.setenv("ENABLE_PHOTO_ORIENTATION_CORRECTION", "false")
    monkeypatch.setenv("ENABLE_PROTECTED_REGION_GUARD", "false")
    get_settings.cache_clear()
    page, _, original = scan_with_clipped_prints(get_settings())
    monkeypatch.setattr(segmentation, "_segment_album_page", lambda *_: original)
    monkeypatch.setattr(pipeline, "run_paddle_ocr", lambda *_: [])
    album = await client.post("/api/v1/albums", json={"name": "Angeschnittener Abzug"})
    assert album.status_code == 201, album.text
    albums = f"/api/v1/albums/{album.json()['id']}"
    ok, image = cv2.imencode(".jpg", page, [cv2.IMWRITE_JPEG_QUALITY, 97])
    assert ok
    upload = await client.post(f"{albums}/pages/upload",
                               files=[("files", ("scan.jpg", image.tobytes(), "image/jpeg"))])
    assert upload.status_code == 201, upload.text
    photo, other = (await client.get(f"{albums}/photos")).json()
    assert photo["segmentation_mask"]["capture_edge_clipped"] is True
    assert other["segmentation_mask"]["capture_edge_clipped"] is True
    url = f"/api/v1/photos/{photo['id']}"
    # Moving the corners inside the capture cannot restore the missing part.
    outer = [{"x": x / 939, "y": y / 639} for x, y in [[399, 52], [927, 52], [927, 262], [399, 262]]]
    inner = [{"x": x / 939, "y": y / 639} for x, y in [[406, 60], [921, 60], [921, 254], [406, 254]]]
    response = await client.patch(f"{url}/bounding-box", json={
        "bounding_box": photo["bounding_box"],
        "segmentation_mask": {"source": "manual_quad", "polygon": outer, "warp_quad": outer, "image_quad": inner,
                              "print_size_mm": {"width": 150., "height": 100.}},
    })
    assert response.status_code == 200, response.text
    revised = response.json()
    assert revised["status"] == "review_needed"
    assert revised["segmentation_mask"]["capture_edge_clipped"] is True
    modes = revised["enhancement_applied"]["correction"]["crop_modes"]
    for mode in ("full_print", "image_only"):
        assert CAPTURE_EDGE_HOLD in modes[mode]["reason_codes"]
        assert revised["crop_download_urls"][mode] is None
        assert (await client.get(f"{url}/image?variant={mode}")).status_code == 404
    assert (await client.get(f"{url}/image?variant=source_corrected_untrimmed")).status_code == 200
    approval = await client.post("/api/v1/review/events", json={
        "task_type": "geometry", "task_id": photo["id"], "answer": "correct", "idempotency_key": str(uuid4()),
    })
    assert approval.status_code == 409, approval.text
