from types import SimpleNamespace
from uuid import uuid4

import cv2
import numpy as np
import pytest

from app.config import Settings
from app.pipeline import segmentation
from app.pipeline.edge_prints import CAPTURE_EDGE_HOLD, CAPTURE_EDGE_SOURCE
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
