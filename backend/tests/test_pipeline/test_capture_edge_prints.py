from dataclasses import replace
from types import SimpleNamespace
from uuid import uuid4

import cv2
import numpy as np
import pytest

from app.config import Settings
from app.pipeline import segmentation
from app.pipeline.edge_prints import (
    CAPTURE_EDGE_HOLD, CAPTURE_EDGE_SOURCE, CLIPPED_PRINT_OVERLAP_REVIEW, SCANNER_BACKING_REVIEW,
)
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


def scan_with_nested_candidate(settings, case):
    page, _, original = scan_with_clipped_prints(settings)
    height, width = page.shape[:2]
    quad = np.float32([[392, 85], [584, 98], [574, 215], [392, 205]])
    if case == "separate_paper":
        cv2.fillConvexPoly(page, quad.astype(np.int32), (98, 105, 114))
    elif case == "low_contrast_paper":
        visible = np.zeros((height, width), np.uint8)
        cv2.fillConvexPoly(visible, quad.astype(np.int32), 1)
        page[visible != 0] -= 2
    elif case == "fully_inside":
        quad[[0, 3], 0] += 20
    extra = segmentation._make_detection(
        box={}, points=quad, confidence=.86, image_width=width, image_height=height,
        settings=settings, source="classical_border_quad", detector_source="edge_contour",
    )
    if case == "model_support":
        extra = replace(extra, mask={**extra.mask, "fusion": {"proposal_families": ["classical", "model"]}})
    return page, segmentation.SegmentationResult([*original.detections, extra], {})


@pytest.mark.parametrize("case", ["fragment", "separate_paper", "low_contrast_paper", "fully_inside", "model_support"])
def test_capture_edge_overlap_holds_unsupported_shared_contour(case):
    settings = Settings(DATABASE_URL="sqlite://", REDIS_URL="redis://")
    page, original = scan_with_nested_candidate(settings, case)
    result = segmentation._recover_capture_edge_prints(page, original, settings)

    ambiguous = case in {"fragment", "low_contrast_paper"}
    assert result.metadata["capture_edge_outlines_recovered"] == 2
    assert result.metadata.get("capture_edge_overlap_reviews", 0) == int(ambiguous)
    assert len(result.detections) == 3
    assert result.detections[-1].mask == original.detections[-1].mask
    assert (CLIPPED_PRINT_OVERLAP_REVIEW in result.detections[-1].review_reasons) == ambiguous


@pytest.mark.parametrize("review_action", ["wrong_object", "manual_correction"])
async def test_nested_capture_candidate_reaches_review_without_a_download(client, monkeypatch, review_action):
    from app.config import get_settings
    from app.services import pipeline

    monkeypatch.setenv("AUTO_ANALYZE_UPLOADS", "true")
    monkeypatch.setenv("MAX_UPLOAD_SIZE", "1048576")
    monkeypatch.setenv("MAX_REQUEST_SIZE", "2097152")
    monkeypatch.setenv("ENABLE_PHOTO_ORIENTATION_CORRECTION", "false")
    monkeypatch.setenv("ENABLE_PROTECTED_REGION_GUARD", "false")
    get_settings.cache_clear()
    case = "fragment" if review_action == "wrong_object" else "low_contrast_paper"
    page, original = scan_with_nested_candidate(get_settings(), case)
    monkeypatch.setattr(segmentation, "_segment_album_page", lambda *_: original)
    monkeypatch.setattr(pipeline, "run_paddle_ocr", lambda *_: [])
    album = await client.post("/api/v1/albums", json={"name": "Unklares Foto im Abzug"})
    assert album.status_code == 201, album.text
    album_id = album.json()["id"]
    base = f"/api/v1/albums/{album_id}"
    ok, image = cv2.imencode(".jpg", page, [cv2.IMWRITE_JPEG_QUALITY, 97])
    assert ok
    upload = await client.post(f"{base}/pages/upload", files=[("files", ("scan.jpg", image.tobytes(), "image/jpeg"))])
    assert upload.status_code == 201, upload.text
    photos = (await client.get(f"{base}/photos")).json()
    candidate, = [photo for photo in photos if not photo["segmentation_mask"].get("capture_edge_clipped")]
    assert candidate["status"] == "review_needed"
    assert CLIPPED_PRINT_OVERLAP_REVIEW in candidate["enhancement_applied"]["segmentation_review_reasons"]
    queue = f"/api/v1/review/queue?album_id={album_id}"
    items = (await client.get(f"{queue}&type=geometry")).json()
    item, = [item for item in items if item["id"] == candidate["id"]]
    assert CLIPPED_PRINT_OVERLAP_REVIEW in item["payload"]["review_reasons"]
    for mode, download_url in candidate["crop_download_urls"].items():
        assert download_url is None
        assert (await client.get(f"/api/v1/photos/{candidate['id']}/image?variant={mode}")).status_code == 404
    assert (await client.get(item["payload"]["untrimmed_url"])).status_code == 200

    body = {"task_type": "geometry", "task_id": candidate["id"], "idempotency_key": str(uuid4())}
    events = "/api/v1/review/events"
    assert (await client.post(events, json={**body, "answer": "correct"})).status_code == 409
    if review_action == "manual_correction":
        outer = candidate["segmentation_mask"]["polygon"]
        photo_url = f"/api/v1/photos/{candidate['id']}"
        corrected = await client.patch(f"{photo_url}/bounding-box", json={
            "bounding_box": candidate["bounding_box"],
            "segmentation_mask": {"source": "manual_quad", "polygon": outer, "warp_quad": outer,
                                  "image_quad": outer, "print_size_mm": {"width": 150., "height": 95.}},
        })
        assert corrected.status_code == 200, corrected.text
        for mode, url in corrected.json()["crop_download_urls"].items():
            assert url, corrected.json()["enhancement_applied"]["correction"]["crop_modes"][mode]
            download = await client.get(url)
            assert download.status_code == 200
            rendered = cv2.imdecode(np.frombuffer(download.content, np.uint8), cv2.IMREAD_COLOR)
            assert abs(rendered.shape[1] / rendered.shape[0] - 150 / 95) < .02
        return
    rejected = await client.post(events, json={**body, "answer": "wrong_object", "metadata": {"ui": "desktop"}})
    assert rejected.status_code == 201, rejected.text
    page_item, = (await client.get(f"{queue}&type=segmentation")).json()
    assert page_item["payload"]["detection_count"] == 2
    assert {entry["photo_id"] for entry in page_item["payload"]["detections"]} == {
        photo["id"] for photo in photos if photo["id"] != candidate["id"]
    }


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


@pytest.mark.parametrize("ambiguity", ["multiple_proposals", "multiple_outlines"])
def test_ambiguous_match_keeps_its_proposals_but_allows_separate_recovery(ambiguity, monkeypatch):
    settings = Settings(DATABASE_URL="sqlite://", REDIS_URL="redis://")
    page, original = scan_with_backing_rails(settings, "scanner")
    if ambiguity == "multiple_proposals":
        lower = original.detections[2]
        original.detections.append(replace(lower, confidence=.82))
    else:
        outlines = segmentation.clipped_print_outlines(page)
        assert len(outlines) == 3
        lower = max(outlines, key=lambda outline: outline[:, 1].mean())
        monkeypatch.setattr(segmentation, "clipped_print_outlines", lambda _: [*outlines, lower.copy()])

    result = segmentation._recover_capture_edge_prints(page, original, settings)

    assert result.metadata["capture_edge_outlines_recovered"] == 2
    assert "scanner_backing_outlines_rejected" not in result.metadata
    assert all(item.mask.get("capture_edge_clipped") and CAPTURE_EDGE_HOLD in item.review_reasons
               for item in result.detections[:2])
    assert len(result.detections) == len(original.detections)
    assert all(item is prior for item, prior in zip(result.detections[2:], original.detections[2:], strict=True))


@pytest.mark.parametrize("case,separate_fragment,scanner_review,fragment_review", [
    ("scanner", False, True, True),
    ("scanner", True, True, False),
    ("single_rail", False, False, True),
    ("model_support", False, False, True),
])
def test_supported_capture_conflict_gives_review_hints_without_removing_candidates(
    case, separate_fragment, scanner_review, fragment_review,
):
    settings = Settings(DATABASE_URL="sqlite://", REDIS_URL="redis://")
    page, original = scan_with_backing_rails(settings, case)
    height, width = page.shape[:2]
    complete = max(segmentation.clipped_print_outlines(page), key=lambda outline: outline[:, 1].mean())
    full = segmentation._make_detection(
        box={}, points=complete, confidence=.91, image_width=width, image_height=height, settings=settings,
        source="yolo_seg_mask", detector_source="yolo_onnx",
    )
    full = replace(full, mask={**full.mask, "fusion": {"proposal_families": ["model"]}})
    fragment = np.float32([[533, 1010], [width - 1, 1001], [width - 1, 1240], [493, 1248]])
    if separate_fragment:
        cv2.fillConvexPoly(page, np.rint(fragment).astype(np.int32), (164, 168, 171))
    original.detections[2] = segmentation._make_detection(
        box={}, points=fragment, confidence=.82, image_width=width, image_height=height, settings=settings,
        source="classical_border_quad", detector_source="edge_contour",
    )
    original.detections.append(full)

    result = segmentation._recover_capture_edge_prints(page, original, settings)

    assert result.metadata["capture_edge_outlines_recovered"] == 2
    assert "scanner_backing_outlines_rejected" not in result.metadata
    assert len(result.detections) == 5
    partial, scanner, measured = result.detections[2:]
    assert (CLIPPED_PRINT_OVERLAP_REVIEW in partial.review_reasons) == fragment_review
    assert (SCANNER_BACKING_REVIEW in scanner.review_reasons) == scanner_review
    assert [item.mask for item in result.detections[2:]] == [item.mask for item in original.detections[2:]]
    assert measured is full


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
