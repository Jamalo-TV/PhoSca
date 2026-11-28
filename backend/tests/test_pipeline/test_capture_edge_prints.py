from dataclasses import replace
from types import SimpleNamespace
from uuid import uuid4

import cv2
import numpy as np
import pytest

from app.config import Settings
from app.pipeline import segmentation
from app.pipeline.edge_prints import (
    CAPTURE_EDGE_HOLD, CAPTURE_EDGE_SOURCE, CAPTURE_SPLIT_REVIEW_SOURCE, CLIPPED_PRINT_OVERLAP_REVIEW,
    SCANNER_BACKING_REVIEW,
)
from app.pipeline.perspective import correct_photo_crop
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

    revision = item["payload"]["geometry_review_revision"]
    body = {"task_type": "geometry", "task_id": candidate["id"], "idempotency_key": str(uuid4()),
            "metadata": {"geometry_review_revision": revision}}
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
        modes = corrected.json()["enhancement_applied"]["correction"]["crop_modes"]
        for mode, url in corrected.json()["crop_download_urls"].items():
            assert url is None
            assert "MANUAL_NEIGHBOR_OVERLAP" in modes[mode]["reason_codes"]
            assert (await client.get(f"{photo_url}/image?variant={mode}")).status_code == 404
        preview = await client.get(f"{photo_url}/image?variant=source_corrected_untrimmed")
        assert preview.status_code == 200
        rendered = cv2.imdecode(np.frombuffer(preview.content, np.uint8), cv2.IMREAD_COLOR)
        assert abs(rendered.shape[1] / rendered.shape[0] - 150 / 95) < .02
        return
    rejected = await client.post(events, json={**body, "answer": "wrong_object"})
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
    if case in {"dark_tone", "small_dark_corner"}:
        patch = np.zeros(page.shape[:2], np.float32)
        if case == "small_dark_corner":
            patch[1002:1048, 382:436] = 4
            blur = 1.2
        else:
            patch[545:640, 394:490] = 10
            blur = 4
        page -= np.rint(cv2.GaussianBlur(patch, (0, 0), blur)).astype(np.uint8)[..., None]
    if case in {"dark_chroma", "tinted_scanner"}:
        area = np.s_[514:630, 378:480] if case == "dark_chroma" else np.s_[:, :]
        patch = cv2.cvtColor(page[area], cv2.COLOR_BGR2LAB)
        patch[..., 1] += 9 if case == "dark_chroma" else 6
        patch[..., 2] -= 8 if case == "dark_chroma" else 4
        page[area] = cv2.cvtColor(patch, cv2.COLOR_LAB2BGR)
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


@pytest.mark.parametrize("case", [
    "scanner", "single_rail", "dark_edge", "dark_detail", "dark_tone", "small_dark_corner",
    "dark_chroma", "tinted_scanner", "model_support",
])
def test_scanner_backing_requires_native_evidence_without_an_independent_photo(case):
    settings = Settings(DATABASE_URL="sqlite://", REDIS_URL="redis://")
    page, original = scan_with_backing_rails(settings, case)
    result = segmentation._recover_capture_edge_prints(page, original, settings)

    expected_rejections = int(case in {"scanner", "tinted_scanner"})
    assert result.metadata["capture_edge_outlines_recovered"] == 3
    assert result.metadata.get("scanner_backing_outlines_rejected", 0) == expected_rejections
    assert sum(item.mask.get("capture_edge_clipped") is True for item in result.detections) == 3
    if expected_rejections:
        assert len(result.detections) == 3
    else:
        assert result.detections[-1].mask == original.detections[-1].mask
        needs_review = case in {"dark_detail", "dark_tone", "small_dark_corner", "dark_chroma"}
        assert (SCANNER_BACKING_REVIEW in result.detections[-1].review_reasons) == needs_review


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


@pytest.mark.parametrize("gap", [1, 8])
def test_capture_recovery_respects_native_gutter_from_paper_to_frame(gap):
    settings = Settings(DATABASE_URL="sqlite://", REDIS_URL="redis://")
    page, _ = scan_with_backing_rails(settings, "scanner")
    page[949:1230, 705:705 + gap] = page[945, 705]

    def lower_outlines():
        return [quad for quad in segmentation.clipped_print_outlines(page) if quad[:, 1].mean() > 1000]

    # A mark stopping before the frame leaves the original paper extent intact.
    full, = lower_outlines()
    assert full[:, 0].min() < 455
    page[1230:, 705:705 + gap] = page[945, 705]

    # A throughgoing gap forbids the merged extent. The wider gap also
    # resolves the right sheet individually in contour discovery.
    measured = lower_outlines()
    if gap == 8:
        right, = measured
        assert right[:, 0].min() > 705
    else:
        assert measured == []
    assert len(segmentation.clipped_print_outlines(page)) == 2 + len(measured)


def test_capture_corner_with_orthogonal_claimants_shows_two_traces_without_changing_geometry():
    settings = Settings(DATABASE_URL="sqlite://", REDIS_URL="redis://")
    page, original = scan_with_backing_rails(settings, "scanner")
    height, width = page.shape[:2]
    candidates = [
        ([[707, 956], [959, 956], [959, 1319], [707, 1319]], "yolo_seg_mask", "yolo_onnx"),
        ([[459, 956], [706, 956], [706, 1319], [459, 1319]], "yolo_seg_mask", "yolo_onnx"),
        ([[502, 996], [959, 1000], [959, 1210], [581, 1220]], "classical_border_quad", "edge_contour"),
    ]
    parts = [segmentation._make_detection(
        box={}, points=np.float32(points), confidence=.91, image_width=width, image_height=height,
        settings=settings, source=source, detector_source=detector_source,
    ) for points, source, detector_source in candidates]
    baseline = segmentation.SegmentationResult([*original.detections[:2], *parts], {})

    result = segmentation._recover_capture_edge_prints(page, baseline, settings)

    assert result.metadata["capture_edge_outlines_recovered"] == 2
    assert result.metadata["capture_edge_split_reviews"] == 1
    traces = result.detections[2].mask["capture_split_review_traces"]
    assert len(traces) == 2
    assert all(len(line) == 2 and not any(line[0][axis] == line[1][axis] in (0., 1.)
                                         for axis in ("x", "y")) for line in traces)
    for prior, item in zip(parts, result.detections[2:], strict=True):
        assert item.bounding_box == prior.bounding_box
        assert item.mask == {**prior.mask, "capture_split_review_traces": traces}
        assert item.review_reasons == [CAPTURE_EDGE_HOLD]
        params = dict(dual_mode=True, auto_orient=False, protect_regions=False)
        previous_crop = correct_photo_crop(page, prior.bounding_box, prior.mask, **params)
        crop = correct_photo_crop(page, item.bounding_box, item.mask, **params)
        np.testing.assert_array_equal(crop.rectified_untrimmed, previous_crop.rectified_untrimmed)
        modes = crop.metadata["crop_modes"]
        assert all(mode["status"] == "review_needed" and CAPTURE_EDGE_HOLD in mode["reason_codes"]
                   for mode in modes.values())
        assert modes["full_print"]["suggested_edge_source"] == CAPTURE_SPLIT_REVIEW_SOURCE
        assert modes["full_print"]["suggested_edge_traces"] == traces
        assert "suggested_warp_quad" not in modes["full_print"] and "suggested_edge_traces" not in modes["image_only"]


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
    assert partial.mask == original.detections[2].mask
    assert scanner.mask == original.detections[3].mask
    mask = measured.mask.copy()
    traces = mask.pop("capture_edge_review_traces", None)
    assert mask == full.mask
    if not fragment_review:
        assert traces is None
        return

    assert len(traces) == 2
    assert all(len(line) == 2 and not any(line[0][axis] == line[1][axis] in (0., 1.)
                                         for axis in ("x", "y")) for line in traces)
    params = dict(dual_mode=True, auto_orient=False, protect_regions=False)
    baseline = correct_photo_crop(page, full.bounding_box, full.mask, **params)
    crop = correct_photo_crop(page, measured.bounding_box, measured.mask, **params)
    np.testing.assert_array_equal(crop.rectified_untrimmed, baseline.rectified_untrimmed)
    modes = crop.metadata["crop_modes"]
    assert all(mode["status"] == "review_needed" and CAPTURE_EDGE_HOLD in mode["reason_codes"] for mode in modes.values())
    whole = modes["full_print"]
    assert whole["warp_quad"] == baseline.metadata["crop_modes"]["full_print"]["warp_quad"]
    assert whole["suggested_edge_source"] == CAPTURE_EDGE_SOURCE
    assert whole["suggested_edge_traces"] == traces
    assert "suggested_warp_quad" not in whole and "suggested_edge_traces" not in modes["image_only"]


@pytest.mark.parametrize("mask", [
    {"capture_edge_clipped": True},
    {"capture_split_review_traces": [[{"x": 0., "y": .6}, {"x": .5, "y": .6}],
                                     [{"x": .5, "y": .6}, {"x": .5, "y": 1.}]]},
])
def test_capture_edge_hold_applies_to_stale_download_metadata(mask):
    modes = ("full_print", "image_only", "source_corrected_trimmed")
    photo = SimpleNamespace(
        segmentation_mask=mask,
        enhancement_applied={
            "correction": {"crop_modes": {mode: {"status": "ready"} for mode in modes[:2]}},
            "variants": {mode: {"path": "/previous-crop.jpg"}
                         for mode in (*modes, "source_corrected_untrimmed")},
        },
    )

    assert all(variant_path_from_photo(photo, mode) is None for mode in modes)
    assert variant_path_from_photo(photo, "source_corrected_untrimmed") == "/previous-crop.jpg"


@pytest.mark.parametrize("case", ["dark_tone", "small_dark_corner", "dark_chroma"])
async def test_dark_content_on_scanner_backing_is_kept_for_review(client, monkeypatch, case):
    from app.config import get_settings
    from app.services import pipeline

    monkeypatch.setenv("AUTO_ANALYZE_UPLOADS", "true")
    monkeypatch.setenv("MAX_UPLOAD_SIZE", "1048576")
    monkeypatch.setenv("MAX_REQUEST_SIZE", "2097152")
    monkeypatch.setenv("ENABLE_PHOTO_ORIENTATION_CORRECTION", "false")
    monkeypatch.setenv("ENABLE_PROTECTED_REGION_GUARD", "false")
    get_settings.cache_clear()
    page, original = scan_with_backing_rails(get_settings(), case)
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


async def test_capture_edge_hold_survives_manual_corners_and_review(submit_geometry_review, client, monkeypatch):
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
    approval = await submit_geometry_review("/api/v1/review/events", json={
        "task_type": "geometry", "task_id": photo["id"], "answer": "correct", "idempotency_key": str(uuid4()),
    })
    assert approval.status_code == 409, approval.text
