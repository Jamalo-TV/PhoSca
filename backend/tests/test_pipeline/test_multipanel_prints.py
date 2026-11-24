from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pytest

from app.config import Settings
from app.pipeline import segmentation
from app.pipeline.crop_validation import automatic_crop_review_reasons
from app.pipeline.multipanel_prints import MULTIPANEL_PRINT_REVIEW, RECOVERED_STOCK_SOURCE


def page_and_proposals(settings, gap, angle, *, mixed=False, jpeg_quality=91):
    page = np.full((640, 940, 3), (77, 84, 88), np.uint8)
    page[94:554, 108:832] = (229, 232, 235)
    rng = np.random.default_rng(551)
    panels = []
    for left in ((130, 365, 605) if mixed else (130, 506)):
        motif = rng.integers(35, 184, size=(16, 12, 3), dtype=np.uint8)
        right = left + (191 if mixed else 303)
        page[118:526, left:right] = cv2.resize(motif, (right - left, 408), interpolation=cv2.INTER_CUBIC)
        panels.append(np.float32([[left, 118], [right - 1, 118], [right - 1, 525], [left, 525]]))
    if gap:
        gutter = 575 if mixed else 474
        page[94:554, gutter:gutter + gap] = (77, 84, 88)
    parent = np.float32([[108, 94], [831, 94], [831, 553], [108, 553]])
    if angle:
        matrix = cv2.getRotationMatrix2D((470, 320), angle, 1)
        page = cv2.warpAffine(page, matrix, (940, 640), borderValue=(77, 84, 88))
        parent, *panels = cv2.transform(np.array([parent, *panels]), matrix)
    if jpeg_quality is not None:
        ok, jpeg = cv2.imencode(".jpg", page, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
        assert ok
        page = cv2.imdecode(jpeg, cv2.IMREAD_COLOR)
    proposals = [
        segmentation._make_detection(
            box={}, points=quad, confidence=.94, image_width=940, image_height=640, settings=settings,
            source="yolo_seg_mask", detector_source="yolo_onnx",
        )
        for quad in (parent, *panels)
    ]
    return page, parent, proposals


@pytest.mark.parametrize(("gap", "angle"), [(0, 0), (0, 8), (2, 0), (7, 8), (24, 0)])
def test_source_distinguishes_shared_stock_from_neighboring_prints(monkeypatch, gap, angle):
    settings = Settings(DATABASE_URL="sqlite://", REDIS_URL="redis://")
    original, parent, proposals = page_and_proposals(settings, gap, angle)
    # Replay the same outer and inner proposals through the normal source-evidence path.
    monkeypatch.setattr(segmentation, "_segment_album_page",
                        lambda *_: segmentation.SegmentationResult(proposals, {}))
    prepared = cv2.convertScaleAbs(original, alpha=1.1, beta=4)

    result = segmentation.segment_album_page(prepared, settings, source_image=original, protected_regions=[])

    if gap:
        assert len(result.detections) == 2
        assert all(MULTIPANEL_PRINT_REVIEW not in item.review_reasons for item in result.detections)
        assert sorted(item.bounding_box["x1"] for item in result.detections) == sorted(
            item.bounding_box["x1"] for item in proposals[1:]
        )
    else:
        item, = result.detections
        assert result.metadata["multipanel_fragments"] == 2
        assert item.mask["polygon"] == proposals[0].mask["polygon"]
        visible = segmentation.normalized_vertices_to_pixels(
            [(point["x"], point["y"]) for point in item.mask["visible_polygon"]], 940, 640,
        )
        assert np.max(np.linalg.norm(visible - parent, axis=1)) < 2.
        assert automatic_crop_review_reasons(settings, .99, True, True, item.review_reasons, None) == [
            MULTIPANEL_PRINT_REVIEW,
        ]


def test_physical_mount_evidence_preserves_nested_print():
    settings = Settings(DATABASE_URL="sqlite://", REDIS_URL="redis://")
    original, _, proposals = page_and_proposals(settings, 0, 0)
    nested = proposals[1]
    proposals[1] = replace(nested, mask={**nested.mask, "detector_source": "chromatic_mount"})

    assert segmentation._group_multipanel_prints(original, proposals) == (proposals, 0)


def test_valid_paper_separator_does_not_absorb_a_third_separate_print():
    settings = Settings(DATABASE_URL="sqlite://", REDIS_URL="redis://")
    original, _, proposals = page_and_proposals(settings, 4, 0, mixed=True)

    assert segmentation._group_multipanel_prints(original, proposals) == (proposals, 0)


def test_shared_outline_and_hold_survive_fusion_with_a_higher_scoring_model():
    settings = Settings(DATABASE_URL="sqlite://", REDIS_URL="redis://")
    original, _, proposals = page_and_proposals(settings, 0, 0)
    classical = replace(proposals[0], confidence=.60, mask={
        **proposals[0].mask, "source": "classical_border_quad", "detector_source": "edge_contour",
    })
    grouped, count = segmentation._group_multipanel_prints(original, [classical, *proposals[1:]])
    assert count == 2
    selected, _ = segmentation.fuse_candidate_evidence(original, proposals[:1], grouped,
                                                       model_evidence={0: {"score": 1.}})

    winner, = selected
    assert winner.mask["detector_source"] == "yolo_onnx"
    assert winner.mask["visible_polygon"] == grouped[0].mask["visible_polygon"]
    assert winner.review_reasons == [MULTIPANEL_PRINT_REVIEW]


@pytest.mark.parametrize(("angle", "scale"), [(0, 1), (8, 1), (-11, 1), (0, 3)])
def test_recovers_measured_stock_when_only_motif_proposals_exist(monkeypatch, angle, scale):
    settings = Settings(DATABASE_URL="sqlite://", REDIS_URL="redis://")
    original, truth, proposals = page_and_proposals(settings, 0, angle, jpeg_quality=91 if scale == 1 else None)
    if scale > 1:
        original = cv2.resize(original, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        ok, jpeg = cv2.imencode(".jpg", original, [cv2.IMWRITE_JPEG_QUALITY, 91])
        assert ok
        original = cv2.imdecode(jpeg, cv2.IMREAD_COLOR)
        # Upsampling replicates the first and last print pixel(s) at each edge.
        truth = scale * truth + np.float32([[0, 0], [1, 0], [1, 1], [0, 1]]) * (scale - 1)
    monkeypatch.setattr(segmentation, "_segment_album_page",
                        lambda *_: segmentation.SegmentationResult(proposals[1:], {}))
    prepared = cv2.convertScaleAbs(original, alpha=.70, beta=12)

    result = segmentation.segment_album_page(prepared, settings, source_image=original, protected_regions=[])

    item, = result.detections
    assert result.metadata["multipanel_outlines_recovered"] == 1
    assert item.mask["detector_source"] == RECOVERED_STOCK_SOURCE
    assert item.review_reasons == [MULTIPANEL_PRINT_REVIEW]
    actual = segmentation.normalized_vertices_to_pixels(
        [(point["x"], point["y"]) for point in item.mask["polygon"]], original.shape[1], original.shape[0],
    )
    assert np.max(np.linalg.norm(actual - truth, axis=1)) < 1


@pytest.mark.parametrize(
    "evidence", ["pale_blurred_gap", "rim_step", "interior_join", "slanted_join", "faint_slanted_join"],
)
def test_join_evidence_preserves_original_proposals_without_outer_mask(evidence):
    settings = Settings(DATABASE_URL="sqlite://", REDIS_URL="redis://")
    original, _, proposals = page_and_proposals(settings, 0, 0)
    if evidence == "pale_blurred_gap":
        backing = (210, 213, 216)
        original[:94] = original[554:] = backing
        original[:, :108] = original[:, 832:] = backing
        original[94:554, 465:466] = backing
        original = cv2.GaussianBlur(original, (0, 0), 1.2)
    elif evidence == "rim_step":
        original[94, 465:832] = (77, 84, 88)
        original[554, 465:832] = (229, 232, 235)
    else:
        bottom_x = 457 if evidence == "interior_join" else 487
        color = (223, 226, 229) if evidence == "faint_slanted_join" else (195, 200, 205)
        cv2.line(original, (457, 103), (bottom_x, 545), color, 2)
    candidates = segmentation.SegmentationResult(proposals[1:], {})

    assert segmentation._recover_multipanel_prints(original, candidates, settings) is candidates
    assert segmentation._group_multipanel_prints(original, proposals) == (proposals, 0)


def test_recovery_preserves_third_intersecting_print():
    settings = Settings(DATABASE_URL="sqlite://", REDIS_URL="redis://")
    original, _, proposals = page_and_proposals(settings, 0, 0)
    neighbor = segmentation._make_detection(
        box={}, points=np.float32([[711, 148], [896, 148], [896, 533], [711, 533]]),
        confidence=.94, image_width=940, image_height=640, settings=settings,
        source="chromatic_mount_boundary", detector_source="chromatic_mount",
    )
    candidates = segmentation.SegmentationResult([*proposals[1:], neighbor], {})

    assert segmentation._recover_multipanel_prints(original, candidates, settings) is candidates


@pytest.mark.parametrize("page", ["page_04", "page_10", "page_26"])
def test_real_separate_prints_with_narrow_gutters_are_retained(page):
    from scripts.validate_segmentation import parse_yolo_segmentation

    fixtures = Path(__file__).resolve().parents[3] / "data" / "golden_fixtures"
    original = cv2.imread(str(fixtures / f"{page}.jpg"))
    height, width = original.shape[:2]
    outlines = parse_yolo_segmentation((fixtures / "labels" / f"{page}.txt").read_text())
    settings = Settings(DATABASE_URL="sqlite://", REDIS_URL="redis://")
    detections = [
        segmentation._make_detection(
            box={}, points=segmentation.normalized_vertices_to_pixels(outline, width, height),
            confidence=.95, image_width=width, image_height=height, settings=settings,
            source="yolo_seg_mask", detector_source="yolo_onnx",
        )
        for outline in outlines
    ]
    proposals = segmentation.SegmentationResult(detections, {})

    assert segmentation._recover_multipanel_prints(original, proposals, settings) is proposals


def grid_page(settings, rows, columns):
    page = np.full((640, 940, 3), (77, 84, 88), np.uint8)
    page[94:554, 108:832] = (229, 232, 235)
    panel_width = (680 - (columns - 1) * 32) // columns
    panel_height = (408 - (rows - 1) * 32) // rows
    rng = np.random.default_rng(924)
    panels = []
    for row in range(rows):
        for column in range(columns):
            left = 130 + column * (panel_width + 32)
            top = 118 + row * (panel_height + 32)
            right = left + panel_width
            bottom = top + panel_height
            texture = rng.integers(35, 184, size=(16, 12, 3), dtype=np.uint8)
            page[top:bottom, left:right] = cv2.resize(texture, (panel_width, panel_height), interpolation=cv2.INTER_CUBIC)
            panels.append(np.float32([[left, top], [right - 1, top], [right - 1, bottom - 1], [left, bottom - 1]]))
    outer = np.float32([[108, 94], [831, 94], [831, 553], [108, 553]])
    detections = [
        segmentation._make_detection(
            box={}, points=quad, confidence=.92, image_width=940, image_height=640, settings=settings,
            source="yolo_seg_mask", detector_source="yolo_onnx",
        )
        for quad in (outer, *panels)
    ]
    return page, outer, detections


@pytest.mark.parametrize(("rows", "columns"), [(1, 3), (1, 4), (2, 2), (2, 3)])
def test_recovers_complete_motif_layout_without_outer_proposal(monkeypatch, rows, columns):
    settings = Settings(DATABASE_URL="sqlite://", REDIS_URL="redis://")
    original, expected, detections = grid_page(settings, rows, columns)
    monkeypatch.setattr(segmentation, "_segment_album_page",
                        lambda *_: segmentation.SegmentationResult(detections[1:], {}))

    result = segmentation.segment_album_page(original, settings, source_image=original, protected_regions=[])

    recovered, = result.detections
    assert result.metadata["multipanel_outlines_recovered"] == 1
    assert recovered.review_reasons == [MULTIPANEL_PRINT_REVIEW]
    outline = segmentation.normalized_vertices_to_pixels(
        [(point["x"], point["y"]) for point in recovered.mask["polygon"]], 940, 640,
    )
    assert np.max(np.linalg.norm(outline - expected, axis=1)) < 1


@pytest.mark.parametrize("color", [(196, 200, 205), (223, 226, 229)])
def test_partial_grid_join_vetoes_group_even_with_other_clean_separators(color):
    settings = Settings(DATABASE_URL="sqlite://", REDIS_URL="redis://")
    original, _, detections = grid_page(settings, 2, 2)
    # Only the bottom half of the vertical separator has a seam. The other
    # neighboring panels would still be connected by common-looking stock.
    cv2.line(original, (464, 339), (472, 545), color, 2)
    without_parent = segmentation.SegmentationResult(detections[1:], {})

    assert segmentation._group_multipanel_prints(original, detections) == (detections, 0)
    assert segmentation._recover_multipanel_prints(original, without_parent, settings) is without_parent


def test_recovers_adjacent_panels_without_absorbing_separate_print_in_same_row():
    settings = Settings(DATABASE_URL="sqlite://", REDIS_URL="redis://")
    original, _, detections = grid_page(settings, 1, 3)
    original[94:554, 584:588] = (77, 84, 88)
    without_parent = segmentation.SegmentationResult(detections[1:], {})

    recovered = segmentation._recover_multipanel_prints(original, without_parent, settings)

    assert len(recovered.detections) == 2
    assert recovered.metadata["multipanel_outlines_recovered"] == 1
    assert detections[-1] in recovered.detections


@pytest.mark.parametrize(("enclosing_proposal", "grid"), [(True, False), (False, False), (False, True)])
async def test_shared_stock_upload_preserves_all_panels_in_review_and_export(client, monkeypatch, enclosing_proposal, grid):
    from app.config import get_settings
    from app.services import pipeline

    monkeypatch.setenv("AUTO_ANALYZE_UPLOADS", "true")
    monkeypatch.setenv("MAX_UPLOAD_SIZE", "1048576")
    monkeypatch.setenv("MAX_REQUEST_SIZE", "2097152")
    monkeypatch.setenv("ENABLE_PHOTO_ORIENTATION_CORRECTION", "false")
    monkeypatch.setenv("ENABLE_PROTECTED_REGION_GUARD", "false")
    get_settings.cache_clear()
    if grid:
        original, _, proposals = grid_page(get_settings(), 2, 2)
    else:
        original, _, proposals = page_and_proposals(get_settings(), 0, 0)
    predictions = proposals if enclosing_proposal else proposals[1:]
    monkeypatch.setattr(segmentation, "_segment_album_page",
                        lambda *_: segmentation.SegmentationResult(predictions, {}))
    monkeypatch.setattr(pipeline, "run_paddle_ocr", lambda *_: [])
    album = await client.post("/api/v1/albums", json={"name": "Mehrbildabzug"})
    assert album.status_code == 201, album.text
    albums = f"/api/v1/albums/{album.json()['id']}"
    ok, payload = cv2.imencode(".jpg", original, [cv2.IMWRITE_JPEG_QUALITY, 97])
    assert ok
    upload = await client.post(f"{albums}/pages/upload",
                               files=[("files", ("album.jpg", payload.tobytes(), "image/jpeg"))])
    assert upload.status_code == 201, upload.text
    photo, = (await client.get(f"{albums}/photos")).json()
    assert photo["enhancement_applied"]["segmentation_review_reasons"] == [MULTIPANEL_PRINT_REVIEW]
    assert all(photo["crop_download_urls"][mode] is None for mode in ("full_print", "image_only"))
    photo_url = f"/api/v1/photos/{photo['id']}"
    outer = proposals[0].mask["polygon"]
    right = 809 if grid else 808
    inner = [{"x": x / 939, "y": y / 639} for x, y in ((130, 118), (right, 118), (right, 525), (130, 525))]
    reviewed = await client.patch(f"{photo_url}/bounding-box", json={
        "bounding_box": photo["bounding_box"],
        "segmentation_mask": {"polygon": outer, "warp_quad": outer, "image_quad": inner, "source": "manual_quad"},
    })
    assert reviewed.status_code == 200, reviewed.text
    for mode, expected in (("full_print", original[94:554, 108:832]), ("image_only", original[118:526, 130:right + 1])):
        response = await client.get(reviewed.json()["crop_download_urls"][mode])
        assert response.status_code == 200, response.text
        crop = cv2.imdecode(np.frombuffer(response.content, np.uint8), cv2.IMREAD_COLOR)
        assert crop.shape == expected.shape
        assert np.abs(crop.astype(float) - expected).mean() < 3
