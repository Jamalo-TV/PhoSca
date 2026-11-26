from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pytest

from app.config import Settings
from app.pipeline import projective_prints, segmentation
from app.pipeline.adjacent_prints import ADJACENT_PRINT_REVIEW, adjacent_album_edge_outlines
from app.pipeline.crop_validation import automatic_crop_review_reasons
from app.pipeline.crossing_prints import staggered_print_source_traces
from app.pipeline.enhancement import prepare_for_cutout
from app.pipeline.fragmented_prints import (
    EXTENDED_PRINT_OUTLINE_REVIEW, FRAGMENT_OUTLINE_REVIEW, FRAGMENTED_PRINT_REVIEW, measure_extended_print_outline,
)
from app.pipeline.geometry import normalized_polygon_overlap, order_quad_cyclic
from app.pipeline.image_ops import load_image
from app.pipeline.perspective import correct_photo_crop
from app.pipeline.projective_prints import PROJECTIVE_PRINT_PROPOSAL


def _scene(case="continuous", turns=0):
    backing = (228, 232, 235)
    page = np.full((590, 680, 3), backing, np.uint8)
    outer = np.float32([[58, 55], [352, 55], [352, 503], [58, 503]])
    upper = np.float32([[68, 121], [346, 61], [290, 186], [62, 221]])
    lower = np.float32([[61, 240], [340, 198], [342, 498], [62, 497]])
    neighbor = np.float32([[370, 65], [627, 65], [627, 464], [370, 464]])
    page[55:504, 58:353] = (95, 124, 143)
    page[65:465, 370:628] = (116, 92, 144)
    if case != "flat_scene":
        for x in (96, 155, 222, 280):
            cv2.line(page, (x, 62), (x + 18, 497), (58, 86, 106), 6)
        cv2.ellipse(page, (201, 387), (75, 57), 0, 0, 360, (130, 135, 120), -1)
    if case in {"narrow_gutter", "blurred_gutter"}:
        page[209:211, 58:353] = backing
        if case == "blurred_gutter":
            page = cv2.GaussianBlur(page, (0, 0), 1.)
    elif case == "touching":
        page[210:504, 58:353] = cv2.add(page[210:504, 58:353], (19, 19, 19, 0))
    elif case == "touching_equal_stock":
        page[203:217, 58:353] = (197, 207, 215)
    elif case == "overlapping":
        page[201:406, 105:343] = (55, 89, 146)
    if turns:
        previous_width = page.shape[1]
        transform = np.float32([[0, 1, 0], [-1, 0, previous_width - 1]])
        outer, upper, lower, neighbor = [cv2.transform(quad[None], transform)[0]
                                         for quad in (outer, upper, lower, neighbor)]
        page = np.ascontiguousarray(np.rot90(page))
    return page, [outer, upper, lower, neighbor]


def _proposals(page, quads):
    settings = Settings(DATABASE_URL="sqlite://", REDIS_URL="redis://")
    height, width = page.shape[:2]
    proposals = []
    for quad, detector, confidence, shape_score in zip(
        quads, ("edge_contour", "contrast_region", "contrast_region", "edge_contour"),
        (.97, .8, .82, .9), (1., .72, .87, 1.), strict=True,
    ):
        proposals.append(segmentation._make_detection(
            box={}, points=quad, confidence=confidence, image_width=width, image_height=height,
            settings=settings, source="classical_border_quad", detector_source=detector,
            scores={"rectangularity": shape_score, "outline_supported_sides": 4.,
                    "exterior_background_sides": 4. if detector == "edge_contour" else 1.},
        ))
    return proposals


@pytest.mark.parametrize("turns", [0, 1])
def test_full_outline_survives_fragmented_classical_proposals(turns):
    page, quads = _scene(turns=turns)
    parent, upper, lower, neighbor = _proposals(page, quads)

    baseline, _ = segmentation.select_candidate_set([parent, upper, lower, neighbor])
    assert parent not in baseline
    selected, _ = segmentation.select_candidate_set([parent, upper, lower, neighbor], source_image=page)

    recovered, other = selected
    assert other == neighbor
    assert recovered.review_reasons == [FRAGMENTED_PRINT_REVIEW]
    assert recovered.mask["fragmented_print"] is True
    points = np.float32([(p["x"], p["y"]) for p in recovered.mask["warp_quad"]]) * (np.array(page.shape[1::-1]) - 1)
    assert min(np.max(np.linalg.norm(points - np.roll(quads[0], index, axis=0), axis=1)) for index in range(4)) < .25
    result = correct_photo_crop(page, recovered.bounding_box, recovered.mask, dual_mode=True, auto_orient=False,
                                protect_regions=False, neighbor_masks=[neighbor.mask])
    assert all(mode["status"] == "review_needed" for mode in result.metadata["crop_modes"].values())
    assert FRAGMENTED_PRINT_REVIEW in result.metadata["crop_modes"]["full_print"]["reason_codes"]


@pytest.mark.parametrize("case", ["narrow_gutter", "blurred_gutter", "touching", "touching_equal_stock",
                                  "overlapping", "flat_scene"])
def test_source_evidence_preserves_possible_separate_prints(case):
    page, quads = _scene(case)
    proposals = _proposals(page, quads)
    baseline, _ = segmentation.select_candidate_set(proposals)
    selected, _ = segmentation.select_candidate_set(proposals, source_image=page)

    assert selected == baseline


def test_supported_independent_candidate_vetoes_fragment_replacement():
    page, quads = _scene()
    parent, upper, lower, neighbor = _proposals(page, quads)
    competing = replace(neighbor, bounding_box={**neighbor.bounding_box, "x1": .43},
                        mask={**neighbor.mask, "polygon": [
                            {**point, "x": .43} if i in (0, 3) else point
                            for i, point in enumerate(neighbor.mask["polygon"])
                        ]})
    selected, _ = segmentation.select_candidate_set([parent, upper, lower, competing], source_image=page)
    assert all(FRAGMENTED_PRINT_REVIEW not in item.review_reasons for item in selected)


@pytest.mark.parametrize("case", [
    "continuous", "narrow_gutter", "touching", "overlapping", "unconnected_third", "fragment_crosses_gutter",
])
def test_weak_outer_edge_is_only_a_correction_suggestion(case):
    page, quads = _scene(case)
    for y in range(56, 503):
        # One bowed edge retains both corner arms but cannot support one line.
        depth = round(2 * np.sin(np.pi * (y - 55) / (503 - 55)) ** 2)
        page[y, 353:353 + depth] = page[y, 352]
    parent, upper, lower, neighbor = _proposals(page, quads)
    proposals = [parent, replace(upper, mask={**upper.mask, "detector_source": "edge_contour"}), lower, neighbor]
    if case in ("unconnected_third", "fragment_crosses_gutter"):
        # Two upper fragments genuinely share a scene. A third lies beyond a
        # strip of backing and only has a partial contour. An inaccurate upper
        # outline can also cross the visible gutter and overlap the third.
        start, gap, bottom = (390, 7, 389) if case == "unconnected_third" else (330, 2, 431)
        page[start:start + gap, 58:355] = (228, 232, 235)
        for y in range(start + gap, 504):
            depth = round(2 * np.sin(np.pi * (y - 55) / (503 - 55)) ** 2)
            page[y, 58:353 + depth] = (25, 170, 69)
        cv2.line(page, (99, start + gap + 10), (132, 499), (81, 178, 144), 8)
        cv2.line(page, (210, start + gap + 20), (240, 495), (123, 49, 112), 9)
        lower_outline = quads[2].copy()
        lower_outline[2:, 1] = bottom
        proposals[2] = _proposals(page, [*quads[:2], lower_outline, quads[3]])[2]
        third_outline = np.float32([[85, start + gap + 5], [348, start + gap + 13], [328, 501], [62, 500]])
        _, _, third, _ = _proposals(page, [*quads[:2], third_outline, quads[3]])
        proposals.append(replace(third, confidence=.76))
    baseline, _ = segmentation.select_candidate_set(proposals)
    selected, _ = segmentation.select_candidate_set(proposals, source_image=page)
    assert [item.bounding_box for item in selected] == [item.bounding_box for item in baseline]
    if case != "continuous":
        assert selected == baseline
        return
    for item, original in zip(selected, baseline, strict=True):
        if item == neighbor:
            continue
        assert FRAGMENT_OUTLINE_REVIEW in item.review_reasons
        assert item.mask["polygon"] == original.mask["polygon"]
        hint = item.mask["fragment_outline_suggestion"]
        points = np.float32([(point["x"], point["y"]) for point in hint["quad"]])
        assert hint["uncertain_edge"] == 1
        assert normalized_polygon_overlap(points, segmentation._detection_polygon(parent)).iou > .98
        result = correct_photo_crop(page, item.bounding_box, item.mask, dual_mode=True, auto_orient=False,
                                    protect_regions=False, neighbor_masks=[other.mask for other in selected if other != item])
        assert all(mode["status"] == "review_needed" for mode in result.metadata["crop_modes"].values())
        assert FRAGMENT_OUTLINE_REVIEW in result.metadata["crop_modes"]["full_print"]["reason_codes"]


def test_nested_classical_fragment_recovers_the_original_print_for_review(tmp_path):
    fixtures = Path(__file__).resolve().parents[3] / "data" / "golden_fixtures"
    image = load_image(fixtures / "page_37.jpg")
    truth = np.loadtxt(fixtures / "labels" / "page_37.txt")[:, 1:].reshape(2, 4, 2)
    prepared, _ = prepare_for_cutout(image)
    settings = Settings(
        DATABASE_URL="sqlite://", REDIS_URL="redis://", YOLO_MODEL_PATH=tmp_path / "unused.onnx",
        ENABLE_CLASSICAL_SEGMENTATION_FALLBACK=True, ENABLE_PRECISE_PHOTO_CROPS=True,
    )

    result = segmentation.segment_album_page(prepared, settings, source_image=image)

    recovered, = result.detections
    assert recovered.mask["detector_source"] == "contrast_region"
    assert EXTENDED_PRINT_OUTLINE_REVIEW in recovered.review_reasons
    assert ADJACENT_PRINT_REVIEW in recovered.review_reasons
    outline = segmentation._detection_polygon(recovered)
    matches = [normalized_polygon_overlap(outline, target) for target in truth]
    assert max(match.iou for match in matches) > .98
    assert min(match.second_containment for match in matches) < .02
    suggestion = [(point["x"], point["y"]) for point in recovered.mask["adjacent_print_suggestion"]]
    assert normalized_polygon_overlap(suggestion, truth[1]).iou > .98
    assert normalized_polygon_overlap(suggestion, truth[0]).first_containment < .02
    assert ADJACENT_PRINT_REVIEW in automatic_crop_review_reasons(
        settings, 1., True, True, recovered.review_reasons, None,
    )
    crop = correct_photo_crop(image, recovered.bounding_box, recovered.mask, dual_mode=True, auto_orient=False,
                              protect_regions=False, enable_classical_retry=True)
    for mode in crop.metadata["crop_modes"].values():
        assert mode["status"] == "review_needed"
        assert EXTENDED_PRINT_OUTLINE_REVIEW in mode["reason_codes"]
    assert "PRINT_BOUNDARY_CURVED" in crop.metadata["crop_modes"]["full_print"]["reason_codes"]


def test_blank_album_leaf_at_a_dark_edge_has_no_adjacent_print_outline():
    image = np.full((820, 1120, 3), 83, np.uint8)
    page = np.int32([[100, 40], [975, 55], [958, 782], [100, 778]])
    cv2.fillConvexPoly(image, page, (200, 207, 210))
    neighbor = np.float32([[211, 168], [558, 174], [557, 668], [211, 663]])
    cv2.fillConvexPoly(image, neighbor.astype(np.int32), (57, 109, 95))

    assert adjacent_album_edge_outlines(image, neighbor) == []


def test_missing_neighbor_traces_require_source_sides_beyond_the_album_edge():
    source = Path(__file__).resolve().parents[3] / "data" / "golden_fixtures" / "page_11.jpg"
    image = load_image(source)
    anchor = np.float32([[784, 633], [1496, 633], [1496, 1740], [784, 1740]])
    assert staggered_print_source_traces(image, anchor)

    neighbor = np.float32([[1537, 764], [2312, 748], [2272, 1835], [1551, 1860]])
    cover = neighbor + (neighbor - neighbor.mean(axis=0)) * .048
    stock = np.median(image[420:590, 1590:2100], axis=(0, 1)).tolist()
    cv2.fillConvexPoly(image, np.rint(cover).astype(np.int32), stock)
    assert staggered_print_source_traces(image, anchor) == []


@pytest.mark.parametrize(("turn", "expected_count"), [(1, 0), (2, 1), (3, 0)])
def test_independently_encoded_quarter_turns_recover_neighbor_traces(turn, expected_count, tmp_path):
    source = Path(__file__).resolve().parents[3] / "data" / "golden_fixtures" / "page_11.jpg"
    image = load_image(source)
    height, width = image.shape[:2]
    ok, jpeg = cv2.imencode(".jpg", np.rot90(image, turn), [cv2.IMWRITE_JPEG_QUALITY, 75])
    assert ok
    rotated = cv2.imdecode(jpeg, cv2.IMREAD_COLOR)
    settings = Settings(DATABASE_URL="sqlite://", REDIS_URL="redis://", YOLO_MODEL_PATH=tmp_path / "absent.onnx")
    prepared, *_ = prepare_for_cutout(rotated)
    result = segmentation.segment_album_page(prepared, settings, source_image=rotated, protected_regions=[])

    assert len(result.detections) == expected_count
    drafts = result.metadata["fusion"].get("deferred_print_proposals", [])
    assert [draft["reason"] for draft in drafts] == ([] if expected_count else [PROJECTIVE_PRINT_PROPOSAL])
    runs = result.metadata["missing_print_edge_traces"]
    assert len(runs) >= 3
    # Undo the source turn to measure the published traces in original pixels.
    xy = np.float32([(point["x"], point["y"]) for run in runs for point in run])
    x, y = xy.T
    x, y = [(1 - y, x), (1 - x, 1 - y), (y, 1 - x)][turn - 1]
    x, y = x * (width - 1), y * (height - 1)
    assert np.all((x >= 1533) & (x <= 2314) & (y >= 745) & (y <= 1862))
    assert np.count_nonzero((x > 2278) & (y > 1000) & (y < 1500)) >= 10
    assert np.count_nonzero(y < 785) >= 20
    assert np.count_nonzero(y > 1825) >= 20


@pytest.mark.parametrize("budget_veto", [True, False])
def test_projective_trace_retry_requires_a_completed_empty_primary_bank(monkeypatch, budget_veto):
    searches = []

    def seeds(image, working_side):
        searches.append(working_side)
        return None if budget_veto else [np.zeros((4, 2), np.float32)]

    monkeypatch.setattr(projective_prints, "_projective_seed_bank", seeds)
    monkeypatch.setattr(projective_prints, "_verified_projective_outline", lambda *_: None)
    page = np.zeros((1920, 2560, 3), np.uint8)

    assert projective_prints.projective_print_review(page) == (None, [])
    assert searches == [1280]


@pytest.mark.parametrize(("join", "drift"), [("none", 0), ("touching", 53), ("gutter", 97), ("overlap", 97)])
def test_diagonal_fragment_cannot_absorb_a_print_across_an_oblique_join(join, drift):
    backing = (213, 220, 225)
    image = np.full((1150, 1000, 3), backing, np.uint8)
    image[130:1001, 125:871] = (105, 125, 149)
    for index, y in enumerate(range(253, 944, 82)):
        color = (35 + index % 3 * 95, 65 + index % 3 * 54, 125 + index % 2 * 75)
        cv2.line(image, (430, y - 23), (854, y + 31), color, 5)
        cv2.ellipse(image, (680, y), (81, 25), 20, 0, 360, (110, 63, 90), 4)
    if join != "none":
        boundary = np.int32([[644, 130], [870, 130], [870, 1000], [644 + drift, 1000]])
        print_pixels = cv2.fillConvexPoly(np.zeros(image.shape[:2], np.uint8), boundary, 1).astype(bool)
        second = np.full_like(image, (55, 111, 132))
        for y in range(259, 940, 83):
            cv2.line(second, (570, y + 45), (865, y - 27), (76, 182, 162), 6)
        image[print_pixels] = second[print_pixels]
        if join == "gutter":
            cv2.line(image, (644, 130), (644 + drift, 1000), backing, 4)
        elif join == "overlap":
            cv2.line(image, (644, 130), (644 + drift, 1000), (77, 93, 102), 3)
    inner = order_quad_cyclic(np.float32([[125, 130], [500, 130], [868, 1000], [125, 1000]]))
    outer = order_quad_cyclic(np.float32([[119, 123], [877, 123], [877, 1008], [119, 1008]]))

    measured = measure_extended_print_outline(image, inner, outer)

    assert (measured is not None) == (join == "none")
