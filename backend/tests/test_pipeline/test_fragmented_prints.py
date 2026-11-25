from dataclasses import replace

import cv2
import numpy as np
import pytest

from app.config import Settings
from app.pipeline import segmentation
from app.pipeline.fragmented_prints import FRAGMENT_OUTLINE_REVIEW, FRAGMENTED_PRINT_REVIEW
from app.pipeline.geometry import normalized_polygon_overlap
from app.pipeline.perspective import correct_photo_crop


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
