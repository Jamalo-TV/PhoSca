import cv2
import numpy as np
import pytest

from app.pipeline.possible_print_regions import measure_possible_print_regions, possible_print_region_window


def _two_regions(variant: str = "observed"):
    height, width = 998, 1400
    lab = np.full((height, width, 3), [195, 128, 142], dtype=np.uint8)
    if variant != "flat_backing":
        phase = 2 * np.pi * (np.arange(height) - 390) / 7
        lab[..., 0] = np.rint(195 + 32 * np.cos(phase))[:, None]
    if variant == "ordinary_stock":
        lab[130:671, 575:877] = [228, 128, 129]
    lab[141:389, 581:840] = [104, 131, 114]
    lab[393:665, 599:866] = [124, 139, 116]
    if variant == "continuous_scene":
        lab[385:398, 577:872] = [120, 132, 115]

    scale = np.float32([width - 1, height - 1])

    def box(left, top, right, bottom):
        return np.float32([[left, top], [right, top], [right, bottom], [left, bottom]]) / scale

    parent = box(578, 133, 891, 673)
    # The upper inference leaks into the backing by roughly 55 pixels.
    candidates = [(box(580, 138, 895, 387), .89, .88), (box(595, 396, 866, 665), .87, .86)]
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR), parent, candidates


def test_possible_regions_localize_both_sides_of_a_narrow_native_gap():
    image, parent, candidates = _two_regions()
    before = parent.copy()

    assert possible_print_region_window(parent, image.shape[:2]) == (359, 0, 1110, 837)
    first, second = measure_possible_print_regions(image, parent, candidates, [])

    assert first[:, 0].max() == pytest.approx(840, abs=3)
    assert first[2:, 1] == pytest.approx([388, 388], abs=1)
    assert second[:2, 1] == pytest.approx([393, 393], abs=1)
    assert cv2.intersectConvexConvex(first, second)[0] == 0
    assert np.array_equal(parent, before)


@pytest.mark.parametrize("variant", ["flat_backing", "ordinary_stock", "continuous_scene"])
def test_possible_regions_require_an_independently_supported_backing_lane(variant):
    image, parent, candidates = _two_regions(variant)
    assert measure_possible_print_regions(image, parent, candidates, []) is None


def test_possible_regions_abstain_when_a_neighbor_claim_intersects_either_region():
    image, parent, candidates = _two_regions()
    width, height = image.shape[1::-1]
    claim = [{"x": x / (width - 1), "y": y / (height - 1)}
             for x, y in ((777, 531), (881, 531), (881, 641), (777, 641))]
    assert measure_possible_print_regions(image, parent, candidates, [{"polygon": claim}]) is None


def test_possible_regions_require_exactly_one_candidate_at_each_end():
    image, parent, candidates = _two_regions()
    assert measure_possible_print_regions(image, parent, candidates[:1], []) is None
    assert measure_possible_print_regions(image, parent, [*candidates, candidates[0]], []) is None


def test_window_rejects_an_ordinary_short_parent_and_accepts_quarter_turn():
    image, parent, candidates = _two_regions()
    assert possible_print_region_window(candidates[1][0], image.shape[:2]) is None
    assert possible_print_region_window(np.float32([[.1, .1]]), image.shape[:2]) is None
    turned = np.column_stack((parent[:, 1], 1 - parent[:, 0]))
    assert possible_print_region_window(turned, image.shape[:2][::-1]) == pytest.approx((0, 289, 837, 1040), abs=1)


def test_second_region_comparison_uses_one_local_view_and_preserves_the_existing_pair(monkeypatch):
    from app.pipeline import segmentation
    from app.pipeline.possible_print_regions import POSSIBLE_PRINT_REGIONS, POSSIBLE_PRINT_REGIONS_REVIEW

    image, parent, candidates = _two_regions()
    height, width = image.shape[:2]
    left, top, right, bottom = possible_print_region_window(parent, (height, width))

    def detection(outline, confidence, **evidence):
        points = [{"x": float(x), "y": float(y)} for x, y in outline]
        return segmentation.SegmentationDetection(
            bounding_box={}, mask={"source": "yolo_seg_mask", "detector_source": "yolo_onnx", "polygon": points,
                                   "fusion": {"proposal_count": 1, "proposal_families": ["model"]}, **evidence},
            confidence=confidence, aspect_ratio=1., geometry_valid=True, review_reasons=[],
        )

    first = detection(parent - [.35, 0], .60, merged_print_pair={"outlines": []})
    unresolved = detection(parent, .61)
    result = segmentation.SegmentationResult([first, unresolved], {"fusion": {}})
    calls = []
    monkeypatch.setattr(segmentation, "_detection_boundary_evidence", lambda *_: {"score": .88})

    def infer(tile):
        calls.append(tile.shape)
        return [detection((outline * [width - 1, height - 1] - [left, top]) / [right - left - 1, bottom - top - 1], score)
                for outline, score, _ in candidates]

    reviewed = segmentation._review_possible_print_regions(image, image, result, infer, .70)
    assert calls == [image[top:bottom, left:right].shape]
    assert len(reviewed.detections) == 2 and reviewed.detections[0] is first
    second = reviewed.detections[1]
    assert second.mask["polygon"] == unresolved.mask["polygon"]
    assert second.review_reasons == [POSSIBLE_PRINT_REGIONS_REVIEW]
    assert len(second.mask[POSSIBLE_PRINT_REGIONS]["outlines"]) == 2

    competing = segmentation.SegmentationResult([unresolved, unresolved], result.metadata)
    assert segmentation._review_possible_print_regions(image, image, competing, infer, .70) is competing
    assert len(calls) == 1
