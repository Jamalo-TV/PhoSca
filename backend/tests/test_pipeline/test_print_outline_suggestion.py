import cv2
import numpy as np
import pytest

from app.pipeline import perspective
from app.pipeline.print_edges import PRINT_EDGE_CONFIRMATION
from app.pipeline.print_outline_suggestion import suggest_outer_print_outline


_OUTER = np.float32([[100, 90], [705, 90], [705, 560], [100, 560]])
_SEED = np.float32([[133, 110], [675, 121], [683, 549], [133, 549]])


def _source(layout: str) -> tuple[np.ndarray, list[np.ndarray]]:
    stock = (224, 127, 122)
    backing = (186, 132, 150)
    lab = np.full((650, 880, 3), stock if layout == "matching_backing" else backing, np.uint8)
    lab[90:561, 100:706] = stock
    lab[118:534, 124:682] = (85, 128, 125)
    lab[263:444, 230:504] = (205, 127, 119)
    neighbors = []
    if layout == "narrow_gap":
        lab[90:561, 689:692] = backing
    elif layout == "overlapping_neighbor":
        # Retain the original colour around the shared boundary. The other
        # proposal's position supplies independent ownership evidence.
        neighbors.append(np.float32([[695, 165], [831, 165], [831, 430], [695, 430]]))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR), neighbors


@pytest.mark.parametrize("layout", ["visible_stock", "matching_backing", "narrow_gap", "overlapping_neighbor"])
def test_alternative_needs_visible_paper_perimeter_without_new_overlap(layout):
    image, neighbors = _source(layout)

    alternative = suggest_outer_print_outline(image, _SEED, _SEED, neighbors)

    if layout == "visible_stock":
        assert alternative is not None
        assert np.max(np.linalg.norm(alternative - _OUTER, axis=1)) <= 2
    else:
        assert alternative is None


def test_outer_suggestion_preserves_the_rendered_crop_and_both_review_statuses(monkeypatch):
    image, _ = _source("visible_stock")
    height, width = image.shape[:2]
    polygon = perspective._vertices_on_page(_SEED, width, height)
    mask = {"polygon": polygon, "detector_source": "contrast_region"}

    def inspect(page, quad, *_args, **_kwargs):
        return perspective._PrintInspection(
            quad, perspective._validate_quad(quad, width, height), {}, ["PRINT_BOUNDARY_UNSUPPORTED"],
            False, {"passed": True, "regions_checked": 0, "violations": []},
        )

    monkeypatch.setattr(perspective, "_inspect_print", inspect)
    options = dict(dual_mode=True, auto_orient=False, protect_regions=False)
    before = perspective.correct_photo_crop(image, {}, mask, **options)
    reviewed = perspective.correct_photo_crop(image, {}, mask, enable_classical_retry=True, **options)

    assert np.array_equal(before.rectified_untrimmed, reviewed.rectified_untrimmed)
    assert before.metadata["warp_quad"] == reviewed.metadata["warp_quad"]
    prior_modes = before.metadata["crop_modes"]
    modes = reviewed.metadata["crop_modes"]
    candidate = [(point["x"], point["y"]) for point in modes["full_print"]["suggested_warp_quad"]]
    assert np.max(np.linalg.norm(np.float32(candidate) * [width - 1, height - 1] - _OUTER, axis=1)) < 2.1
    for name, mode in modes.items():
        assert mode["status"] == prior_modes[name]["status"] == "review_needed"
        assert mode["reason_codes"] == [*prior_modes[name]["reason_codes"], PRINT_EDGE_CONFIRMATION]
        assert mode.get("page_to_output") == prior_modes[name].get("page_to_output")
    assert "warp_quad" not in mask
