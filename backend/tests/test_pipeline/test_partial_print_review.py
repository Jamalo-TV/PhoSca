from dataclasses import replace

import numpy as np
import pytest

from app.pipeline import segmentation
from app.pipeline.partial_print_review import PARTIAL_PRINT_EXTENT_REVIEW, PARTIAL_PRINT_EXTENT_TRACES
from app.pipeline.perspective import correct_photo_crop


@pytest.mark.parametrize("case", ["observed", "one_terminal_edge", "opaque_overlap", "shifted_flank", "competing_claim"])
def test_partial_print_keeps_open_guides_separate_from_crop_geometry(case):
    page = np.full((1110, 1441, 3), (135, 163, 187), np.uint8)
    page[690:997, 360:651] = (94, 79, 69)
    for left, right, color in ((405, 424, (152, 132, 116)), (478, 496, (58, 49, 40)), (558, 577, (170, 158, 147))):
        page[705:978, left:right] = color
    if case != "one_terminal_edge":
        page[997:1003, 360:651] = (222, 231, 238)
    if case == "opaque_overlap":
        page[690:851, 360:651] = (204, 212, 219)
    if case == "shifted_flank":
        page[858:997, 643:651] = (135, 163, 187)

    def quad(left, top, right, bottom):
        return [{"x": x / 1440, "y": y / 1109} for x, y in ((left, top), (right, top), (right, bottom), (left, bottom))]

    claim = quad(360, 690, 650, 850)
    tile = quad(360, 690, 650, 996)
    neighbor = {"polygon": quad(644, 886, 676, 940)}
    item = segmentation.SegmentationDetection(
        bounding_box={"x1": 360 / 1441, "y1": 690 / 1110, "x2": 651 / 1441, "y2": 851 / 1110},
        mask={"source": "classical_min_area_rect", "detector_source": "edge_contour", "polygon": claim,
              "fusion": {"proposal_count": 1, "proposal_families": ["classical"]}},
        confidence=.83, aspect_ratio=1.81, geometry_valid=True, review_reasons=[],
    )
    detections = [item]
    if case == "competing_claim":
        detections.append(replace(item, mask=neighbor))
    initial = segmentation.SegmentationResult(detections, {"fallback_used": True})
    output = segmentation._review_partial_tile_extent(page, initial, [tile])
    assert output.metadata == initial.metadata
    if case != "observed":
        assert output.detections == detections
        return

    checked, = output.detections
    traces = checked.mask[PARTIAL_PRINT_EXTENT_TRACES]
    assert len(traces) == 4 and all(len(span) == 2 for span in traces)
    assert checked == replace(item, mask={**item.mask, PARTIAL_PRINT_EXTENT_TRACES: traces},
                              review_reasons=[PARTIAL_PRINT_EXTENT_REVIEW])
    ending = sorted(np.mean([point["y"] * 1109 for point in span]) for span in traces[2:])
    assert ending == pytest.approx([996.5, 1002.5], abs=1)

    crop = correct_photo_crop(page, checked.bounding_box, checked.mask, dual_mode=True, auto_orient=False,
                              protect_regions=False)
    modes = crop.metadata["crop_modes"]
    assert crop.full_print_preview is None and crop.image_only is None
    assert modes["full_print"][PARTIAL_PRINT_EXTENT_TRACES] == traces
    assert all(mode["status"] == "review_needed" and PARTIAL_PRINT_EXTENT_REVIEW in mode["reason_codes"]
               for mode in modes.values())
    assert "suggested_warp_quad" not in modes["full_print"]
    after = correct_photo_crop(page, checked.bounding_box, checked.mask, dual_mode=True, auto_orient=False,
                               protect_regions=False, neighbor_masks=[neighbor])
    assert PARTIAL_PRINT_EXTENT_TRACES not in after.metadata["crop_modes"]["full_print"]
    assert after.metadata["crop_modes"]["full_print"]["status"] == "review_needed"
