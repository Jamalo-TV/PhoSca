from dataclasses import replace

import numpy as np
import pytest

from app.pipeline import segmentation
from app.pipeline.merged_print_review import MERGED_PRINT_PAIR, MERGED_PRINT_PAIR_REVIEW
from app.pipeline.perspective import correct_photo_crop


@pytest.mark.parametrize("case", ["observed", "flat_gap", "shifted_gap", "pending_claim", "pending_separator", "competing_owner"])
def test_merged_print_outlines_are_advice_on_one_held_row(case):
    width, height = 940, 730
    ribs = np.rint(24 * np.cos(2 * np.pi * (np.arange(height) - 328) / 12)).astype(np.int16)
    page = np.broadcast_to(np.uint8(np.array([124, 162, 194]) + ribs[:, None])[:, None, :], (height, width, 3)).copy()
    page[75:321, 280:541] = (105, 79, 64)
    page[339:596, 274:532] = (100, 73, 61)
    if case == "flat_gap":
        page[321:339, 270:547] = page[328, 240]
    if case == "shifted_gap":
        page[321:339, 270:547] = page[325:343, 600, None]

    def quad(left, top, right, bottom):
        return [{"x": x / (width - 1), "y": y / (height - 1)}
                for x, y in ((left, top), (right, top), (right, bottom), (left, bottom))]

    def detection(outline, confidence, detector_source):
        return segmentation.SegmentationDetection(
            bounding_box={"x1": min(p["x"] for p in outline), "y1": min(p["y"] for p in outline),
                          "x2": max(p["x"] for p in outline), "y2": max(p["y"] for p in outline)},
            mask={"polygon": outline, "source": "yolo_seg_mask", "detector_source": detector_source},
            confidence=confidence, aspect_ratio=.8, geometry_valid=True, review_reasons=[],
        )

    parent = detection(quad(248, 69, 550, 605), .49, "yolo_onnx")
    above = detection(quad(251, 75, 540, 320), .66, "yolo_onnx_tiled")
    below = detection(quad(274, 339, 531, 595), .86, "yolo_onnx_tiled")
    nominations = segmentation._nominate_merged_tile_pair([parent], [above, below], {0: {"score": .85}, 1: {"score": .85}})
    original, = nominations
    conflict = quad(430, 465, 549, 591)
    pending = quad(252, 325, 260, 332) if case == "pending_separator" else conflict
    metadata = {"fusion": {"deferred_print_proposals": [{"id": 0, "polygon": pending}]}} if case.startswith("pending_") else {}
    detections = [parent, detection(conflict, .81, "yolo_onnx")] if case == "competing_owner" else [parent]
    initial = segmentation.SegmentationResult(detections, metadata)
    result = segmentation._review_merged_tile_pair(page, initial, *original)
    if case != "observed":
        assert result is initial
        return

    checked, = result.detections
    pair = checked.mask[MERGED_PRINT_PAIR]
    assert result.metadata == initial.metadata
    assert len(pair["outlines"]) == 2 and all(len(quad) == 4 for quad in pair["outlines"])
    assert len(pair["separator"]) == 2
    assert checked == replace(parent, mask={**parent.mask, MERGED_PRINT_PAIR: pair}, review_reasons=[MERGED_PRINT_PAIR_REVIEW])
    assert pair["outlines"][0][0]["x"] * (width - 1) == pytest.approx(280, abs=9)
    crop = correct_photo_crop(page, checked.bounding_box, checked.mask, dual_mode=True, auto_orient=False,
                              protect_regions=False)
    modes = crop.metadata["crop_modes"]
    assert all(mode["status"] == "review_needed" and MERGED_PRINT_PAIR_REVIEW in mode["reason_codes"]
               for mode in modes.values())
    assert "suggested_warp_quad" not in modes["full_print"]
    assert crop.full_print_preview is None and crop.image_only is None
