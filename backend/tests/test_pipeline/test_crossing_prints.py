from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pytest

from app.config import Settings
from app.pipeline import segmentation
from app.pipeline.crop_validation import automatic_crop_review_reasons
from app.pipeline.crossing_prints import CONTESTED_TILE_OUTLINE, CROSSING_PRINT_REVIEW, near_leaf_edge_traces
from app.pipeline.enhancement import prepare_for_cutout
from app.pipeline.geometry import normalized_polygon_overlap, normalized_vertices_to_pixels
from app.pipeline.image_ops import load_image
from app.pipeline.perspective import correct_photo_crop


@pytest.mark.parametrize("case", ["supported_tile", "covered_tile", "interior_frame", "second_owner", "two_choices", "pending"])
def test_tile_outline_is_an_exclusive_alternative_to_unsupported_frame_claim(case):
    image = np.full((1001, 1441, 3), (120, 68, 41), np.uint8)
    image[385:651, 95:326] = (207, 230, 232)
    image[400:637, 110:313] = (69, 60, 101)
    frame = np.float32([[0, 245], [435, 0], [605, 97], [57, 525]])
    tile = np.float32([[95, 385], [325, 385], [325, 650], [95, 650]])
    if case == "covered_tile":
        cv2.fillConvexPoly(image, frame.astype(np.int32), (185, 200, 200))
    if case == "interior_frame":
        frame += 5

    def polygon(points):
        return [{"x": float(x / 1440), "y": float(y / 1000)} for x, y in points]

    mask = {"source": "classical_min_area_rect", "detector_source": "contrast_region", "polygon": polygon(frame),
            "fusion": {"proposal_count": 1, "proposal_families": ["classical"]}}
    candidate = segmentation.SegmentationDetection(
        bounding_box={"x1": 0, "y1": 0, "x2": .43, "y2": .54}, mask=mask,
        confidence=.81, aspect_ratio=2.4, geometry_valid=True, review_reasons=[],
    )
    offers = [polygon(tile)]
    detections = [candidate]
    metadata = {"fallback_used": True}
    if case == "second_owner":
        detections.append(replace(candidate, mask={**mask, "polygon": polygon(tile + 10)}))
    if case == "two_choices":
        offers.append(polygon(tile + 1))
    if case == "pending":
        metadata["fusion"] = {"deferred_print_proposals": [{"id": 0, "polygon": polygon(tile + 10)}]}
    original = segmentation.SegmentationResult(detections, metadata)

    result = segmentation._offer_unclaimed_tiled_prints(image, original, offers)
    assert result.metadata == original.metadata
    if case == "supported_tile":
        edited, = result.detections
        assert edited == replace(candidate, mask={
            **mask, "crossing_print_suggestion": offers[0], "crossing_print_suggestion_source": CONTESTED_TILE_OUTLINE,
        }, review_reasons=[CROSSING_PRINT_REVIEW])
        crop = correct_photo_crop(image, edited.bounding_box, edited.mask, dual_mode=True, auto_orient=False,
                                  protect_regions=False)
        assert all(mode["status"] == "review_needed" for mode in crop.metadata["crop_modes"].values())
    else:
        assert result.detections == original.detections


def test_crossing_proposal_suggests_native_outline_without_claiming_a_print(tmp_path, monkeypatch):
    fixtures = Path(__file__).resolve().parents[3] / "data" / "golden_fixtures"
    image = load_image(fixtures / "page_04.jpg")
    truth = np.loadtxt(fixtures / "labels" / "page_04.txt")[:, 1:].reshape(3, 4, 2)
    prepared, _ = prepare_for_cutout(image)
    settings = Settings(
        DATABASE_URL="sqlite://", REDIS_URL="redis://", YOLO_MODEL_PATH=tmp_path / "unused.onnx",
        ENABLE_CLASSICAL_SEGMENTATION_FALLBACK=True, ENABLE_PRECISE_PHOTO_CROPS=True,
    )
    enabled = segmentation._suggest_crossing_print_outline
    with monkeypatch.context() as baseline_run:
        baseline_run.setattr(segmentation, "_suggest_crossing_print_outline", lambda _image, result, **_: result)
        baseline = segmentation.segment_album_page(prepared, settings, source_image=image)

    result = segmentation.segment_album_page(prepared, settings, source_image=image)
    other, reviewed = result.detections
    assert other == baseline.detections[0]
    assert reviewed.bounding_box == baseline.detections[1].bounding_box
    assert reviewed.mask["polygon"] == baseline.detections[1].mask["polygon"]
    assert "warp_quad" not in reviewed.mask
    assert reviewed.review_reasons == [CROSSING_PRINT_REVIEW]
    suggestion = [(point["x"], point["y"]) for point in reviewed.mask["crossing_print_suggestion"]]
    assert normalized_polygon_overlap(suggestion, truth[2]).iou > .98
    assert all(normalized_polygon_overlap(suggestion, outline).first_containment < .02 for outline in truth[:2])
    assert CROSSING_PRINT_REVIEW in automatic_crop_review_reasons(settings, 1., True, True, reviewed.review_reasons, None)
    traces = result.metadata["missing_print_edge_traces"]
    height, width = image.shape[:2]
    runs = [normalized_vertices_to_pixels([(point["x"], point["y"]) for point in run], width, height) for run in traces]
    assert len(runs) >= 4
    assert all(run[:, 0].max() < 870 and np.linalg.norm(run[0] - run[-1]) > 5 for run in runs)
    outer, = [run for run in runs if run[:, 0].max() < 160]
    assert np.ptp(outer[:, 1]) > 1000
    chord = np.interp(outer[:, 1], outer[[-1, 0], 1], outer[[-1, 0], 0])
    assert np.max(np.abs(outer[:, 0] - chord)) > 4
    assert all("missing_print_edge_traces" not in photo.mask for photo in result.detections)

    crop = correct_photo_crop(image, reviewed.bounding_box, reviewed.mask, dual_mode=True, auto_orient=False,
                              protect_regions=False, enable_classical_retry=True, neighbor_masks=[other.mask])
    assert all(mode["status"] == "review_needed" for mode in crop.metadata["crop_modes"].values())
    assert CROSSING_PRINT_REVIEW in crop.metadata["crop_modes"]["full_print"]["reason_codes"]

    # Even the same pixel evidence cannot replace existing ownership. This
    # also covers an outer print whose printed shading mimics a separate print.
    quad = normalized_vertices_to_pixels(suggestion, image.shape[1], image.shape[0])
    assert not near_leaf_edge_traces(image, prepared, quad, [truth[1]], segmentation._quadrilateral_from_contour)
    monkeypatch.setattr(segmentation, "crossing_print_outlines", lambda *_: [quad])
    owner = replace(other, mask={**other.mask, "polygon": reviewed.mask["crossing_print_suggestion"]})
    conflict = replace(baseline, detections=[*baseline.detections, owner])
    assert enabled(image, conflict) == conflict
    assert enabled(image, replace(baseline, metadata={"fallback_used": True})).detections == baseline.detections
