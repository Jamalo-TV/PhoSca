from dataclasses import replace
from pathlib import Path

import numpy as np

from app.config import Settings
from app.pipeline import segmentation
from app.pipeline.crop_validation import automatic_crop_review_reasons
from app.pipeline.crossing_prints import CROSSING_PRINT_REVIEW, near_leaf_edge_traces
from app.pipeline.enhancement import prepare_for_cutout
from app.pipeline.geometry import normalized_polygon_overlap, normalized_vertices_to_pixels
from app.pipeline.image_ops import load_image
from app.pipeline.perspective import correct_photo_crop


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
