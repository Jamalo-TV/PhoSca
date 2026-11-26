from dataclasses import replace

import cv2
import numpy as np
import pytest

from app.config import Settings
from app.pipeline import segmentation
from app.pipeline.geometry import normalized_polygon_overlap, order_quad_cyclic
from app.pipeline.perspective import correct_photo_crop
from app.pipeline.scalloped_prints import SCALLOPED_PRINT_CONFLICT, SCALLOPED_PRINT_REVIEW, measure_scalloped_print


def _scan(kind: str, turns: int = 0):
    page = np.full((525, 705, 3), 255, np.uint8)
    paper = np.float32([[65, 62], [575, 62], [575, 425], [65, 425]])
    motif = np.float32([[86, 81], [554, 81], [554, 406], [86, 406]])
    ring = []
    for start, end in zip(paper, np.roll(paper, -1, axis=0), strict=True):
        edge = end - start
        length = np.linalg.norm(edge)
        fractions = np.linspace(0, 1, round(length) + 1, dtype=np.float32)
        depth = 4 * (1 - np.cos(2 * np.pi * round(length / 29) * fractions)) if kind != "straight" else 0 * fractions
        inward = np.float32([-edge[1], edge[0]]) / length
        ring.extend(start + fractions[:, None] * edge + depth[:, None] * inward)
    ring = np.rint(ring).astype(np.int32)
    cv2.fillPoly(page, [ring], (198, 211, 228))
    if kind == "layered":
        cv2.rectangle(page, (81, 76), (559, 411), (155, 169, 181), -1)
    if kind == "thin_layer":
        page[80, 86:555] = (65, 70, 75)
    cv2.fillConvexPoly(page, motif.astype(np.int32), (108, 124, 144) if kind == "thin_layer" else (85, 104, 125))
    cv2.circle(page, (213, 185), 45, (110, 148, 179), -1)
    if kind == "occluded":
        cv2.rectangle(page, (396, 40), (505, 119), (66, 90, 115), -1)
    if turns:
        transform = np.float32([[0, 1, 0], [-1, 0, page.shape[1] - 1]])
        ring, paper, motif = [cv2.transform(points.astype(np.float32)[None], transform)[0]
                              for points in (ring, paper, motif)]
        page = np.ascontiguousarray(np.rot90(page))
    return page, ring, order_quad_cyclic(paper), order_quad_cyclic(motif)


def _propose(page, quad, source):
    settings = Settings(DATABASE_URL="sqlite://", REDIS_URL="redis://")
    width, height = page.shape[1], page.shape[0]
    return segmentation._make_detection(
        box={}, points=quad, confidence=.84, image_width=width, image_height=height,
        settings=settings, source="classical_border_quad", detector_source=source,
    )


@pytest.mark.parametrize("turns", [0, 1])
def test_scalloped_silhouette_recovers_stock_and_keeps_downloads_in_review(turns):
    page, outline, paper, motif = _scan("scalloped", turns)
    measured = measure_scalloped_print(page, motif)
    assert measured is not None
    scale = np.array(page.shape[1::-1]) - 1
    assert normalized_polygon_overlap(measured.outline / scale, outline / scale, raster_size=1024).iou > .995
    inside = _propose(page, motif, "edge_contour")
    outside = _propose(page, paper, "contrast_region")
    result = segmentation._recover_scalloped_prints(page, segmentation.SegmentationResult([inside, outside], {}))
    (detection,) = result.detections
    assert detection.mask.get("scalloped_alternate_polygon") == inside.mask["polygon"]
    crop = correct_photo_crop(page, detection.bounding_box, detection.mask, dual_mode=True, auto_orient=False,
                              protect_regions=False)
    modes = crop.metadata["crop_modes"]
    assert all(mode["status"] == "review_needed" for mode in modes.values())
    assert modes["full_print"]["reason_codes"] == [SCALLOPED_PRINT_REVIEW]
    assert crop.full_print_preview is not None
    pixels = crop.full_print_preview
    assert pixels.shape[2] == 4 and np.any(pixels[..., 3] == 0)
    assert np.max(pixels[pixels[..., 3] > 0, :3]) < 245
    assert crop.image_only_preview is not None
    assert len(modes["image_only"]["suggested_warp_quad"]) == 4


@pytest.mark.parametrize("kind", ["straight", "occluded"])
def test_straight_or_interrupted_stock_does_not_infer_a_scalloped_outline(kind):
    page, _, paper, _ = _scan(kind)
    assert measure_scalloped_print(page, paper) is None


@pytest.mark.parametrize("kind", ["layered", "thin_layer"])
def test_scalloped_outline_retains_a_separate_layer_hypothesis(kind):
    page, _, paper, motif = _scan(kind)
    inside = _propose(page, motif, "edge_contour")
    outside = _propose(page, paper, "contrast_region")
    result = segmentation._recover_scalloped_prints(page, segmentation.SegmentationResult([inside, outside], {}))
    assert len(result.detections) == 2
    assert [item.mask for item in result.detections] == [inside.mask, outside.mask]
    if kind == "thin_layer":
        assert all(item.review_reasons == ["PRINT_SCALLOPED_OWNERSHIP_UNCERTAIN"] for item in result.detections)


@pytest.mark.parametrize("separate_layer", [False, True])
def test_scalloped_outline_guides_review_of_a_partial_claim_without_adopting_it(separate_layer):
    page, outline, _, motif = _scan("scalloped")
    partial = np.float32([[310, 105], [528, 105], [528, 378], [310, 378]])
    if separate_layer:
        cv2.fillConvexPoly(page, partial.astype(np.int32), (80, 105, 130))
        cv2.polylines(page, [partial.astype(np.int32)], True, (216, 221, 226), 2)
    parent = _propose(page, motif, "edge_contour")
    contender = _propose(page, partial, "yolo_onnx")
    result = segmentation._recover_scalloped_prints(page, segmentation.SegmentationResult([parent, contender], {}))
    main, other = result.detections
    assert main.review_reasons == [SCALLOPED_PRINT_CONFLICT]
    assert main.mask["polygon"] == parent.mask["polygon"] and other.mask["polygon"] == contender.mask["polygon"]
    group = main.mask["scalloped_ownership_group"]
    assert other.mask["scalloped_ownership_group"] == group
    assert "scalloped_ownership_outline" not in other.mask
    hint = main.mask["scalloped_ownership_outline"]
    scale = np.float32(page.shape[1::-1]) - 1
    points = np.float32([(p["x"], p["y"]) for p in hint])
    assert normalized_polygon_overlap(points, outline / scale, raster_size=1024).iou > .995
    params = dict(dual_mode=True, auto_orient=False, protect_regions=False, neighbor_masks=[other.mask])
    baseline = correct_photo_crop(page, parent.bounding_box, parent.mask, **params)
    current = correct_photo_crop(page, main.bounding_box, main.mask, **params)
    assert current.metadata["source_quad"] == baseline.metadata["source_quad"]
    np.testing.assert_array_equal(current.rectified_untrimmed, baseline.rectified_untrimmed)
    mode = current.metadata["crop_modes"]["full_print"]
    assert mode["suggested_warp_quad"] == group
    assert mode["suggested_edge_traces"][-1] == [*hint, hint[0]]


@pytest.mark.parametrize("candidate", ["inside", "scallops", "overlapping_corner"])
def test_manual_scalloped_review_protects_backing_without_blocking_an_independent_motif(candidate):
    page, outline, paper, motif = _scan("scalloped")
    w, h = page.shape[1::-1]
    measured = measure_scalloped_print(page, motif)
    assert measured is not None
    outline = segmentation._normalized_polygon(outline, w, h)
    outer = segmentation._normalized_polygon(paper, w, h)
    inner = outer if candidate == "scallops" else segmentation._normalized_polygon(motif, w, h)
    neighbors = []
    if candidate == "overlapping_corner":
        corner = np.float32([[553, 405.5], [563, 405.5], [563, 414], [553, 414]])
        neighbors = [{"polygon": segmentation._normalized_polygon(corner, w, h)}]
    result = correct_photo_crop(
        page, {}, {"source": "manual_quad", "warp_quad": outer, "image_quad": inner},
        dual_mode=True, auto_orient=False, protect_regions=False,
        print_aspect_ratio=510 / 363, manual_scalloped_polygon=outline, neighbor_masks=neighbors,
    )
    full, motif_mode = (result.metadata["crop_modes"][mode] for mode in ("full_print", "image_only"))
    assert full["status"] == "review_needed" and SCALLOPED_PRINT_REVIEW in full["reason_codes"]
    assert result.full_print_preview is not None
    preview = result.full_print_preview
    assert np.any(preview[:, :, 3] == 0) and preview[preview[:, :, 3] > 0, :3].max() < 245
    if candidate != "inside":
        assert motif_mode["status"] == "review_needed" and result.image_only is None
    else:
        assert motif_mode["status"] == "ready" and motif_mode["independent_of_full_print"]
        assert result.image_only is not None and result.image_only.max() < 245
