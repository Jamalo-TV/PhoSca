from dataclasses import replace

import cv2
import numpy as np

from app.pipeline.album_leaf_review import (
    ALBUM_LEAF_COUNT_REVIEW, ALBUM_LEAF_REVIEW_CHILD, ALBUM_LEAF_REVIEW_TRACES, album_leaf_review_traces,
)
from app.pipeline.segmentation import SegmentationDetection, SegmentationResult, _flag_album_leaf_count_review


def _book_scene(binding: tuple[int, int, int] = (31, 42, 54)) -> tuple[np.ndarray, np.ndarray]:
    page = np.full((1300, 1600, 3), 23, dtype=np.uint8)
    page[140:1171, 130:1490] = binding
    page[140:1171, 130:791] = (150, 162, 169)
    page[140:1171, 840:1490] = (150, 162, 169)
    page[300:930, 254:672] = (92, 101, 121)
    page[270:900, 939:1330] = (94, 117, 135)
    outline = np.float32([[130, 140], [790, 140], [790, 1170], [130, 1170]])
    return page, outline


def _detection(points, *, model):
    vertices = np.float32(points) / np.float32([1599, 1299])
    polygon = [{"x": float(x), "y": float(y)} for x, y in vertices]
    box = {f"{axis}{index}": fn(p[axis] for p in polygon)
           for axis in ("x", "y") for index, fn in ((1, min), (2, max))}
    return SegmentationDetection(box, {
        "polygon": polygon,
        "source": "yolo_seg_mask" if model else "classical_border_quad",
        "detector_source": "yolo_onnx" if model else "edge_contour",
        "fusion": {"proposal_count": 1, "proposal_families": ["model" if model else "classical"]},
        "boundary_evidence": {"score": .65, "border_contrast": .75},
    }, confidence=.60 if model else .82, aspect_ratio=.70, geometry_valid=True, review_reasons=[])


def test_album_leaf_gutter_stays_local_under_rotation_and_rescaling():
    page, outline = _book_scene()
    traces = album_leaf_review_traces(page, outline)
    assert len(traces) == 2
    assert np.allclose([rail[:, 0].mean() for rail in traces], [790, 840], atol=5.)
    smaller = cv2.resize(page, None, fx=.5, fy=.5, interpolation=cv2.INTER_AREA)
    rotated = cv2.rotate(smaller, cv2.ROTATE_90_CLOCKWISE)
    points = (outline + .5) * .5 - .5
    rotated_outline = np.float32(np.column_stack((smaller.shape[0] - 1 - points[:, 1], points[:, 0])))
    after = album_leaf_review_traces(rotated, rotated_outline)
    assert len(after) == 2
    assert np.allclose([rail[:, 1].mean() for rail in after], [395, 420], atol=3.)


def test_album_leaf_review_keeps_instances_and_declines_a_claimed_or_unsupported_child():
    page, outline = _book_scene()
    leaf = _detection(outline, model=False)
    print_inside = _detection([[254, 300], [671, 300], [671, 929], [254, 929]], model=True)
    other = _detection([[939, 270], [1329, 270], [1329, 899], [939, 899]], model=True)
    payload = SegmentationResult([leaf, print_inside, other], {"fallback_used": True})
    proposed = _flag_album_leaf_count_review(page, payload)
    marked, left, right = proposed.detections
    assert left == print_inside and right == other
    assert marked.review_reasons == [ALBUM_LEAF_COUNT_REVIEW]
    assert marked.mask[ALBUM_LEAF_REVIEW_CHILD] == print_inside.mask["polygon"]
    fresh_mask = {key: value for key, value in marked.mask.items()
                  if key not in (ALBUM_LEAF_REVIEW_CHILD, ALBUM_LEAF_REVIEW_TRACES)}
    assert replace(marked, mask=fresh_mask, review_reasons=[]) == leaf
    neutral_gap, _ = _book_scene((23, 23, 23))
    assert _flag_album_leaf_count_review(neutral_gap, payload) == payload

    competing = SegmentationResult([leaf, print_inside, replace(print_inside, confidence=.25), other], payload.metadata)
    assert _flag_album_leaf_count_review(page, competing) == competing
    pending = SegmentationResult(payload.detections, {
        "fallback_used": True, "fusion": {"deferred_print_proposals": [{"polygon": print_inside.mask["polygon"]}]},
    })
    assert _flag_album_leaf_count_review(page, pending) == pending
    unsupported = SegmentationResult([leaf, replace(print_inside, confidence=.40), other], payload.metadata)
    assert _flag_album_leaf_count_review(page, unsupported) == unsupported
