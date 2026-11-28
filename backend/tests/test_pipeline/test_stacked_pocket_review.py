from dataclasses import replace

import cv2
import numpy as np

from app.pipeline.segmentation import SegmentationDetection, SegmentationResult, _flag_stacked_pocket_count_review
from app.pipeline.stacked_pocket_review import (
    STACKED_POCKET_COUNT_REVIEW, STACKED_POCKET_REVIEW_CHILDREN, STACKED_POCKET_REVIEW_TRACES, stacked_pocket_review_traces,
)


def _source():
    backing = (170, 194, 221)
    image = np.full((1040, 1024, 3), backing, np.uint8)
    rng = np.random.default_rng(42)
    for top, bottom in ((135, 180), (226, 273), (335, 604), (622, 918)):
        patch = cv2.resize(rng.integers(0, 195, (max(5, (bottom - top) // 9), 74, 3), dtype=np.uint8), (880, bottom - top))
        fine = rng.integers(-23, 24, patch.shape[:2], dtype=np.int16)
        image[top:bottom, 65:945] = np.clip(patch.astype(np.int16) + fine[..., None], 0, 255)
    carriers = [
        np.float32([(65, 55), (950, 70), (920, 970), (60, 961)]),
        np.float32([(107, 98), (904, 100), (885, 949), (101, 940)]),
    ]
    photos = [
        np.float32([(155, 335), (855, 335), (852, 604), (152, 604)]),
        np.float32([(155, 622), (855, 622), (854, 918), (153, 918)]),
    ]
    return image, carriers, photos


def test_stacked_pocket_scope_checks_texture_and_stays_outside_the_photo_claims():
    image, carriers, photos = _source()
    spans = stacked_pocket_review_traces(image, carriers, photos)
    assert len(spans) == 3 and all(span[:, 1].max() < 335 for span in spans)
    assert all(np.linalg.norm(span[1] - span[0]) > 400 for span in spans)

    smooth = image.copy()
    smooth[:310] = cv2.GaussianBlur(image[:310], (0, 0), 5.)
    assert stacked_pocket_review_traces(smooth, carriers, photos) == []
    overlapping = [photos[0], photos[1] - [0., 180.]]
    assert stacked_pocket_review_traces(image, carriers, overlapping) == []


def test_stacked_pocket_marks_only_the_two_disputed_carriers():
    image, carriers, photos = _source()
    size = np.float32(image.shape[1::-1]) - 1

    def detection(outline, source, detector, family, confidence):
        polygon = [{"x": float(x), "y": float(y)} for x, y in outline / size]
        return SegmentationDetection({}, {
            "polygon": polygon, "source": source, "detector_source": detector,
            "fusion": {"proposal_count": 1, "proposal_families": [family]},
        }, confidence, 1., True, [])

    rows = [
        detection(carriers[0], "classical_border_quad", "contrast_region", "classical", .59),
        detection(carriers[1], "classical_border_quad", "edge_contour", "classical", .64),
        *(detection(photo, "yolo_seg_mask", "yolo_onnx", "model", .79) for photo in photos),
    ]
    baseline = SegmentationResult(rows, {"fallback_used": True})
    marked = _flag_stacked_pocket_count_review(image, baseline, .70)
    assert marked.metadata == baseline.metadata and marked.detections[2:] == rows[2:]
    for before, after in zip(rows[:2], marked.detections[:2], strict=True):
        evidence = marked.detections[0].mask
        assert after == replace(before, review_reasons=[STACKED_POCKET_COUNT_REVIEW], mask={
            **before.mask,
            STACKED_POCKET_REVIEW_TRACES: evidence[STACKED_POCKET_REVIEW_TRACES],
            STACKED_POCKET_REVIEW_CHILDREN: [row.mask["polygon"] for row in rows[2:]],
        })
    pending = replace(baseline, metadata={**baseline.metadata, "fusion": {"deferred_print_proposals": [{"id": 4}]}})
    assert _flag_stacked_pocket_count_review(image, pending, .70) == pending
