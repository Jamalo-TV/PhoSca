import cv2
import numpy as np
import pytest

from app.pipeline.near_capture_review import near_capture_print_traces, quarter_turn


def _near_frame_scene():
    height, width = 600, 660
    quad = np.float32([[36, 157], [367, 114], [347, 425], [1, 473]])
    y, x = np.indices((height, width))
    lab = np.full((height, width, 3), [202, 129, 144], dtype=np.uint8)
    lab[..., 0] = 202 + 39 * np.cos(2 * np.pi * (y + 1) / 8)
    inside = np.zeros((height, width), dtype=np.uint8)
    cv2.fillConvexPoly(inside, quad.astype(np.int32), 1)
    photo = 110 + 16 * np.sin(.117 * (x + y)) - 7 * np.sin(.44 * x + .062 * y)
    lab[inside > 0, 0] = photo[inside > 0]
    lab[inside > 0, 1:] = [136, 121]
    seeds = [
        quad + np.float32([[1, 1], [-1, 0], [1, -1], [-1, 0]]),
        quad + np.float32([[0, -1], [0, 1], [-1, 0], [0, 1]]),
    ]
    return lab, inside, seeds


@pytest.mark.parametrize("turns", range(4))
def test_near_frame_guide_only_contains_native_open_segments(turns):
    lab, _, seeds = _near_frame_scene()
    scale = np.float32(lab.shape[1::-1]) - 1
    page = np.ascontiguousarray(np.rot90(cv2.cvtColor(lab, cv2.COLOR_LAB2BGR), turns))
    polygons = [quarter_turn(seed / scale, turns) for seed in seeds]

    spans = near_capture_print_traces(page, polygons, [])

    assert len(spans) == 4
    source_spans = [quarter_turn(segment, (-turns) % 4) * scale for segment in spans]
    assert all(segment.shape == (2, 2) for segment in source_spans)
    assert min(segment[:, 0].min() for segment in source_spans) > 2
    assert all(np.linalg.norm(segment[1] - segment[0]) > 150 for segment in source_spans)


@pytest.mark.parametrize("variant", ["clipped", "flat", "claimed", "conflicting_views"])
def test_near_frame_guide_rejects_missing_backing_or_conflicting_ownership(variant):
    lab, inside, seeds = _near_frame_scene()
    claims = []
    if variant == "clipped":
        lab = lab[:, 6:].copy()
    elif variant == "flat":
        lab[inside == 0, 0] = 202
    elif variant == "conflicting_views":
        seeds[1] += np.float32([40, 0])
    scale = np.float32(lab.shape[1::-1]) - 1
    if variant == "clipped":
        seeds = [np.clip(seed - np.float32([6, 0]), 0, scale) for seed in seeds]
    if variant == "claimed":
        claims = [{"polygon": [{"x": float(x), "y": float(y)} for x, y in seeds[0] / scale]}]

    image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    assert near_capture_print_traces(image, [seed / scale for seed in seeds], claims) == []
