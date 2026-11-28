import cv2
import numpy as np

from app.pipeline import segmentation
from app.pipeline.contested_scene_edge import contested_scene_edge_traces
from app.pipeline.crossing_prints import CONTESTED_TILE_OUTLINE


def test_scene_edge_requires_agreeing_spans_clear_of_current_claims():
    width, height = 620, 420
    lab = np.full((height, width, 3), (210, 124, 118), np.uint8)
    lab[:130] = (191, 128, 152)
    lab[116:119, :, 0] = 222  # A competing lightness edge in the warm backing.
    source = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    scale = [width - 1, height - 1]
    first = np.float32([[110, 120], [430, 121], [431, 320], [110, 320]]) / scale
    second = np.float32([[108, 123], [432, 122], [431, 322], [109, 322]]) / scale

    trace, = contested_scene_edge_traces(source, [first, second], [])
    assert np.all(np.abs(trace[:, 1] - 129.5) < 1.25)
    assert trace[1, 0] - trace[0, 0] > 200
    assert trace[0, 0] > 110 and trace[1, 0] < 430

    near = np.float32([[210, 120], [275, 120], [275, 127], [210, 127]]) / scale
    claim = {"source_mask": {"warp_quad": [{"x": x, "y": y} for x, y in near]}}
    assert not contested_scene_edge_traces(source, [first, second], [claim])

    lab[:130, :, 2] = 118
    neutral = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    assert not contested_scene_edge_traces(neutral, [first, second], [])
    assert not contested_scene_edge_traces(source, [first, second / 2], [])


def test_failed_local_scene_view_preserves_the_existing_detections(monkeypatch):
    def rect(left, top, right, bottom):
        return [{"x": x, "y": y} for x, y in ((left, top), (right, top), (right, bottom), (left, bottom))]

    owner = segmentation.SegmentationDetection({}, {
        "polygon": rect(.40, .40, .65, .72), "crossing_print_suggestion": rect(.45, .52, .65, .72),
        "crossing_print_suggestion_source": CONTESTED_TILE_OUTLINE,
    }, .81, 1., True, [])
    scout = segmentation.SegmentationDetection({}, {"polygon": rect(.10, .10, .45, .50)}, .66, 1., True, [])
    search = segmentation._TileReviewSearch(lambda _: [], [], None, [scout])
    result = segmentation.SegmentationResult([owner], {"fusion": {}})
    raster = np.zeros((420, 620, 3), np.uint8)
    attempts = []

    def unavailable(*args, **kwargs):
        attempts.append(kwargs["window"])
        raise RuntimeError("local model unavailable")

    monkeypatch.setattr(segmentation, "_centered_tile_print_mask", unavailable)
    assert segmentation._offer_contested_scene_edge(raster, raster, result, search, .70) is result
    assert attempts == [(.56, .60)]
