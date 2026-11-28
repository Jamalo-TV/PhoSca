import cv2
import numpy as np

from app.pipeline.print_edges import _PRINT_SIDES
from app.pipeline.visible_scene_edge import measure_stacked_scene_comparison, suggest_visible_scene_edge


def test_scene_edge_requires_a_quiet_field_and_has_no_supported_enclosing_sheet(visible_scene_edge_page):
    image, quad = visible_scene_edge_page
    boundary = {name: {"support": 0., "ambiguous": False} for name in _PRINT_SIDES}
    trace, = suggest_visible_scene_edge(image, quad, boundary, [])
    np.testing.assert_allclose(trace[:, 0], [467.5, 467.5], atol=.5)
    assert abs(trace[1, 1] - trace[0, 1]) > 440
    neighbor = np.float32([[210, 380], [381, 380], [381, 470], [210, 470]])
    assert not suggest_visible_scene_edge(image, quad, boundary, [neighbor])

    sheet = np.full_like(image, (112, 137, 146))
    sheet[126:722, 171:980] = image[126:722, 171:980]
    assert not suggest_visible_scene_edge(sheet, quad, boundary, [])

    inscription = image.copy()
    cv2.putText(inscription, "1987", (205, 355), cv2.FONT_HERSHEY_SIMPLEX, 2, (55, 91, 124), 3)
    assert not suggest_visible_scene_edge(inscription, quad, boundary, [])


def test_stacked_scene_comparison_keeps_separator_and_requires_a_distinct_band():
    backing = (165, 193, 220)
    image = np.full((1700, 1800, 3), backing, np.uint8)
    rng = np.random.default_rng(13)
    image[155:1240, 188:1540] = rng.integers(25, 114, (1085, 1352, 3), np.uint8)
    image[1262:1340, 180:1570] = rng.integers(59, 151, (78, 1390, 3), np.uint8)
    quad = np.float32([[150, 130], [1600, 130], [1600, 1600], [150, 1600]])
    boundary = {name: {"support": .60 if name == "bottom" else .10} for name in _PRINT_SIDES}

    outline, traces = measure_stacked_scene_comparison(image, quad, boundary)
    np.testing.assert_allclose(outline[:2], quad[:2], atol=.01)
    assert np.all((1242 < outline[2:, 1]) & (outline[2:, 1] < 1250))
    assert traces and all(np.all(np.abs(trace[:, 1] - 1240.) <= 2.) for trace in traces)

    blank_separator = image.copy()
    blank_separator[1262:1340] = backing
    assert measure_stacked_scene_comparison(blank_separator, quad, boundary) is None

    two_transitions = image.copy()
    two_transitions[1150:1172] = backing
    assert measure_stacked_scene_comparison(two_transitions, quad, boundary) is None
