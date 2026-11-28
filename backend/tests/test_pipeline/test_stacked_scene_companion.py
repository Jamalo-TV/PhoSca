import cv2
import numpy as np

from app.pipeline.stacked_scene_companion import measure_stacked_scene_companion


def _stacked_print():
    image = np.full((1700, 1750, 3), (170, 194, 221), np.uint8)
    rng = np.random.default_rng(13)
    image[245:1200, 195:1310] = rng.integers(30, 129, (955, 1115, 3), np.uint8)
    image[1240:1375, 190:1325] = rng.integers(35, 163, (135, 1135, 3), np.uint8)
    original = np.float32([[150, 200], [1580, 200], [1580, 1600], [150, 1600]])
    shorter = np.float32([[150, 200], [1580, 200], [1580, 1205], [150, 1205]])
    return image, original, shorter


def test_companion_requires_the_scene_to_end_at_the_first_edge():
    image, original, shorter = _stacked_print()
    outline, traces = measure_stacked_scene_companion(image, original, shorter)

    np.testing.assert_allclose(outline[[0, 3]], shorter[[0, 3]], atol=.05)
    np.testing.assert_allclose(outline[1:3, 1], shorter[1:3, 1], atol=.05)
    assert np.all((1315 < outline[1:3, 0]) & (outline[1:3, 0] < 1322))
    trace, = traces
    assert np.all(np.abs(trace[:, 0] - 1310) < 2.)

    image[1190:1230, 1140:1310] = (52, 66, 81)
    assert measure_stacked_scene_companion(image, original, shorter) is None


def test_companion_preserves_a_mark_or_color_detail_outside_the_main_scene():
    image, original, shorter = _stacked_print()
    marked = image.copy()
    cv2.putText(marked, "1987", (1370, 670), cv2.FONT_HERSHEY_SIMPLEX, 1.25, (90, 120, 140), 2, cv2.LINE_AA)
    assert measure_stacked_scene_companion(marked, original, shorter) is None

    image[600:700, 1440:1490] = (45, 30, 35)
    assert measure_stacked_scene_companion(image, original, shorter) is None
