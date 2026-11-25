from pathlib import Path

import cv2
import numpy as np
import pytest

from app.pipeline.edge_trace import measure_edge_trace
from app.pipeline.image_ops import load_image


def _measure_in_pixels(image, points):
    height, width = image.shape[:2]
    size = np.array([width - 1, height - 1])
    return [
        np.array([(point["x"], point["y"]) for point in trace]) * size
        for trace in measure_edge_trace(image, np.asarray(points) / size)
    ]


def test_real_flank_measurements_leave_gaps_open():
    source = Path(__file__).resolve().parents[3] / "data" / "golden_fixtures" / "page_04.jpg"
    image = load_image(source)

    left, = _measure_in_pixels(image, [(139, 1842), (110, 1296), (70, 750)])
    assert np.ptp(left[:, 1]) > 1000
    chord_x = np.interp(left[:, 1], [750, 1842], [70, 139])
    assert np.max(left[:, 0] - chord_x) > 4

    outer = _measure_in_pixels(image, [(2352, 735), (2317, 1266), (2296, 1797)])
    assert np.vstack(outer)[:, 1].min() < 780
    assert np.vstack(outer)[:, 1].max() > 1750
    assert all(trace[:, 1].max() < 1120 or trace[:, 1].min() > 1340 for trace in outer)
    assert not _measure_in_pixels(image, [(2352, 735), (2324, 1266), (2296, 1797)])

    top, = _measure_in_pixels(image, [(1617, 728), (1970, 731), (2350, 736)])
    assert np.ptp(top[:, 0]) > 670
    lower_right, lower_left = _measure_in_pixels(image, [(2298, 1798), (1950, 1807), (1596, 1814)])
    assert lower_right[:, 0].min() > 1855
    assert lower_left[:, 0].max() < 1695
    assert not _measure_in_pixels(image, [(1596, 1816), (1604, 1273), (1620, 730)])


def _edge_control(surface):
    y, x = np.indices((550, 700), dtype=np.float32)
    start_y, end_y, x_at_ends = 50, 500, 250
    t = np.clip((y - start_y) / (end_y - start_y), 0, 1)
    edge = x_at_ends + 20 * t * (1 - t)
    image = np.full_like(x, 220)
    if surface == "partly_covered":
        image[(x >= edge) & ((y < 240) | (y > 310))] = 75
    elif surface == "half_covered":
        image[(x >= edge) & (y > 275)] = 75
    elif surface == "thin_ink":
        image[(x >= edge) & (x < edge + 6)] = 65
    elif surface == "parallel_edges":
        image[x >= edge - 1.5] -= 48
        image[x >= edge + 2] -= 48
    elif surface == "glare":
        image += 30 * np.exp(-.5 * ((x - edge) / 2.6) ** 2)
    image = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    return cv2.GaussianBlur(image, (0, 0), .6), [(250, 50), (255, 275), (250, 500)]


@pytest.mark.parametrize("angle", [0, 17, 90])
@pytest.mark.parametrize("reverse", [False, True])
def test_traces_follow_visible_original_pixels_at_either_orientation(angle, reverse):
    image, points = _edge_control("partly_covered")
    transform = cv2.getRotationMatrix2D((350, 275), angle, 1)
    source = cv2.warpAffine(image, transform, (700, 550), borderMode=cv2.BORDER_REPLICATE)
    anchors = np.asarray(points) @ transform[:, :2].T + transform[:, 2]
    if reverse:
        anchors = anchors[::-1]
    traces = _measure_in_pixels(source, anchors)
    inverse = cv2.invertAffineTransform(transform)
    spans = sorted(
        (trace @ inverse[:, :2].T + inverse[:, 2] for trace in traces),
        key=lambda trace: trace[:, 1].min(),
    )
    upper, lower = spans
    assert upper[:, 1].max() < 241 and lower[:, 1].min() > 309
    assert np.ptp(upper[:, 1]) > 150 and np.ptp(lower[:, 1]) > 150
    samples = np.vstack(spans)
    t = (samples[:, 1] - 50) / 450
    assert np.max(np.abs(samples[:, 0] - (250 + 20 * t * (1 - t)))) < 1


@pytest.mark.parametrize("surface", ["blank", "half_covered", "thin_ink", "parallel_edges", "glare"])
def test_limited_or_competing_source_evidence_does_not_draw_a_trace(surface):
    image, points = _edge_control(surface)
    assert not _measure_in_pixels(image, points)
