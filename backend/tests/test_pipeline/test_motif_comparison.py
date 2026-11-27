import cv2
import numpy as np
import pytest

from app.pipeline.geometry import order_quad_cyclic
from app.pipeline.motif_comparison import measured_motif_comparison
from app.pipeline.near_capture_review import quarter_turn


def _photographed_print(*, with_mark=False):
    motif = np.float32([[150, 148], [397, 106], [379, 385], [162, 451]])
    paper = np.int32([[137, 136], [406, 95], [390, 398], [149, 466]])
    proposal = np.float32([[137, 136], [404, 101], [451, 439], [154, 471]])
    image = np.full((620, 690, 3), (222, 225, 226), np.uint8)
    cv2.fillConvexPoly(image, paper, (225, 229, 230))
    cv2.fillConvexPoly(image, motif.astype(np.int32), (92, 104, 120))
    cv2.circle(image, (292, 237), 31, (150, 109, 96), -1)
    if with_mark:
        cv2.putText(image, "86", (260, 422), cv2.FONT_HERSHEY_SIMPLEX, .62, (216, 219, 220), 1, cv2.LINE_AA)
    return image, motif, proposal


@pytest.mark.parametrize("turn", range(4))
def test_measured_motif_comparison_covers_scene_at_a_different_slant(turn):
    image, motif, proposal = _photographed_print()
    scale = np.float32(image.shape[1::-1]) - 1
    rotated = np.rot90(image, turn)
    rotated_scale = np.float32(rotated.shape[1::-1]) - 1
    polygon = quarter_turn(proposal / scale, turn)
    measured = measured_motif_comparison(rotated, polygon, order_quad_cyclic(polygon * rotated_scale))

    assert measured is not None and len(measured.motif_traces) >= 4
    outline = quarter_turn(measured.outline / rotated_scale, (-turn) % 4) * scale
    assert all(cv2.pointPolygonTest(outline, tuple(corner), True) >= 0 for corner in motif)
    assert np.linalg.norm(order_quad_cyclic(outline) - motif, axis=1).max() <= 5
    assert cv2.pointPolygonTest(outline, tuple(proposal[2]), True) < -50


def test_motif_comparison_abstains_for_small_pale_margin_marks():
    image, _, proposal = _photographed_print(with_mark=True)
    assert measured_motif_comparison(image, proposal / (np.float32(image.shape[1::-1]) - 1), proposal) is None
