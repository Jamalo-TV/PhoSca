from pathlib import Path

import cv2
import numpy as np
import pytest

from app.pipeline.geometry import order_quad_cyclic
from app.pipeline.image_ops import load_image
from app.pipeline.inward_scene_edge import measure_inward_scene_edge
from app.pipeline.print_edges import refine_print_boundary


@pytest.mark.parametrize("case", ["native", "quarter_turn", "outside_warp_claim", "missing_right_arm"])
def test_dark_print_hint_uses_only_the_clear_end_of_the_measured_edge(case):
    source = Path(__file__).resolve().parents[3] / "data/golden_fixtures/page_10.jpg"
    image = load_image(source)
    # The stored detector warp starts the search. The development annotation is excluded.
    quad = np.float32([[1296.09, 1137.17], [2300.03, 1129.04], [2310, 1856], [1298.08, 1856]])
    expected = np.float32([[2229.05, 1809.96], [1379.03, 1832.45]])
    left = np.float32([[1100, 1700], [1324, 1700], [1324, 1900], [1100, 1900]])
    peers = [left]
    if case == "outside_warp_claim":
        claim = np.float32([[2320, 1797], [2368, 1797], [2368, 1850], [2320, 1850]])
        assert cv2.intersectConvexConvex(quad, claim)[0] == 0
        peers.append(claim)
    if case == "missing_right_arm":
        image[1724:1828, 2286:2340] = (205, 209, 215)
    _, boundary, _ = refine_print_boundary(image, quad)
    if case == "quarter_turn":
        width = image.shape[1]

        def rotate(points):
            return np.column_stack((points[:, 1], width - 1 - points[:, 0]))

        image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        quad, expected, peers = order_quad_cyclic(rotate(quad)), rotate(expected), [rotate(left)]
        _, boundary, _ = refine_print_boundary(image, quad)

    guide = measure_inward_scene_edge(image, quad, boundary, peers)

    if case in {"native", "quarter_turn"}:
        assert guide is not None
        np.testing.assert_allclose(guide.trace, expected, atol=.8)
        assert cv2.intersectConvexConvex(guide.scope, peers[0])[0] == 0
    else:
        assert guide is None
