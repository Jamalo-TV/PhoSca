import cv2
import numpy as np
import pytest

from app.pipeline.geometry import order_quad_cyclic
from app.pipeline.open_paper_edge import measure_open_paper_edge


def _rectangle(left, top, right, bottom):
    return np.float32([[left, top], [right, top], [right, bottom], [left, bottom]])


@pytest.fixture
def stock():
    backing, paper = (116, 98, 81), (219, 226, 235)
    image = np.full((1100, 1000, 3), backing, np.uint8)
    image[100:981, 100:874] = paper
    image[167:930, 172:815] = (119, 73, 70)
    image[390:700, 70:100] = paper
    held = _rectangle(123, 100, 873, 980)
    earlier = {"side": "top", "companion": "right"}
    return image, held, earlier, [_rectangle(15, 150, 75, 924)]


@pytest.mark.parametrize("turned", [False, True])
def test_open_paper_edge_displays_only_measured_segments(stock, turned):
    image, held, earlier, claims = stock
    if turned:
        source_height = image.shape[0]

        def rotate(points):
            return np.column_stack((source_height - 1 - points[:, 1], points[:, 0]))

        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        held = order_quad_cyclic(rotate(held))
        claims = [rotate(claim) for claim in claims]
        earlier = {"side": "right", "companion": "bottom"}
    traces, anchor = measure_open_paper_edge(image, held, earlier, claims)
    expected = [np.float32([[100, 122], [100, 326]]), np.float32([[100, 718], [100, 918]])]
    for measured, exact in zip(traces, expected, strict=True):
        np.testing.assert_allclose(measured, rotate(exact) if turned else exact, atol=.05)
    expected_anchor = np.float32([[92, 100], [108, 100]])
    np.testing.assert_allclose(anchor, rotate(expected_anchor) if turned else expected_anchor, atol=.05)


@pytest.mark.parametrize("case", ["missing_endpoint", "rival_through_gap", "claimed_segment", "claimed_anchor", "claim_in_gap"])
def test_open_paper_edge_checks_visible_witnesses_and_leaves_the_gap_open(stock, case):
    image, held, earlier, claims = stock
    if case == "missing_endpoint":
        image[209:354, 70:100] = (219, 226, 235)
    elif case == "rival_through_gap":
        image[410:671, 82:99] = (116, 98, 81)
    elif case == "claimed_segment":
        claims.append(_rectangle(83, 738, 104, 760))
    elif case == "claimed_anchor":
        claims.append(_rectangle(80, 72, 106, 120))
    else:
        claims.append(_rectangle(80, 469, 104, 553))
    result = measure_open_paper_edge(image, held, earlier, claims)
    if case == "claim_in_gap":
        assert [tuple(trace[:, 1]) for trace in result[0]] == [(122, 326), (718, 918)]
    else:
        assert result is None
