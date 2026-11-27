import cv2
import numpy as np
import pytest

from app.pipeline.faint_paper_review import FAINT_PAPER_WITNESS_MARGIN, _is_repeated_ridge, adjacent_faint_paper_steps
from app.pipeline.geometry import order_quad_cyclic
from app.pipeline.near_capture_review import quarter_turn
from app.pipeline.perspective import review_traces_intersect_claim


def _ink_and_exterior(*, left=True, bottom=True, same_polarity=False):
    image = np.full((660, 730, 3), 211, np.uint8)
    image[150:501, 200:521] = (95, 113, 140)
    cv2.circle(image, (348, 295), 79, (119, 105, 95), -1)
    if left:
        image[155:517, 181:184] = 221
    if bottom:
        image[514:517, 184:527] = 221 if same_polarity else 201
    quad = np.float32([[200, 132], [540, 132], [540, 500], [200, 500]])
    return cv2.GaussianBlur(image, (3, 3), .58), quad


@pytest.mark.parametrize("turn", range(4))
def test_adjacent_faint_guides_stay_outside_the_stronger_ink_lines(turn):
    image, quad = _ink_and_exterior()
    scale = np.float32(image.shape[1::-1]) - 1
    rotated = np.rot90(image, turn)
    rotated_scale = np.float32(rotated.shape[1::-1]) - 1
    source = order_quad_cyclic(quarter_turn(quad / scale, turn) * rotated_scale)
    traces = adjacent_faint_paper_steps(rotated, source)

    assert len(traces) == 2
    bottom, left = [quarter_turn(trace / rotated_scale, (-turn) % 4) * scale for trace in traces]
    np.testing.assert_allclose(left[:, 0], 182, atol=1.6)
    np.testing.assert_allclose(bottom[:, 1], 515, atol=1.6)
    assert np.ptp(left[:, 1]) > 270 and np.ptp(bottom[:, 0]) > 230


@pytest.mark.parametrize("alteration", [{"left": False}, {"bottom": False}, {"same_polarity": True}])
def test_one_line_or_matching_ridges_cannot_supply_a_paired_guide(alteration):
    image, quad = _ink_and_exterior(**alteration)
    assert adjacent_faint_paper_steps(image, quad) == []


def test_claims_in_faint_measurement_collars_count_even_if_the_center_is_clear():
    width, height = 201, 201

    def point(x, y):
        return {"x": x / 200, "y": y / 200}

    traces = [[point(40, 100), point(160, 100)]]
    claim = {"polygon": [point(55, 105.5), point(110, 105.5), point(110, 107.), point(55, 107.)]}
    assert not review_traces_intersect_claim(traces, claim, width, height)
    assert review_traces_intersect_claim(traces, claim, width, height, margin_px=FAINT_PAPER_WITNESS_MARGIN)


def test_periodic_album_ridges_have_matching_peaks_on_both_sides():
    image = np.full((290, 400, 3), 215, np.uint8)
    line = np.float32([[330, 150], [70, 150]])
    distance = np.float32([26, 26])
    image[149:152] = 203
    assert not _is_repeated_ridge(image, line, distance, -1)
    image[133:136] = image[165:168] = 203
    assert _is_repeated_ridge(image, line, distance, -1)
