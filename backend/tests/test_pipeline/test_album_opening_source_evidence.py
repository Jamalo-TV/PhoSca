import cv2
import numpy as np
import pytest

from app.pipeline.album_opening_review import album_opening_traces


def _scene(*, bow=30):
    page = np.full((840, 800, 3), (167, 170, 177), np.uint8)
    x = np.linspace(205, 630, 128)
    y = 125 + bow * ((x - 418) / 213) ** 2
    paper = np.float32(np.vstack((np.column_stack((x, y)), [630, 706], [172, 706])))
    cv2.fillPoly(page, [np.int32(np.rint(paper))], (46, 46, 47))
    proposal = np.float32([[99, 127], [99, 696], [190, 706], [619, 704], [630, 540], [630, 127]])
    sides = ("top", "right", "bottom", "left")
    boundary = {side: {"support": score} for side, score in zip(sides, (.37, .98, .98, .24), strict=True)}
    return page, paper, proposal, boundary


@pytest.mark.parametrize("turns", [0, 1, 2, 3])
def test_trace_follows_only_the_measured_opening_transitions(turns):
    page, _, proposal, boundary = _scene()
    sides = tuple(boundary)
    for _ in range(turns):
        proposal = np.column_stack((page.shape[0] - 1 - proposal[:, 1], proposal[:, 0]))
        page = cv2.rotate(page, cv2.ROTATE_90_CLOCKWISE)
        boundary = {side: boundary[sides[index - 1]] for index, side in enumerate(sides)}

    top, left = album_opening_traces(page, proposal, boundary)
    for _ in range(turns):
        top, left = [np.column_stack((trace[:, 1], page.shape[1] - 1 - trace[:, 0])) for trace in (top, left)]
        page = cv2.rotate(page, cv2.ROTATE_90_COUNTERCLOCKWISE)

    assert top.shape[0] >= 4 and left.shape == (2, 2)
    assert np.allclose(top[[0, -1]], [[205, 155], [630, 155]], atol=4)
    assert abs(top[:, 1].min() - 125) <= 4
    assert np.allclose(left, [[200, 208], [176, 625]], atol=4)


def test_requires_a_broad_apron_a_bow_and_supported_opposite_sides():
    page, paper, proposal, boundary = _scene()
    assert album_opening_traces(page, paper, boundary) == []
    flat, *_ = _scene(bow=0)
    assert album_opening_traces(flat, proposal, boundary) == []
    assert album_opening_traces(page, proposal, {**boundary, "right": {"support": .85}}) == []
