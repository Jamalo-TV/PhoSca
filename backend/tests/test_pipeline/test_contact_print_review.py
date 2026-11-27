import cv2
import numpy as np
import pytest

from app.pipeline.contact_print_review import _chroma_line, _remeasure_chroma_line, contact_print_review_outline
from app.pipeline.geometry import order_quad_cyclic


def _contact_grid(*, positive: bool, split: bool, periodic: bool):
    backing = (105, 150, 180)
    page = np.full((1130, 910, 3), backing, dtype=np.uint8)
    page[75:1076, 110:790] = (217, 221, 227)
    for row in range(7):
        upper = 164 + 130 * row
        lower = upper + 97
        body, rail = (127, 34) if positive else (23, 105)
        page[upper:lower + 7, 165:735] = body
        for position in (upper, lower):
            page[position - 4:position + 6, 165:735] = rail
            if periodic:
                for x in range(170, 730, 15):
                    page[position - 4:position + 6, x:x + 6] = 231
    if split:
        page[535:539, 110:790] = backing
    expected = np.float32([[110, 75], [789, 75], [789, 1075], [110, 1075]])
    seed = np.float32([[154, 120], [743, 120], [743, 1043], [154, 1043]])
    return page, seed, expected


@pytest.mark.parametrize("rotation", [0, 3])
def test_source_outline_recovers_the_extent_of_a_printed_contact_grid(rotation):
    page, seed, expected = _contact_grid(positive=True, split=False, periodic=True)
    for _ in range(rotation):
        width = page.shape[1]
        seed, expected = [order_quad_cyclic(np.column_stack((quad[:, 1], width - 1 - quad[:, 0])))
                          for quad in (seed, expected)]
        page = cv2.rotate(page, cv2.ROTATE_90_COUNTERCLOCKWISE)
    outline = contact_print_review_outline(page, seed, seed)
    assert outline is not None
    np.testing.assert_allclose(outline, expected, atol=1)


@pytest.mark.parametrize("positive,split,periodic", [(False, False, True), (True, True, True), (True, False, False)])
def test_source_outline_declines_negative_density_broken_stock_and_missing_rails(positive, split, periodic):
    page, seed, _ = _contact_grid(positive=positive, split=split, periodic=periodic)
    assert contact_print_review_outline(page, seed, seed) is None


@pytest.mark.parametrize(("ink_x", "seed_offset", "accepted"), [(119, 0., True), (118, 0., False), (121, 2.5, False)])
def test_remeasured_chroma_side_needs_three_close_native_fits(ink_x, seed_offset, accepted):
    page, _, expected = _contact_grid(positive=True, split=False, periodic=True)
    page[260:480, ink_x:ink_x + 6] = (105, 150, 180)
    start, end = expected[[3, 0]] + np.float32([seed_offset, 0])
    assert _chroma_line(page, start, end) is None

    line = _remeasure_chroma_line(page, start, end)

    if accepted:
        assert line is not None
        np.testing.assert_allclose(line, [1., 0., -109.5], atol=.05)
    else:
        assert line is None


def test_remeasured_chroma_side_cannot_narrow_the_original_outline():
    page, _, expected = _contact_grid(positive=True, split=False, periodic=True)
    inward = np.float32([1, 0])
    start, end = expected[[3, 0]] - 1.5 * inward
    assert all(_chroma_line(page, start - d * inward, end - d * inward) is not None for d in (1, 2, 3))

    assert _remeasure_chroma_line(page, start, end) is None
