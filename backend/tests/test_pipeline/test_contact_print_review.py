import cv2
import numpy as np
import pytest

from app.pipeline.contact_print_review import contact_print_review_outline
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
