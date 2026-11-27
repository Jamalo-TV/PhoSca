import cv2
import numpy as np
import pytest

from app.pipeline.geometry import order_quad_cyclic
from app.pipeline.near_capture_review import quarter_turn
from app.pipeline.neighboring_print_review import neighboring_print_outline, next_print_position
from app.pipeline.tile_print_review import TILED_PRINT_PROPOSAL


def _outline(points):
    return [{"x": float(x), "y": float(y)} for x, y in points]


def _rect(left, top, right, bottom):
    return np.float32([[left, top], [right, top], [right, bottom], [left, bottom]])


@pytest.mark.parametrize("case", ["vacant", "claimed", "wrong_pitch", "two_positions"])
def test_neighboring_search_requires_one_vacant_position_from_measured_neighbors(case):
    width, height = 1040, 1000
    scale = np.float32([width - 1, height - 1])
    a = _rect(480, 455, 725, 685)
    b = _rect(480, 705, 725, 935)
    if case == "wrong_pitch":
        b += [80, 10]
    elif case == "two_positions":
        a -= [0, 200]
        b -= [0, 200]
    pending = [{"reason": TILED_PRINT_PROPOSAL, "polygon": _outline(quad / scale)} for quad in (a, b)]
    claims = pending.copy()
    if case == "claimed":
        claims.append({"polygon": _outline(_rect(500, 230, 610, 365) / scale)})

    position = next_print_position(pending, claims, width, height)

    if case == "vacant":
        projected, pitch = position
        np.testing.assert_allclose(pitch, [0, -250], atol=.01)
        np.testing.assert_allclose(projected, a + pitch, atol=.01)
    else:
        assert position is None


def _source_scene():
    height, width = 600, 750
    quad = np.float32([[182, 131], [470, 116], [477, 392], [188, 413]])
    y, x = np.indices((height, width))
    lab = np.full((height, width, 3), [199, 129, 144], dtype=np.uint8)
    ribs = 39 * np.cos(2 * np.pi * y / 13)
    lab[..., 0] = 199 + ribs
    inside = np.zeros((height, width), dtype=np.uint8)
    cv2.fillConvexPoly(inside, quad.astype(np.int32), 1)
    art = 97 + 6 * np.sin(.68 * y + .081 * x) + 8 * np.cos(.29 * x - .15 * y)
    lab[inside > 0, 0] = art[inside > 0]
    lab[inside > 0, 1:] = [135, 124]
    scale = np.float32([width - 1, height - 1])
    masks = [(quad + np.float32(shift)) / scale for shift in (
        [[1, -1], [1, 0], [-1, 0], [0, -1]], [[-1, 0], [0, 1], [0, -1], [1, 0]],
    )]
    return lab, ribs, inside, masks, quad


@pytest.mark.parametrize("turns", range(4))
def test_neighboring_outline_matches_original_edges_across_rotations(turns):
    lab, _, _, masks, quad = _source_scene()
    dims = np.float32(lab.shape[1::-1]) - 1
    page = np.ascontiguousarray(np.rot90(cv2.cvtColor(lab, cv2.COLOR_LAB2BGR), turns))
    pitch = np.float32(([0, -250], [-250, 0], [0, 250], [250, 0])[turns])
    rotated = [quarter_turn(polygon, turns) for polygon in masks]

    outline = neighboring_print_outline(page, rotated, pitch, [])

    assert outline is not None
    restored = quarter_turn(outline / (np.float32(page.shape[1::-1]) - 1), (-turns) % 4) * dims
    np.testing.assert_allclose(order_quad_cyclic(restored), quad, atol=1.4)


@pytest.mark.parametrize("case", ["flat_backing", "ribs_continue", "claimed_witness", "enclosing_sheet", "inconsistent_masks"])
def test_neighboring_outline_abstains_when_native_witness_conflicts(case):
    lab, ribs, inside, masks, _ = _source_scene()
    dims = np.float32(lab.shape[1::-1]) - 1
    claims = []
    if case == "flat_backing":
        lab[inside == 0, 0] = 199
    elif case == "ribs_continue":
        lab[inside > 0, 0] = 97 + ribs[inside > 0]
    elif case == "claimed_witness":
        claims = [{"polygon": _outline(_rect(479, 218, 499, 280) / dims)}]
    elif case == "enclosing_sheet":
        inner = lab[73:510, 103:621].copy()
        lab[46:540, 72:649] = [80, 125, 125]
        lab[73:510, 103:621] = inner
    elif case == "inconsistent_masks":
        masks[1] += np.float32([35, 0]) / dims

    page = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    assert neighboring_print_outline(page, masks, np.float32([0, -250]), claims) is None
