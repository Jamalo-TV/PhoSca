import cv2
import numpy as np
import pytest

from app.pipeline.geometry import order_quad_cyclic
from app.pipeline.near_capture_review import quarter_turn
from app.pipeline.neighboring_print_review import neighboring_layout_traces, neighboring_print_outline, next_print_position
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


def _layout_scene():
    height, width = 1040, 1200
    scale = np.float32([width - 1, height - 1])
    y, x = np.indices((height, width))
    ribs = 38 * np.cos(2 * np.pi * y / 11)
    lab = np.empty((height, width, 3), dtype=np.uint8)
    lab[:] = [205, 129, 143]
    lab[..., 0] = 205 + ribs
    footprints = [_rect(400, 220, 640, 475), _rect(675, 222, 915, 477),
                  _rect(403, 515, 643, 770), _rect(72, 218, 314, 455), _rect(678, 517, 918, 772)]
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [quad.astype(np.int32) for quad in footprints], 1)
    art = 91 + 9 * np.sin(.25 * y + .33 * x) + 7 * np.sin(.44 * x - .11 * y)
    lab[mask > 0, 0] = art[mask > 0]
    lab[mask > 0, 1:] = [143, 124]
    return lab, scale, ribs, footprints


@pytest.mark.parametrize("turns", range(4))
def test_vacant_layout_returns_only_two_observed_open_sides(turns):
    lab, scale, _, footprints = _layout_scene()
    image = np.ascontiguousarray(np.rot90(cv2.cvtColor(lab, cv2.COLOR_LAB2BGR), turns))
    anchor, *held, _ = [quarter_turn(quad / scale, turns) for quad in footprints]
    pending = [{"reason": TILED_PRINT_PROPOSAL, "polygon": _outline(quad)} for quad in held]
    traces = neighboring_layout_traces(image, [anchor], pending, [{"polygon": _outline(anchor)}, *pending])

    assert len(traces) == 2
    restored = [quarter_turn(trace / (np.float32(image.shape[1::-1]) - 1), (-turns) % 4) * scale
                for trace in traces]
    for trace, x in zip(sorted(restored, key=lambda span: span.mean(axis=0)[0]), [678, 918], strict=True):
        assert abs(trace[:, 0] - x).max() <= 1
        assert np.all((535 < trace[:, 1]) & (trace[:, 1] < 755))
        assert np.linalg.norm(trace[1] - trace[0]) > 180


@pytest.mark.parametrize("case", ["covered_gap", "no_ribs", "ribs_continue", "extra_held"])
def test_vacant_layout_suppresses_source_or_count_conflicts(case):
    lab, scale, ribs, footprints = _layout_scene()
    anchor, *held, _ = [quad / scale for quad in footprints]
    pending = [{"reason": TILED_PRINT_PROPOSAL, "polygon": _outline(quad)} for quad in held]
    claims = [{"polygon": _outline(anchor)}, *pending]
    if case == "covered_gap":
        claims.append({"visible_polygon": _outline(_rect(660, 600, 670, 645) / scale)})
    elif case == "no_ribs":
        lab[520:768, 920:955] = [205, 129, 143]
    elif case == "ribs_continue":
        lab[518:772, 679:918, 0] = 91 + ribs[518:772, 679:918]
    elif case == "extra_held":
        pending.append({"reason": TILED_PRINT_PROPOSAL, "polygon": _outline(_rect(900, 20, 1150, 250) / scale)})
    image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    assert neighboring_layout_traces(image, [anchor], pending, claims) == []


@pytest.mark.parametrize("bounds", [
    (367, 478, 957, 808),
    (646, 488, 954, 810),
    (646, 180, 954, 810),
])
def test_vacant_layout_suppresses_a_visible_carrier_around_cell_or_either_neighbor(bounds):
    lab, scale, _, footprints = _layout_scene()
    anchor, *held, _ = [quad / scale for quad in footprints]
    pending = [{"reason": TILED_PRINT_PROPOSAL, "polygon": _outline(quad)} for quad in held]
    claims = [{"polygon": _outline(anchor)}, *pending]
    left, top, right, bottom = bounds
    sheet = lab[top:bottom, left:right]
    inner = sheet[11:-11, 11:-11].copy()
    sheet[:] = [80, 125, 124]
    sheet[11:-11, 11:-11] = inner
    image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    assert neighboring_layout_traces(image, [anchor], pending, claims) == []
