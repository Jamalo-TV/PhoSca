import cv2
import numpy as np
import pytest

from app.pipeline.tile_print_review import measured_adjacent_tile_outline, tiled_outline_conflicts


def _scene():
    height, width = 600, 696
    y, x = np.indices((height, width))
    lab = np.full((height, width, 3), [202, 129, 144], dtype=np.uint8)
    lab[..., 0] = 202 + 38 * np.cos(2 * np.pi * (y + 1) / 8)
    photo = 117 + 18 * np.sin(.114 * (x + y)) - 7 * np.sin(.43 * x + .071 * y)
    photo[y > 426] += 23
    lab[153:456, 160:518, 0] = photo[153:456, 160:518]
    lab[153:456, 160:518, 1:] = [136, 121]
    # A strong rib outside the photo can attract an unsigned line fit.
    lab[467:469, 164:516, 0] = 245
    lab[469:471, 164:516, 0] = 151
    scale = np.float32([width - 1, height - 1])
    observations = [
        (np.float32([[159, 153], [516, 153], [518, 454], [159, 452]]) / scale, .87),
        (np.float32([[160, 153], [518, 154], [519, 469], [160, 469]]) / scale, .62),
    ]
    return lab, observations, scale


def _quarter_turn(points, turns):
    x, y = points.T
    return np.column_stack(((x, y, 1 - x, 1 - y)[turns], (y, 1 - x, 1 - y, x)[turns]))


@pytest.mark.parametrize("turns", range(4))
def test_adjacent_tile_outline_replaces_a_backing_rib_with_the_observed_print_edge(turns):
    lab, observations, _ = _scene()
    image = np.ascontiguousarray(np.rot90(cv2.cvtColor(lab, cv2.COLOR_LAB2BGR), turns))
    rotated = [(_quarter_turn(polygon, turns), score) for polygon, score in reversed(observations)]

    outline = measured_adjacent_tile_outline(image, rotated, [])

    assert outline is not None
    canvas = np.float32([image.shape[1] - 1, image.shape[0] - 1])
    unrotated = _quarter_turn(outline / canvas, (-turns) % 4) * [lab.shape[1] - 1, lab.shape[0] - 1]
    expected = np.float32([[160, 153], [517, 153], [517, 456], [160, 456]])
    assert min(np.max(np.linalg.norm(np.roll(unrotated, k, axis=0) - expected, axis=1)) for k in range(4)) < 1.2


@pytest.mark.parametrize("kind", ["flat_stock", "ribs_in_motif", "enclosing_sheet", "claimed_raw", "claimed_ribs"])
def test_adjacent_tile_requires_unclaimed_distinct_ribs_outside_the_photo(kind):
    lab, observations, scale = _scene()
    claims = []
    if kind == "flat_stock":
        lab[:153, :, 0] = lab[456:, :, 0] = 202
        lab[:, :160, 0] = lab[:, 518:, 0] = 202
    elif kind == "ribs_in_motif":
        lab[153:456, 160:170, 0] = lab[153:456, 149:150, 0] - 55
    elif kind == "enclosing_sheet":
        original = lab[130:479, 141:538].copy()
        lab[116:491, 128:551] = [70, 126, 129]
        lab[130:479, 141:538] = original
    else:
        left, top, right, bottom = (209, 461, 275, 465) if kind == "claimed_raw" else (155, 235, 157, 380)
        box = np.float32([[left, top], [right, top], [right, bottom], [left, bottom]]) / scale
        claims = [{"polygon": [{"x": float(x), "y": float(y)} for x, y in box]}]
        if kind == "claimed_ribs":
            # Both model masks miss the claim. It only intersects the photo's optical witness corridor.
            for polygon, _ in observations:
                vertices = [{"x": float(x), "y": float(y)} for x, y in polygon]
                assert not tiled_outline_conflicts(vertices, claims, lab.shape[1], lab.shape[0])

    image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    assert measured_adjacent_tile_outline(image, observations, claims) is None
