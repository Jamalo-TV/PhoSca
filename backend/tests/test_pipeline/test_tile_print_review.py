import cv2
import numpy as np
import pytest

from app.pipeline import segmentation, tile_print_review
from app.pipeline.merged_print_review import MERGED_PRINT_PAIR


@pytest.mark.parametrize("case", ["separate", "flat_inside", "single_carrier", "shared_carrier", "wrong_seed"])
def test_contested_tile_followup_needs_independent_visible_photo_edges(case):
    height, width = 720, 660
    image = np.full((height, width, 3), [166, 192, 225], dtype=np.uint8)
    y, x = np.indices((height, width))
    texture = 35 * np.sin(.18 * x + .35 * y) + 24 * np.cos(.23 * x - .13 * y)
    image[98:350, 146:425] = [63, 82, 108]
    image[363:608, 149:432] = [81, 97, 122]
    if case != "flat_inside":
        image[110:338, 158:413] = (image[110:338, 158:413].astype(np.float32)
                                   + texture[110:338, 158:413, None]).astype(np.uint8)
    if case in {"single_carrier", "shared_carrier"}:
        bottom = 377 if case == "single_carrier" else 638
        top, left, right = 71, 116, 459
        sheet = image[top:bottom, left:right]
        inner = sheet[9:-9, 9:-9].copy()
        sheet[:] = [46, 40, 39]
        sheet[9:-9, 9:-9] = inner
    scale = np.float32([width - 1, height - 1])
    outline = np.float32([[146, 98], [424, 98], [424, 349], [146, 349]])
    observation = (outline + [2, -1]) / scale
    seed = outline / scale
    if case == "wrong_seed":
        seed += np.float32([35, 5]) / scale
    replacement = [{"x": float(x), "y": float(y)}
                   for x, y in np.float32([[149, 363], [431, 363], [431, 607], [149, 607]]) / scale]

    result = tile_print_review.contested_tile_followup_outline(image, seed, observation, replacement)

    if case == "separate":
        np.testing.assert_allclose(result, outline, atol=1.)
    else:
        assert result is None


@pytest.mark.parametrize("case", ["ribbed", "flat", "broken_rib", "raw_claim", "claimed_rib", "crossing", "merged"])
def test_ribbed_tile_requires_distributed_original_backing_and_an_unclaimed_outline(case):
    height, width = 600, 640
    y, x = np.indices((height, width))
    light = 201 + 41 * np.cos(2 * np.pi * y / 8)
    if case == "flat":
        light[:] = 201
    elif case == "broken_rib":
        light[(x >= 160) & (x <= 520) & ((y < 145) | (y > 455))] = 201
    lab = np.full((height, width, 3), [201, 129, 144], dtype=np.uint8)
    lab[..., 0] = light
    lab[145:456, 160:520] = [110, 136, 127]
    lab[260:331, 220:450] = [153, 141, 122]
    image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    scale = np.float32([width - 1, height - 1])
    seed = np.float32([[163, 149], [516, 142], [521, 453], [162, 459]]) / scale

    def rect(left, top, right, bottom):
        return [{"x": float(x / scale[0]), "y": float(y / scale[1])} for x, y in (
            (left, top), (right, top), (right, bottom), (left, bottom),
        )]

    if case in {"crossing", "merged"}:
        original = rect(69, 174, 154, 419)
        alternative = rect(136, 187, 259, 406)
        extra = {"crossing_print_suggestion": alternative} if case == "crossing" else {
            MERGED_PRINT_PAIR: {"outlines": [alternative]},
        }
        item = segmentation.SegmentationDetection({}, {"polygon": original, **extra}, .93, 1., True, [])
        prior = segmentation.SegmentationResult([item], {"tiled_print_review": {"candidates": 1, "runtime_ms": 0.}})
        revised = segmentation._offer_ribbed_backing_tile(image, prior, seed)
        assert revised.detections == [item]
        assert "fusion" not in revised.metadata
        return

    boxes = {
        "raw_claim": [(448, 143.55, 499, 144.25)],
        "claimed_rib": [(130, 113, 152, 144), (529, 113, 551, 144), (130, 455.5, 152, 480), (529, 455.5, 551, 480)],
    }
    claims = [{"polygon": rect(*box)} for box in boxes.get(case, [])]

    outline = tile_print_review.ribbed_backing_tile_outline(image, seed, claims)

    if case != "ribbed":
        assert outline is None
        return
    assert outline is not None
    corners = np.float32([(point["x"], point["y"]) for point in outline]) * scale
    np.testing.assert_allclose(corners, [[160, 145], [519, 145], [519, 455], [160, 455]], atol=1.5)


@pytest.mark.parametrize(
    ("reason", "support", "ambiguous", "expected"),
    [
        ("PRINT_BOUNDARY_CURVED", .95, False, True),
        ("PRINT_BOUNDARY_CURVED", .89, False, False),
        ("PRINT_BOUNDARY_CURVED", .95, True, False),
        ("PRINT_BOUNDARY_UNSUPPORTED", .95, False, False),
    ],
)
def test_curved_tile_suggestion_requires_all_four_native_edges(monkeypatch, reason, support, ambiguous, expected):
    image = np.zeros((240, 320, 3), dtype=np.uint8)
    seed = np.float32([[30, 20], [270, 24], [262, 190], [29, 192]])
    sides = {
        name: {"support": .98, "ambiguous": False}
        for name in ("top", "right", "bottom", "left")
    }
    sides["bottom"] = {"support": support, "ambiguous": ambiguous}
    monkeypatch.setattr(tile_print_review, "refine_print_boundary", lambda image, quad: (seed, sides, [reason]))

    outline = tile_print_review.tile_print_outline(image, seed / [319, 239])

    assert (outline is not None) == expected
    if expected:
        np.testing.assert_array_equal(outline, seed)


def test_rotated_review_requires_unique_agreement_and_consistent_native_fit(monkeypatch):
    image = np.zeros((240, 320, 3), dtype=np.uint8)
    scale = np.float32([319, 239])
    measured = np.float32([[30, 20], [270, 24], [262, 190], [29, 192]])
    first = measured / scale
    second = (measured + [2, 1]) / scale
    separate = first / 4 + .65
    monkeypatch.setattr(tile_print_review, "tile_print_outline",
                        lambda image, shape, *, require_clear_sides: measured)

    def collect(first_view, second_view):
        return tile_print_review.opposing_view_print_outlines(image, first_view, second_view, acceptance=.70)

    outline, = collect([(first, .80)], [(second, .56), (separate, .95)])
    np.testing.assert_array_equal(outline, measured)
    assert collect([(first, .80)], [(separate, .95)]) == []
    assert collect([(first, .80)], [(second, .56), (first, .80)]) == []
    assert collect([(first, .69)], [(second, .65)]) == []

    def disagreeing_fit(image, shape, *, require_clear_sides):
        return measured if shape is first else measured + [5, 0]

    monkeypatch.setattr(tile_print_review, "tile_print_outline", disagreeing_fit)
    assert collect([(first, .80)], [(second, .56)]) == []
