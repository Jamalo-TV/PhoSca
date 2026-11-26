import numpy as np
import pytest

from app.pipeline import tile_print_review


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
