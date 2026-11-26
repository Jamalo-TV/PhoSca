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
