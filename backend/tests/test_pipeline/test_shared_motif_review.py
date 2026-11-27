import numpy as np
import pytest

from app.pipeline.shared_motif_review import measure_shared_motif, near_nested_model_masks

_WIDTH, _HEIGHT = 840, 650
_SCALE = np.float32([_WIDTH - 1, _HEIGHT - 1])
_LARGER = (329, 199, 551, 452)
_SMALLER = (338, 217, 557, 439)
_MOTIF = (340, 220, 548, 436)


def _quad(bounds: tuple[int, int, int, int]) -> np.ndarray:
    left, top, right, bottom = bounds
    return np.float32([(left, top), (right, top), (right, bottom), (left, bottom)])


def _paint_scene(
    image: np.ndarray, bounds: tuple[int, int, int, int],
    ink: tuple[int, int, int], *, fade_top: bool = False,
) -> None:
    left, top, right, bottom = bounds
    y, x = np.indices((bottom - top + 1, right - left + 1), dtype=np.float32)
    texture = 8 * np.sin(y / 19) * np.cos(x / 13) + .035 * x
    scene = np.asarray(ink) + texture[..., None]
    target = image[top:bottom + 1, left:right + 1]
    if fade_top:
        opacity = np.minimum(1., y / 32)[..., None]
        scene = scene * opacity + target * (1. - opacity)
    target[:] = np.rint(scene).astype(np.uint8)


def _page(case: str) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    image = np.full((_HEIGHT, _WIDTH, 3), (190, 179, 173), np.uint8)
    if case == "stacked":
        # Each print has its own pale stock and distinct content. The top
        # print extends past one side of the still visible rear print.
        image[199:453, 329:552] = (225, 230, 236)
        _paint_scene(image, (339, 209, 541, 442), (62, 110, 142))
        image[217:440, 338:558] = (229, 233, 237)
        _paint_scene(image, (351, 230, 544, 426), (130, 75, 96))
    else:
        image[156:493, 285:601] = (227, 232, 238)
        _paint_scene(image, _MOTIF, (71, 94, 117), fade_top=case == "faint_top")
    return image, (_quad(_LARGER) / _SCALE, _quad(_SMALLER) / _SCALE)


def test_shared_motif_measures_the_authored_inner_boundary_and_respects_another_claim():
    image, proposals = _page("clear")
    assert near_nested_model_masks(*proposals)

    outline = measure_shared_motif(image, proposals, [])
    assert outline is not None
    assert np.linalg.norm(outline - _quad(_MOTIF), axis=1).max() < .4

    crossing_claim = _quad((514, 324, 600, 427)) / _SCALE
    assert measure_shared_motif(image, proposals, [crossing_claim]) is None


@pytest.mark.parametrize("case", ["faint_top", "stacked"])
def test_shared_motif_abstains_on_an_open_boundary_or_distinct_overlapping_prints(case: str):
    image, proposals = _page(case)
    assert near_nested_model_masks(*proposals)
    assert measure_shared_motif(image, proposals, []) is None
