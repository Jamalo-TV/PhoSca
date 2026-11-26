import cv2
import numpy as np
import pytest

from app.pipeline.source_strip_review import source_strip_outline


def _strip_scene():
    page = np.full((760, 1050, 3), (146, 152, 158), np.uint8)
    page[91:421, 210:711] = (78, 84, 87)
    texture = np.random.default_rng(42).integers(49, 112, (59, 516), dtype=np.uint8)
    page[517:576, 225:741] = np.stack((texture, texture + 9, texture + 13), axis=-1)
    page[576:642, 225:741] = (6, 7, 8)
    held = np.float32([[210, 91], [710, 91], [710, 420], [210, 420]])
    boundary = {"bottom": {"support": .96, "corner_support": .92}}
    return page, held, boundary


@pytest.mark.parametrize("rotate", [False, True])
def test_locates_only_visible_picture_between_the_album_band_and_dark_backing(rotate):
    page, held, boundary = _strip_scene()
    if rotate:
        height = page.shape[0]
        page = cv2.rotate(page, cv2.ROTATE_90_CLOCKWISE)
        held = np.column_stack((height - 1 - held[:, 1], held[:, 0]))
    visible = source_strip_outline(page, held, boundary, [])
    assert visible is not None
    if rotate:
        visible = np.column_stack((visible[:, 1], height - 1 - visible[:, 0]))
    assert np.all((visible[:, 0] > 225) & (visible[:, 0] < 740))
    assert np.all((visible[:, 1] > 517) & (visible[:, 1] < 576))


@pytest.mark.parametrize("competing", ["dark_backing_alone", "flat_card", "already_claimed", "uncertain_anchor"])
def test_requires_texture_both_material_transitions_and_an_unclaimed_location(competing):
    page, held, boundary = _strip_scene()
    claims = []
    if competing == "dark_backing_alone":
        page[517:576, 225:741] = (146, 152, 158)
    elif competing == "flat_card":
        page[517:576, 225:741] = (73, 82, 86)
    elif competing == "already_claimed":
        claims.append(np.float32([[255, 522], [401, 523], [401, 561], [255, 560]]))
    else:
        boundary["bottom"]["ambiguous"] = True
    assert source_strip_outline(page, held, boundary, claims) is None
