import numpy as np

from app.pipeline.bowed_scene_preview import measure_bowed_scene
from app.pipeline.crop_sampling import _outline_mask
from app.pipeline.print_edges import _PRINT_SIDES


def test_bowed_scene_preview_adds_only_visible_image_centres():
    y, x = np.indices((1040, 1130), dtype=np.float32)
    quad = np.float32([[180, 112], [930, 112], [910, 940], [180, 940]])
    along = (y - 112) / 828
    current = 930 - 20 * along
    right = current + 18 - 59 * along + 14 * along ** 2
    in_height = (y >= 112) & (y <= 940)
    visible = in_height & (x >= 180) & (x <= right)
    pale = in_height & (x > right) & (x <= right + 8)
    page = np.full((*y.shape, 3), 145, np.uint8)
    page[pale] = 220
    page[visible] = 83
    boundary = {side: {"support": 1., "corner_support": 1., "ambiguous": False} for side in _PRINT_SIDES}
    boundary["right"] = {"support": .1, "corner_support": None, "ambiguous": False}

    measured = measure_bowed_scene(page, quad, boundary, [])
    assert measured is not None
    before = _outline_mask(quad, y.shape) > 0
    after = _outline_mask(measured.outline, y.shape) > 0
    extra = after & ~before
    assert np.count_nonzero(extra) > 1_500
    assert not np.any(before & ~after) and not np.any(extra & ~visible)

    peer = np.float32([[934, 151], [950, 151], [950, 191], [934, 191]])
    assert measure_bowed_scene(page, quad, boundary, [peer]) is None
    page[pale] = 145
    assert measure_bowed_scene(page, quad, boundary, []) is None
