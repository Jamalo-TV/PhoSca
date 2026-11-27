import numpy as np

from app.pipeline.bowed_scene_preview import measure_bowed_scene
from app.pipeline.crop_sampling import _outline_mask
from app.pipeline.print_edges import _PRINT_SIDES


def test_bowed_scene_preview_preserves_image_and_pale_stock():
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
    assert measured.outside_trace is not None
    before = _outline_mask(quad, y.shape) > 0
    after = _outline_mask(measured.outline, y.shape) > 0
    extra = after & ~before
    removed = before & ~after
    assert np.count_nonzero(extra) > 1_500
    assert np.count_nonzero(removed) > 500
    assert not np.any(removed & (visible | pale)) and not np.any(extra & ~visible)

    right_y, right_x = np.argwhere(removed)[np.count_nonzero(removed) // 2]
    mark = np.float32([[right_x - 3, right_y - 3], [right_x + 3, right_y - 3],
                       [right_x + 3, right_y + 3], [right_x - 3, right_y + 3]])
    protected = measure_bowed_scene(page, quad, boundary, [], protected=[mark])
    assert protected is not None and protected.outside_trace is None
    add_only = _outline_mask(protected.outline, y.shape) > 0
    np.testing.assert_array_equal(add_only, before | extra)

    written = page.copy()
    written[right_y:right_y + 2, right_x - 1:right_x + 2] = 139
    marked = measure_bowed_scene(written, quad, boundary, [])
    assert marked is not None and marked.outside_trace is None
    np.testing.assert_array_equal(_outline_mask(marked.outline, y.shape) > 0, add_only)

    wider_stock = page.copy()
    wider_stock[in_height & (x >= right + 24)] = (192, 169, 126)
    separate_edge = measure_bowed_scene(wider_stock, quad, boundary, [])
    assert separate_edge is not None and separate_edge.outside_trace is None
    np.testing.assert_array_equal(_outline_mask(separate_edge.outline, y.shape) > 0, add_only)

    peer = np.float32([[934, 151], [950, 151], [950, 191], [934, 191]])
    assert measure_bowed_scene(page, quad, boundary, [peer]) is None
    page[pale] = 145
    assert measure_bowed_scene(page, quad, boundary, []) is None
