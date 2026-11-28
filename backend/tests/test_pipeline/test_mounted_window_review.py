import numpy as np

from app.pipeline.mounted_window_review import mounted_window_traces


def test_visible_album_window_preserves_only_the_unclaimed_source_piece():
    width, height = 800, 1000
    page = np.full((height, width, 3), (195, 203, 213), dtype=np.uint8)
    y, x = np.mgrid[:340, :225]
    light = 74 + 15 * np.sin(x / 4) + 21 * np.cos(y / 5) + 13 * np.sin((x + y) / 7)
    page[62:402, 465:690] = np.clip(light[..., None] + [0, 5, 12], 0, 255).astype(np.uint8)
    leaf = np.float32([[20, 20], [779, 20], [779, 969], [20, 969]])

    def claim(left, top, right, bottom):
        return {"polygon": [{"x": x / (width - 1), "y": y / (height - 1)}
                            for x, y in ((left, top), (right, top), (right, bottom), (left, bottom))]}

    right, bottom = mounted_window_traces(page, leaf, 1, [], [])
    assert np.max(np.abs(right[:, 0] - 689.5)) < 2
    assert np.max(np.abs(bottom[:, 1] - 401.5)) < 2
    clipped, = mounted_window_traces(page, leaf, 1, [], [claim(370, 225, 730, 540)])
    assert np.max(clipped[:, 1]) < 210
    assert np.min(clipped[:, 1]) > 62
    assert not mounted_window_traces(page, leaf, 1, [claim(470, 70, 690, 390)], [])

    page[62:402, 465:690] = (48, 48, 52)
    assert not mounted_window_traces(page, leaf, 1, [], [])
