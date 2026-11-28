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

    (right, bottom), lower = mounted_window_traces(page, leaf, 1, [], [])
    assert not lower
    assert np.max(np.abs(right[:, 0] - 689.5)) < 2
    assert np.max(np.abs(bottom[:, 1] - 401.5)) < 2
    (clipped,), lower = mounted_window_traces(page, leaf, 1, [], [claim(370, 225, 730, 540)])
    assert not lower
    assert np.max(clipped[:, 1]) < 210
    assert np.min(clipped[:, 1]) > 62
    assert mounted_window_traces(page, leaf, 1, [claim(470, 70, 690, 390)], []) == ([], {})

    page[62:402, 465:690] = (48, 48, 52)
    assert mounted_window_traces(page, leaf, 1, [], []) == ([], {})


def test_compare_a_separate_lower_window_inside_two_disputed_claims():
    width, height = 800, 1050
    page = np.full((height, width, 3), (195, 203, 213), dtype=np.uint8)
    leaf = np.float32([[20, 20], [779, 20], [779, 1029], [20, 1029]])

    def claim(left, top, right, bottom):
        return {"polygon": [{"x": x / (width - 1), "y": y / (height - 1)}
                            for x, y in ((left, top), (right, top), (right, bottom), (left, bottom))]}

    y, x = np.mgrid[:310, :225]
    light = 74 + 15 * np.sin(x / 4) + 21 * np.cos(y / 5) + 13 * np.sin((x + y) / 7)
    scene = np.clip(light[..., None] + [0, 5, 12], 0, 255).astype(np.uint8)
    page[62:372, 465:690] = scene
    page[659:969, 465:690] = scene
    overlaps = [claim(370, 270, 747, 996), claim(394, 595, 730, 995)]

    (upper,), lower = mounted_window_traces(page, leaf, 1, [], overlaps)
    assert upper[:, 1].max() < 255
    right, bottom, left = lower["traces"]
    assert np.max(np.abs(right[:, 0] - 689.5)) < 2
    assert np.max(np.abs(bottom[:, 1] - 968.5)) < 2
    assert np.max(np.abs(left[:, 0] - 464.5)) < 2
    assert all(scope[:, 1].min() > 600 for scope in lower["scopes"])

    for claims, weak in (([claim(522, 740, 605, 869)], overlaps), ([], overlaps[:1])):
        (unchanged,), lower = mounted_window_traces(page, leaf, 1, claims, weak)
        np.testing.assert_array_equal(unchanged, upper)
        assert not lower

    # Flat dark material has the same edges but no independent interior detail.
    page[659:969, 465:690] = (48, 48, 52)
    (unchanged,), lower = mounted_window_traces(page, leaf, 1, [], overlaps)
    np.testing.assert_array_equal(unchanged, upper)
    assert not lower
