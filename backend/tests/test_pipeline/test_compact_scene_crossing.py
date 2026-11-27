import numpy as np
import pytest

from app.pipeline.fragmented_prints import compact_scene_crossing_trace

_PARENT = np.float32([[105, 78], [383, 78], [383, 397], [105, 397]])
_FRAGMENT = np.float32([[212, 78], [383, 78], [383, 255], [212, 255]])


def _source(case: str = "continuous") -> np.ndarray:
    image = np.full((485, 550, 3), (214, 225, 230), np.uint8)
    image[78:398, 105:384] = (85, 99, 111)
    # Local scene features cross the left candidate cut. The bottom cut
    # has no such continuations. The top and right follow the parent.
    for row in (111, 135, 161, 181, 208, 229):
        image[row:row + 4, 165:260] = (174, 182, 188)

    if case == "distinct_print":
        image[78:256, 212:384] = (223, 231, 235)
        image[85:249, 219:377] = (142, 120, 84)
        for row in (108, 136, 162, 184, 208, 230):
            image[row:row + 3, 225:260] = (82, 115, 146)
    elif case == "paper_gap":
        image[78:256, 210:215] = (227, 233, 238)
    elif case == "one_sided_art":
        image[78:256, 212:384] = (85, 99, 111)
    return image


def test_compact_scene_crossing_traces_only_the_observed_cut():
    image = _source()
    trace = compact_scene_crossing_trace(image, _FRAGMENT, _PARENT, [])
    assert trace is not None
    np.testing.assert_allclose(trace, [[212, 233], [212, 115]], atol=2)

    # A separate current paper claim covering the measured part of the cut
    # must block the trace even when the raster under the claim is unchanged.
    crossing_claim = np.float32([[196, 105], [228, 105], [228, 235], [196, 235]])
    assert compact_scene_crossing_trace(image, _FRAGMENT, _PARENT, [crossing_claim]) is None


@pytest.mark.parametrize("case", ["distinct_print", "paper_gap", "one_sided_art"])
def test_compact_scene_crossing_abstains_when_scene_features_do_not_cross(case: str):
    assert compact_scene_crossing_trace(_source(case), _FRAGMENT, _PARENT, []) is None
