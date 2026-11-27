import cv2
import numpy as np

from app.pipeline.manual_paper_extent import measure_manual_paper_extent


def test_manual_extent_requires_a_local_material_transition_and_clear_source_scope(rounded_corner_page):
    page, quad, paper, _, backing = rounded_corner_page
    inset = quad.copy()
    inset[:2, 1] += 3
    result = measure_manual_paper_extent(page, inset, [])
    assert result is not None
    assert np.linalg.norm(np.diff(result.trace, axis=0)) > 150
    assert np.allclose(result.trace[:, 1], inset[0, 1] - .5)
    assert measure_manual_paper_extent(page, quad, []) is None

    # The claim intersects the source sampling band while lying outside the trace.
    claim = np.float32([[155, 68], [173, 68], [173, 73], [155, 73]])
    assert measure_manual_paper_extent(page, inset, [claim]) is None

    rib = quad.copy()
    rib[:2, 1] -= 6
    assert measure_manual_paper_extent(page, rib, []) is None

    # A blurred neutral shadow on the album creates a luma rail without a paper transition.
    shadow = backing.copy()
    shade = np.zeros(shadow.shape[:2], np.float32)
    shade[68:74, 100:375] = 42
    shadow[..., 0] = np.clip(shadow[..., 0] - cv2.GaussianBlur(shade, (7, 7), 1.0), 0, 255)
    shadow[paper] = (135, 144, 123)
    assert measure_manual_paper_extent(cv2.cvtColor(shadow, cv2.COLOR_LAB2BGR), quad, []) is None
