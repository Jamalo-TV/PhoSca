import cv2
import numpy as np

from app.pipeline.print_edges import _PRINT_SIDES
from app.pipeline.visible_scene_edge import suggest_visible_scene_edge


def test_scene_edge_requires_a_quiet_field_and_has_no_supported_enclosing_sheet(visible_scene_edge_page):
    image, quad = visible_scene_edge_page
    boundary = {name: {"support": 0., "ambiguous": False} for name in _PRINT_SIDES}
    trace, = suggest_visible_scene_edge(image, quad, boundary, [])
    np.testing.assert_allclose(trace[:, 0], [467.5, 467.5], atol=.5)
    assert abs(trace[1, 1] - trace[0, 1]) > 440
    neighbor = np.float32([[210, 380], [381, 380], [381, 470], [210, 470]])
    assert not suggest_visible_scene_edge(image, quad, boundary, [neighbor])

    sheet = np.full_like(image, (112, 137, 146))
    sheet[126:722, 171:980] = image[126:722, 171:980]
    assert not suggest_visible_scene_edge(sheet, quad, boundary, [])

    inscription = image.copy()
    cv2.putText(inscription, "1987", (205, 355), cv2.FONT_HERSHEY_SIMPLEX, 2, (55, 91, 124), 3)
    assert not suggest_visible_scene_edge(inscription, quad, boundary, [])
