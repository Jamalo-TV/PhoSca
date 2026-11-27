import cv2
import numpy as np

from app.pipeline.ribbed_outline_review import measure_ribbed_outline_comparison


def test_ribbed_outline_requires_backing_and_no_enclosing_paper(rounded_corner_page):
    _, paper, _, rectangle, backing = rounded_corner_page
    lab = backing.copy()
    lab[rectangle] = (121, 151, 119)
    image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    inset = paper + np.float32([[2, 2], [-2, 2], [-2, -2], [2, -2]])

    measured, scope = measure_ribbed_outline_comparison(image, inset, [])
    boundary = paper - np.float32([[.5, .5], [-.5, .5], [-.5, -.5], [.5, -.5]])
    np.testing.assert_allclose(measured, boundary, atol=.1)
    neighbor = np.float32([[413, 180], [443, 180], [443, 310], [413, 310]])
    assert cv2.intersectConvexConvex(measured, neighbor)[0] == 0
    assert cv2.intersectConvexConvex(scope, neighbor)[0] > 1
    assert measure_ribbed_outline_comparison(image, inset, [neighbor]) is None

    flat = np.full_like(lab, (185, 128, 144))
    flat[rectangle] = lab[rectangle]
    assert measure_ribbed_outline_comparison(cv2.cvtColor(flat, cv2.COLOR_LAB2BGR), inset, []) is None

    enclosed = np.full_like(image, (65, 80, 105))
    enclosed[46:434, 46:433] = image[46:434, 46:433]
    assert measure_ribbed_outline_comparison(enclosed, inset, []) is None
