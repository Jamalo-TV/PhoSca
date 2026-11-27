import numpy as np
import pytest

from app.pipeline.film_neighbor_edge import _corridor_scope


@pytest.mark.parametrize("case", ["exposed", "short_bridge", "shifted_edge", "neutral_strip"])
def test_corridor_requires_two_separate_native_transitions(case):
    wood = (94, 145, 176)
    stock = (216, 221, 226)
    image = np.full((480, 720, 3), wood, dtype=np.uint8)
    image[35:221, 40:680] = stock
    image[233:408, 40:680] = stock
    if case == "short_bridge":
        image[220:234, 335:348] = stock
    if case == "neutral_strip":
        image[221:233, 40:680] = (178, 178, 178)

    paper = np.float32([[40, 232.5], [679, 232.5]])
    film = np.float32([[679, 220.5], [40, 220.5]])
    if case == "shifted_edge":
        film[:, 1] += 2.5
    scope = _corridor_scope(image, paper, film)

    if case == "exposed":
        assert scope is not None
        np.testing.assert_allclose(scope[:, 1], [232.5, 232.5, 220.5, 220.5])
    else:
        assert scope is None
