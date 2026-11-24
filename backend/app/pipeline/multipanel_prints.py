"""Look for a continuous sheet of stock around separately detected motifs."""

from collections.abc import Iterator
from itertools import combinations

import cv2
import numpy as np

from app.pipeline.print_edges import _is_pale_stock, refine_print_boundary


MULTIPANEL_PRINT_REVISION = 1
MULTIPANEL_PRINT_REVIEW = "MULTIPANEL_PRINT_REVIEW"
_UNIT_QUAD = np.float32([[0, 0], [1, 0], [1, 1], [0, 1]])


def _sample_lab(image: np.ndarray, points: np.ndarray) -> np.ndarray | None:
    if np.any((points < 0) | (points > np.array(image.shape[1::-1]) - 1)):
        return None
    pixels = cv2.remap(image, points[..., 0].copy(), points[..., 1].copy(), cv2.INTER_LINEAR)
    return cv2.cvtColor(pixels, cv2.COLOR_BGR2LAB).astype(np.float32)


def _common_stock(image: np.ndarray, quad: np.ndarray) -> np.ndarray | None:
    """Check the physical perimeter at native resolution, including narrow breaks."""
    samples = []
    for start, end in zip(quad, np.roll(quad, -1, axis=0), strict=True):
        edge = end - start
        length = float(np.linalg.norm(edge))
        inward = np.float32([-edge[1], edge[0]]) / length
        along = np.linspace(.05, .95, max(32, int(np.ceil(length / .8))), dtype=np.float32)
        points = start + along[:, None] * edge
        colors = _sample_lab(image, points[None, :] + np.float32([2., 5., -2., -5.])[:, None, None] * inward)
        if colors is None:
            return None
        samples.append(colors)
    stock = np.median(np.concatenate(samples, axis=1)[:2], axis=(0, 1))
    if not _is_pale_stock(stock):
        return None
    for colors in samples:
        inside, outside = colors[:2], colors[2:]
        # Veto a break anywhere on the outer edge, including when only some
        # of the panels share a sheet and a separate print sits next to it.
        rim_matches = np.linalg.norm(inside - stock, axis=2) <= 7
        if rim_matches.mean() < .85 or not rim_matches.any(axis=0).all():
            return None
        if np.mean(np.linalg.norm(outside - stock, axis=2) >= 11) < .80:
            return None
    return stock


def _bridge_intervals(
    bounds: list[tuple[np.ndarray, np.ndarray]], dimensions: np.ndarray,
) -> Iterator[tuple[int, float, float]]:
    """Find a band between two side-by-side motifs in the physical print plane."""
    for (a_lo, a_hi), (b_lo, b_hi) in combinations(bounds, 2):
        for axis in range(2):
            cross = 1 - axis
            left = min(a_hi[axis], b_hi[axis])
            right = max(a_lo[axis], b_lo[axis])
            shared = min(a_hi[cross], b_hi[cross]) - max(a_lo[cross], b_lo[cross])
            short = min(a_hi[cross] - a_lo[cross], b_hi[cross] - b_lo[cross])
            if 4 / dimensions[axis] <= right - left <= .22 and shared >= .60 * short:
                yield axis, float(left), float(right)


def _panel_layout(panels: list[np.ndarray], quad: np.ndarray) -> Iterator[tuple[int, float, float]]:
    to_unit = cv2.getPerspectiveTransform(quad, _UNIT_QUAD)
    lengths = np.linalg.norm(np.roll(quad, -1, axis=0) - quad, axis=1)
    dimensions = (lengths[:2] + lengths[2:]) * .5
    projected = [cv2.perspectiveTransform(panel[None], to_unit)[0] for panel in panels]
    bounds = [(panel.min(axis=0), panel.max(axis=0)) for panel in projected]
    return _bridge_intervals(bounds, dimensions)


def shared_stock_outline(
    image: np.ndarray, seed: np.ndarray, panels: list[np.ndarray],
) -> np.ndarray | None:
    """Measure an outer edge and check that a separator joins opposite paper rims."""
    if np.linalg.norm(np.roll(seed, -1, axis=0) - seed, axis=1).min() < 48:
        return None
    if next(_panel_layout(panels, seed), None) is None:
        return None
    quad, sides, reasons = refine_print_boundary(image, seed)
    if quad is None or reasons or any(side["support"] < .80 for side in sides.values()):
        return None
    stock = _common_stock(image, quad)
    if stock is None:
        return None
    from_unit = cv2.getPerspectiveTransform(_UNIT_QUAD, quad)
    for axis, left, right in _panel_layout(panels, quad):
        if left < .06 or right > .94:
            continue
        along = np.linspace(.035, .965, 75, dtype=np.float32)
        across = np.linspace(left, right, 7, dtype=np.float32)[1:-1]
        xx, yy = np.meshgrid(across, along)
        unit = np.stack((xx, yy) if axis == 0 else (yy, xx), axis=-1)
        points = cv2.perspectiveTransform(unit.reshape(1, -1, 2), from_unit)[0].reshape(unit.shape)
        colors = _sample_lab(image, points)
        if colors is not None and np.mean(np.linalg.norm(colors - stock, axis=2) <= 8) >= .88:
            return quad
    return None
