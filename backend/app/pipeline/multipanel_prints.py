"""Look for a continuous sheet of stock around separately detected motifs."""

from collections.abc import Iterator
from itertools import combinations
from math import sqrt

import cv2
import numpy as np

from app.pipeline.geometry import order_quad_cyclic, quad_from_polygon
from app.pipeline.print_edges import _is_pale_stock, refine_print_boundary


MULTIPANEL_PRINT_REVISION = 3
MULTIPANEL_PRINT_REVIEW = "MULTIPANEL_PRINT_REVIEW"
RECOVERED_STOCK_SOURCE = "shared_stock_boundary"
_UNIT_QUAD = np.float32([[0, 0], [1, 0], [1, 1], [0, 1]])
_STOCK_DISCOVERY_PIXELS = 800_000


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
        backing = np.linalg.norm(outside - stock, axis=2) >= 11
        if backing.mean() < .80 and backing[0].mean() < .90:
            # In a narrow gutter, the nearer sample can be album backing even
            # when the farther sample is already on a neighboring print.
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


def _panel_layout(panels: list[np.ndarray], quad: np.ndarray) -> list[tuple[int, float, float]]:
    to_unit = cv2.getPerspectiveTransform(quad, _UNIT_QUAD)
    lengths = np.linalg.norm(np.roll(quad, -1, axis=0) - quad, axis=1)
    dimensions = (lengths[:2] + lengths[2:]) * .5
    projected = [cv2.perspectiveTransform(panel[None], to_unit)[0] for panel in panels]
    bounds = [(panel.min(axis=0), panel.max(axis=0)) for panel in projected]
    if len(panels) == 2:
        return list(_bridge_intervals(bounds, dimensions))

    # Find full strips of paper that no motif crosses. Every motif must occupy
    # its own cell, so a single clean seam cannot justify grouping a mixed layout.
    separators = []
    for axis in range(2):
        ranges = sorted((float(low[axis]), float(high[axis])) for low, high in bounds)
        end = ranges[0][1]
        for start, stop in ranges[1:]:
            if 4 / dimensions[axis] <= start - end <= .22:
                separators.append((axis, end, start))
            end = max(end, stop)
    cells = {
        tuple(sum(low[axis] >= right for direction, _, right in separators if direction == axis) for axis in range(2))
        for low, _ in bounds
    }
    return separators if len(cells) == len(panels) else []


def _source_points(transform: np.ndarray, axis: int, across: np.ndarray, along: np.ndarray) -> np.ndarray:
    xx, yy = np.meshgrid(across, along)
    unit = np.stack((xx, yy) if axis == 0 else (yy, xx), axis=-1).astype(np.float32)
    return cv2.perspectiveTransform(unit.reshape(1, -1, 2), transform)[0].reshape(unit.shape)


def shared_stock_outline(
    image: np.ndarray, seed: np.ndarray, panels: list[np.ndarray], *, observed_stock: np.ndarray | None = None,
) -> np.ndarray | None:
    """Measure an outer edge and check that a separator joins opposite paper rims."""
    if np.linalg.norm(np.roll(seed, -1, axis=0) - seed, axis=1).min() < 48:
        return None
    if not _panel_layout(panels, seed):
        return None
    quad, sides, reasons = refine_print_boundary(image, seed)
    if observed_stock is not None and reasons == ["PRINT_BOUNDARY_AMBIGUOUS"]:
        # Two sides of a gutter can produce equally strong edges. Require
        # the connected stock seen inside the print to be on the inner side.
        quad, sides, reasons = refine_print_boundary(image, seed, rim_color=observed_stock)
    support = .95 if observed_stock is not None else .80
    if quad is None or reasons or any(side["support"] < support for side in sides.values()):
        return None
    stock = _common_stock(image, quad)
    if stock is None:
        return None
    layout = _panel_layout(panels, quad)
    if not layout or any(left < .06 or right > .94 for _, left, right in layout):
        return None
    from_unit = cv2.getPerspectiveTransform(_UNIT_QUAD, quad)
    axes = {axis for axis, _, _ in layout}
    if any(_edge_has_step(image, quad[index], quad[(index + 1) % 4])
           for axis in axes for index in (axis, axis + 2)):
        return None
    for axis, left, right in layout:
        crosses = sorted((start + stop) / 2 for other, start, stop in layout if other != axis)
        bounds = [0., *crosses, 1.]
        # Test each row or column as well as the entire separator. A seam along
        # just one row of a mosaic can disappear in the full-height average.
        parts = [(0., 1.)] if not crosses else [(0., 1.), *zip(bounds[:-1], bounds[1:], strict=True)]
        if any(not _continuous_bridge(image, quad, from_unit, stock, axis, left, right, start, stop)
               for start, stop in parts):
            return None
    return quad


def _continuous_bridge(
    image: np.ndarray, quad: np.ndarray, transform: np.ndarray, stock: np.ndarray,
    axis: int, left: float, right: float, start: float, stop: float,
) -> bool:
    span = stop - start
    along = start + np.linspace(.035, .965, 75, dtype=np.float32) * span
    across = np.linspace(left, right, 7, dtype=np.float32)[1:-1]
    colors = _sample_lab(image, _source_points(transform, axis, across, along))
    if colors is None or np.mean(np.linalg.norm(colors - stock, axis=2) <= 8) < .88:
        return False
    return not _has_stock_join(image, quad, transform, stock, axis, left, right, start, stop)


def _has_stock_join(
    image: np.ndarray, quad: np.ndarray, transform: np.ndarray, stock: np.ndarray,
    axis: int, left: float, right: float, start: float, stop: float,
) -> bool:
    edges = np.linalg.norm(np.roll(quad, -1, axis=0) - quad, axis=1)
    longest = max(edges[axis::2])
    width = (right - left) * longest
    # Ignore interpolation at the detected motif edges. Sample the remaining
    # band densely enough across its width to resolve a narrow print seam.
    margin = min((right - left) * .20, 4 / longest)
    count = max(5, min(2048, int(width * 2)))
    along = start + np.linspace(.12, .88, 96, dtype=np.float32) * (stop - start)
    colors = _sample_lab(image, _source_points(transform, axis, np.linspace(left + margin, right - margin, count), along))
    if colors is None:
        return True
    departures = np.linalg.norm(colors - stock, axis=2) > 8
    # Either a fixed column or a line slanting through most rows can be a join.
    if np.mean(departures, axis=0).max() >= .55 or np.mean(departures.any(axis=1)) >= .70:
        return True
    # A blurred pale gutter may differ by only a few lightness levels.
    # Compare with the stock on each side of it at the same row.
    samples_per_pixel = (count - 1) / (width - 2 * margin * longest)
    flank = max(1, round(3 * samples_per_pixel))
    if 2 * flank < count:
        light = colors[:, :, 0]
        first, middle, last = light[:, :-2 * flank], light[:, flank:-flank], light[:, 2 * flank:]
        trough = np.minimum(first, last) - middle
        ridge = middle - np.maximum(first, last)
        step = np.abs(last - first)
        seam = (trough > 1.5) | (ridge > 1.5) | (step > 2.5)
        length = .76 * (stop - start) * max(edges[1 - axis::2])
        drift = round(length * .10 * samples_per_pixel)
        if _coherent_join(seam, drift):
            return True
        delta = last - first
        if max(np.mean(delta > 1.5, axis=0).max(), np.mean(delta < -1.5, axis=0).max()) >= .85:
            return True
    return False


def _coherent_join(seam: np.ndarray, max_drift: int) -> bool:
    """Check straight and slightly slanted joins in the rectified paper strip."""
    rows, columns = seam.shape
    max_drift = min(max_drift, columns // 2)
    y = np.arange(rows)[:, None]
    for drift in np.unique(np.rint(np.linspace(-max_drift, max_drift, 37)).astype(int)):
        shift = np.rint(y * drift / (rows - 1)).astype(int)
        x = np.arange(max(0, -drift), min(columns, columns - drift))
        if np.mean(seam[y, x + shift], axis=0).max() >= .60:
            return True
    return False


def _edge_has_step(image: np.ndarray, start: np.ndarray, end: np.ndarray) -> bool:
    edge = end - start
    length = float(np.linalg.norm(edge))
    inward = np.float32([-edge[1], edge[0]]) / length
    along = np.linspace(.06, .94, max(60, min(2048, int(length))), dtype=np.float32)
    offsets = np.arange(-4, 4.01, .5, dtype=np.float32)
    points = start + along[:, None] * edge
    colors = _sample_lab(image, points[:, None, :] + offsets[None, :, None] * inward)
    if colors is None:
        return True
    exterior, interior = colors[:, :3].mean(axis=1), colors[:, -3:].mean(axis=1)
    difference = interior - exterior
    squared_contrast = (difference ** 2).sum(axis=1)
    progress = ((colors - exterior[:, None]) * difference[:, None]).sum(axis=2)
    crossing = (progress >= .5 * squared_contrast[:, None]).argmax(axis=1)
    rows = np.arange(len(crossing))
    before, after = progress[rows, crossing - 1], progress[rows, crossing]
    transitions = offsets[crossing] - .5 * (after - .5 * squared_contrast) / np.maximum(after - before, 1)
    reliable = (squared_contrast >= 81) & (crossing > 0)
    bands = [transitions[band][reliable[band]] for band in np.array_split(rows, 16)]
    if any(len(band) < 2 for band in bands):
        return True
    medians = np.array([np.median(band) for band in bands])
    return bool(np.ptp(medians) > .9 or np.max(np.abs(np.diff(medians))) > .65)


def recover_shared_stock_outline(image: np.ndarray, panels: list[np.ndarray]) -> np.ndarray | None:
    """Find a physical outline when the detector proposed only the interior panels."""
    points = np.concatenate(panels)
    frame = order_quad_cyclic(cv2.boxPoints(cv2.minAreaRect(points)))
    layout = _panel_layout(panels, frame)
    if not layout and len(panels) > 2:
        # Perspective can close a narrow projected divider in the rotated box.
        frame, _ = quad_from_polygon(points)
        layout = _panel_layout(panels, frame)
    if not layout:
        return None
    axis, left, right = max(layout, key=lambda separator: separator[2] - separator[1])
    low, high = frame.min(axis=0), frame.max(axis=0)
    pad = np.linalg.norm(np.roll(frame, -1, axis=0) - frame, axis=1).min() * .16
    origin = np.maximum(0, np.floor(low - pad).astype(int))
    limit = np.minimum(image.shape[1::-1], np.ceil(high + pad + 1).astype(int))
    crop = image[origin[1]:limit[1], origin[0]:limit[0]]
    height, width = crop.shape[:2]
    if min(height, width) < 48:
        return None
    transform = cv2.getPerspectiveTransform(_UNIT_QUAD, frame - origin.astype(np.float32))
    locations = _source_points(transform, axis, np.linspace(left, right, 5)[1:-1], np.linspace(.39, .61, 5))
    samples = _sample_lab(crop, locations)
    if samples is None:
        return None
    stock = np.median(samples, axis=(0, 1))
    if not _is_pale_stock(stock) or np.mean(np.linalg.norm(samples - stock, axis=2) <= 7) < .8:
        return None
    scale = min(1., sqrt(_STOCK_DISCOVERY_PIXELS / (width * height)))
    if scale < 1.:
        crop = cv2.resize(crop, (max(1, round(width * scale)), max(1, round(height * scale))), interpolation=cv2.INTER_AREA)
    # Convert pixel centers between the reduced discovery raster and the original.
    ratio = np.float32([crop.shape[1] / width, crop.shape[0] / height])
    point = (locations[2, 1] + .5) * ratio - .5
    x, y = np.rint(point).astype(int)
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).astype(np.float32)
    same_stock = np.uint8(np.linalg.norm(lab - stock, axis=2) <= 7)
    if not same_stock[y, x]:
        return None
    cv2.floodFill(same_stock, None, (int(x), int(y)), 2)
    component = np.uint8(same_stock == 2)
    if component[[0, -1]].any() or component[:, [0, -1]].any():
        return None
    contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) != 1:
        return None
    outline = (contours[0][:, 0].astype(np.float32) + .5) / ratio - .5 + origin.astype(np.float32)
    contour = cv2.convexHull(outline)
    if any(cv2.pointPolygonTest(contour, tuple(point), True) < -2 for panel in panels for point in panel):
        return None
    seed, _ = quad_from_polygon(outline)
    return shared_stock_outline(image, seed, panels, observed_stock=stock)
