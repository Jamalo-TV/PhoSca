"""Look for a continuous sheet of stock around separately detected motifs."""

from collections.abc import Iterator
from itertools import combinations
from math import sqrt

import cv2
import numpy as np

from app.pipeline.geometry import normalized_polygon_overlap, order_quad_cyclic, quad_from_polygon
from app.pipeline.print_edges import _edge_band, _is_pale_stock, refine_print_boundary


MULTIPANEL_PRINT_REVISION = 9
MULTIPANEL_PRINT_REVIEW = "MULTIPANEL_PRINT_REVIEW"
FILM_GRID_REVIEW = "FILM_GRID_REVIEW"
PAPER_GRID_REVIEW = "PAPER_GRID_REVIEW"
RECOVERED_STOCK_SOURCE = "shared_stock_boundary"
GRID_OUTLINE_REVIEW_SOURCE = "shared_grid_outline_review"
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


def nested_print_on_shared_stock(image: np.ndarray, outer_quad: np.ndarray, inner_quad: np.ndarray) -> bool:
    """Find a count-review candidate using ordered quadrilaterals in native pixels."""
    scale = np.float32([image.shape[1] - 1, image.shape[0] - 1])
    if np.any(scale <= 0):
        return False
    overlap = normalized_polygon_overlap(outer_quad / scale, inner_quad / scale)
    if overlap.second_containment < .99 or not .60 <= overlap.iou <= .88:
        return False

    to_unit = cv2.getPerspectiveTransform(outer_quad, _UNIT_QUAD)
    projected = cv2.perspectiveTransform(inner_quad[None], to_unit)[0]
    margins = np.minimum(projected, 1 - projected)
    if margins.min() < .004 or margins.max() > .18:
        return False
    edges = np.roll(projected, -1, axis=0) - projected
    tilt = np.r_[edges[::2, 1], edges[1::2, 0]]
    if np.max(np.abs(tilt)) > .02:
        return False

    sides = list(zip(inner_quad, np.roll(inner_quad, -1, axis=0), strict=True))
    fractions = np.linspace(.10, .90, 120, dtype=np.float32)
    rim = [_edge_band(image, start, stop, [-13, -10, -7, -4], fractions=fractions) for start, stop in sides]
    if any(band is None for band in rim):
        return False
    tones = np.float32(rim)
    stock = np.median(tones, axis=(0, 1, 2))
    if not _is_pale_stock(stock):
        return False
    matching = np.linalg.norm(tones - stock, axis=3) <= 11
    if np.any(matching.mean(axis=(1, 2)) < .66):
        return False

    # A separate inset can share the paper color. A distributed shadow
    # supplies contrary evidence, while a flush inset can stay ambiguous.
    return not any(_nested_print_cast_shadow(image, start, stop) for start, stop in sides)


def _nested_print_cast_shadow(image: np.ndarray, start: np.ndarray, stop: np.ndarray) -> bool:
    fractions = np.linspace(.04, .96, 220, dtype=np.float32)
    offsets = [-32, -23, -16, -12, -9, -6, -4]
    colors = _edge_band(image, start, stop, offsets, fractions=fractions)
    if colors is None:
        return True
    lightness = colors[..., 0].astype(np.float32)
    far = lightness[:, :3].mean(axis=1)
    middle = lightness[:, 3:5].mean(axis=1)
    near = lightness[:, 5:].mean(axis=1)
    darkens_toward_edge = (far > 172) & (far - middle >= 1) & (middle - near >= 2) & (far - near >= 4)
    return bool(
        darkens_toward_edge.mean() >= .42
        and all(part.mean() >= .27 for part in np.array_split(darkens_toward_edge, 4))
        and _is_pale_stock(np.median(colors, axis=(0, 1)))
    )


def grid_fragment_on_shared_stock(
    image: np.ndarray, outer_quad: np.ndarray, inner_quad: np.ndarray, *, check_seam_edge: bool = False,
) -> bool:
    """Check whether a proposed fragment cuts across a regular printed grid."""
    return _paper_grid_seam(image, outer_quad, inner_quad, check_seam_edge=check_seam_edge) is not None


def paper_grid_review_seam(image: np.ndarray, outer_quad: np.ndarray, inner_quad: np.ndarray) -> np.ndarray | None:
    """Return the proposal's cut when nearby print texture supports a review."""
    return _paper_grid_seam(image, outer_quad, inner_quad, check_seam_edge=True, nearby_rows=8)


def _paper_grid_seam(
    image: np.ndarray, outer_quad: np.ndarray, inner_quad: np.ndarray, *, check_seam_edge: bool, nearby_rows: int = 0,
) -> np.ndarray | None:
    if min(np.linalg.norm(np.roll(outer_quad, -1, axis=0) - outer_quad, axis=1)) < 480:
        return None
    for turn in range(4):
        outer = np.roll(outer_quad, -turn, axis=0)
        transform = cv2.getPerspectiveTransform(outer, _UNIT_QUAD)
        inner = order_quad_cyclic(cv2.perspectiveTransform(inner_quad[None], transform)[0])
        left, right = float(inner[:, 0].min()), float(inner[:, 0].max())
        upper_edge = inner[:2, 1]
        seam = float(upper_edge.mean())
        if (not .26 <= seam <= .74 or np.ptp(upper_edge) > .025 or inner[2:, 1].min() < .96
                or inner[:, 1].max() > 1.012 or left < .04 or right > .96 or right - left < .78):
            continue

        from_unit = np.linalg.inv(transform)
        native = cv2.perspectiveTransform(inner[None], from_unit)[0]
        if _grid_fragment_has_paper_edge(image, native, check_seam_edge=check_seam_edge):
            return None
        if any(_grid_fragment_corner_shadow(image, from_unit, corner, outward)
               for corner, outward in zip(inner[:2], (-1, 1), strict=True)):
            return None
        if (_grid_fragment_side_shadow(image, from_unit, inner[0], inner[3], -1)
                or _grid_fragment_side_shadow(image, from_unit, inner[1], inner[2], 1)):
            return None

        width, height = 800, 1040
        to_grid = np.diag([width - 1., height - 1., 1.]) @ transform
        aligned = cv2.warpPerspective(image, to_grid, (width, height))
        lab = cv2.cvtColor(aligned, cv2.COLOR_BGR2LAB)
        center = round(seam * (height - 1))
        # A separate layer can interrupt either lateral strip beside the cut.
        strips = (lab[center - 38:center + 39, round(.024 * width):round((left - .012) * width)],
                  lab[center - 38:center + 39, round((right + .012) * width):round(.976 * width)])
        if not all(_continuous_grid_margin(strip) for strip in strips):
            return None
        start, end = round((left + .10 * (right - left)) * width), round((right - .10 * (right - left)) * width)
        witness = _perforation_witness(lab[:, start:end, 0], center, nearby_rows)
        if witness is None:
            return None
        if check_seam_edge and _grid_fragment_has_offset_rim(image, native[0], native[1]):
            return None
        if nearby_rows and not _native_perforations_supported(image, np.linalg.inv(to_grid), start, end, *witness):
            return None
        return native[:2]
    return None


def _grid_fragment_has_paper_edge(image: np.ndarray, quad: np.ndarray, *, check_seam_edge: bool) -> bool:
    # A border along the proposed cut or a shadow alongside it is evidence
    # for a layered print even when both layers carry the same design.
    offsets = list(range(0 if check_seam_edge else 2, 9))
    seam = _edge_band(image, quad[0], quad[1], offsets)
    if seam is None:
        return True
    pale = (seam[..., 0] >= 172) & (np.max(np.abs(seam[..., 1:] - 128), axis=2) <= 16)
    if np.count_nonzero(pale.mean(axis=0) >= .75) >= 2:
        return True
    sides = ((quad[1], quad[2]), (quad[3], quad[0]))
    return any(_nested_print_cast_shadow(image, start, stop) for start, stop in sides)


def _grid_fragment_has_offset_rim(image: np.ndarray, start: np.ndarray, stop: np.ndarray) -> bool:
    """Look for distributed paper color near an approximate model edge."""
    edge = stop - start
    inward = np.float32([-edge[1], edge[0]]) / np.linalg.norm(edge)
    fractions = np.linspace(.06, .94, 256, dtype=np.float32)
    along = start + fractions[:, None] * edge
    offsets = np.arange(-12., 12.01, .5, dtype=np.float32)
    tilts = np.arange(-6., 6.01, 1.5, dtype=np.float32)
    depths = offsets[None, :, None] + tilts[:, None, None] * (fractions - .5)
    points = along + depths[..., None] * inward
    sampled = _sample_lab(image, points.reshape(-1, len(fractions), 2))
    if sampled is None:
        return True
    stock = np.median(sampled, axis=1)
    neutral = (stock[:, 0] >= 125) & (np.abs(stock[:, 1:] - 128).max(axis=1) <= 25)
    matching = np.linalg.norm(sampled - stock[:, None], axis=2) <= 12
    # A paper rim must have support along its length. A row of light holes
    # in an otherwise dark printed rail can have a similar overall mean.
    sections = matching.reshape(-1, 8, len(fractions) // 8).mean(axis=2)
    return bool(np.any(neutral & (sections.min(axis=1) >= .65)))


def _grid_fragment_corner_shadow(
    image: np.ndarray, from_unit: np.ndarray, corner: np.ndarray, outward: int,
) -> bool:
    # Look outside the ink for a shadow that ends at the proposed cut.
    for near, far, above, below in (
        ([.001, .002, .003], [.014, .016, .018, .020], [-.003, -.002, -.001], [.005, .007, .009, .010]),
        ([.004, .006, .008, .010], [.020, .025, .032], [-.009, -.007, -.005, -.003], [.003, .005, .007, .009, .011]),
    ):
        dx = outward * np.float32([*near, *far])
        dy = np.float32([*above, *below])
        colors = _sample_lab(image, _source_points(from_unit, 0, corner[0] + dx, corner[1] + dy))
        if colors is None:
            return True
        lightness = colors[..., 0]
        height, width = len(above), len(near)
        shaded = lightness[:height, :width]
        other_stock = min(np.median(lightness[:height, width:]), np.median(lightness[height:, :width]))
        if np.percentile(shaded, 20) >= 185 and other_stock - np.median(shaded) >= 5:
            return True
    return False


def _grid_fragment_side_shadow(
    image: np.ndarray, from_unit: np.ndarray, start: np.ndarray, end: np.ndarray, outward: int,
) -> bool:
    fractions = np.linspace(.06, .87, 96, dtype=np.float32)
    along = start + fractions[:, None] * (end - start)
    offsets = outward * np.float32([.0025, .0035, .0045, .0055, .022, .026, .030, .034])
    points = along[:, None, :] + np.column_stack((offsets, np.zeros_like(offsets)))
    native = cv2.perspectiveTransform(points.reshape(1, -1, 2), from_unit)[0].reshape(points.shape)
    colors = _sample_lab(image, native)
    if colors is None:
        return True
    near, far = (part.mean(axis=1) for part in np.split(colors[..., 0], 2, axis=1))
    return bool(np.mean((near >= 172) & (far >= 183) & (far - near >= 5)) >= .70)


def _continuous_grid_margin(strip: np.ndarray) -> bool:
    if strip.shape[1] < 3:
        return False
    rows = np.median(strip, axis=1)
    stock = np.median(rows, axis=0)
    first, last = (np.median(half, axis=0) for half in np.array_split(rows, 2))
    levels = np.percentile(rows[:, 0], (10, 90))
    return bool(_is_pale_stock(stock) and levels[1] - levels[0] <= 7
                and np.linalg.norm(last - first) <= 3.5
                and np.mean(np.linalg.norm(rows - stock, axis=1) <= 8) >= .92)


def _perforation_rails_cross_split(luminance: np.ndarray, split: int) -> bool:
    return _perforation_witness(luminance, split) is not None


def _perforation_witness(luminance: np.ndarray, split: int, nearby_rows: int = 0) -> tuple[int, int] | None:
    light = luminance.astype(np.float32)
    detail = np.clip(light - cv2.GaussianBlur(light, (41, 1), 7), -64, 64)
    strength = np.minimum(np.percentile(np.abs(light[:, 2:] - light[:, :-2]), 90, axis=1), 100) / 100
    lags = np.arange(8, 38)
    correlations = []
    for lag in lags:
        left, right = detail[:, :-lag], detail[:, lag:]
        energy = np.sqrt(np.maximum(1, (left ** 2).sum(axis=1) * (right ** 2).sum(axis=1)))
        correlations.append((left * right).sum(axis=1) / energy)
    correlations = np.stack(correlations, axis=1)
    best = correlations.max(axis=1)
    scores = best * strength
    # Repeated holes can correlate just as well at twice the true pitch.
    pitch = lags[(correlations >= best[:, None] - .012).argmax(axis=1)]
    active = scores >= .50
    bounds = np.flatnonzero(np.diff(np.r_[False, active, False]))
    rails = np.array([(start + stop) // 2 for start, stop in bounds.reshape(-1, 2)
                      if stop - start >= 3], dtype=int)
    rows = sorted(range(split - nearby_rows, split + nearby_rows + 1), key=lambda row: abs(row - split))
    for row in rows:
        before, after = rails[rails < row], rails[rails > row]
        if len(before) < 3 or len(after) < 3:
            continue
        near = np.r_[before[-3:], after[:3]]
        if np.ptp(pitch[near]) > 1 or scores[near].min() < .65:
            continue
        gaps = np.diff(near)
        short, long = gaps[::2], gaps[1::2]
        # Printed strips repeat a short inter-strip gap and a wider frame row.
        if (short.min() >= 12 and short.max() <= 55
                and short.max() <= 1.3 * short.min() and long.max() <= 1.3 * long.min()
                and 2 * short.max() <= long.min() <= 4.5 * short.min()):
            witness = min(near[2:4], key=lambda rail: abs(rail - split))
            if not nearby_rows or abs(witness - split) <= nearby_rows:
                return int(witness), int(pitch[witness])
    return None


def _native_perforations_supported(
    image: np.ndarray, from_grid: np.ndarray, start: int, end: int, row: int, pitch: int,
) -> bool:
    ends = cv2.perspectiveTransform(np.float32([[[start, row], [end - 1, row]]]), from_grid)[0]
    count = min(2000, int(np.linalg.norm(ends[1] - ends[0])))
    expected = round(pitch * (count - 1) / (end - start - 1))
    if count < 160 or expected < 4 or expected + 1 >= count // 4:
        return False
    x = np.linspace(start, end - 1, count, dtype=np.float32)
    points = np.column_stack((x, np.full_like(x, row)))
    samples = np.rint(cv2.perspectiveTransform(points[None], from_grid)[0]).astype(int)
    if ((samples < 0) | (samples >= np.array(image.shape[1::-1]))).any():
        return False
    # Read the nearest source pixels to corroborate the periodic ink contrast
    # without the interpolation used by the normalized search.
    bgr = image[samples[:, 1], samples[:, 0]][None]
    light = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)[0, :, 0].astype(np.float32)
    detail = np.clip(light - cv2.GaussianBlur(light[None], (31, 1), 4)[0], -64, 64)
    strong = 0
    for block, ink in zip(np.array_split(detail, 4), np.array_split(light, 4), strict=True):
        correlations = []
        for lag in range(expected - 1, expected + 2):
            left, right = block[:-lag], block[lag:]
            correlations.append((left @ right) / np.sqrt(max(1, (left @ left) * (right @ right))))
        best = max(correlations)
        if best < .45:
            return False
        strong += best >= .70 and np.percentile(np.abs(ink[2:] - ink[:-2]), 90) >= 30
    return strong >= 3


def film_grid_review_seam(image: np.ndarray, first: np.ndarray, second: np.ndarray) -> np.ndarray | None:
    """Locate a regular film-grid divider between two counted proposals."""
    directions = []
    for quad in (first, second):
        edges = np.roll(quad, -1, axis=0) - quad
        lengths = np.linalg.norm(edges, axis=1)
        if lengths.min() < 1:
            return None
        directions.append(edges / lengths[:, None])
    # Near a diagonal rotation, the same physical corner need not be the
    # first corner in both image-ordered quads.
    first_edges, second_edges = directions
    offset = max(range(4), key=lambda turn: np.sum(first_edges * np.roll(second_edges, -turn, axis=0)))
    second = np.roll(second, -offset, axis=0)
    for turn in range(4):
        a, b = np.roll(first, -turn, axis=0), np.roll(second, -turn, axis=0)
        dx, dy = a[1] - a[0]
        inward = np.float32([-dy, dx])
        upper, lower = sorted((a, b), key=lambda quad: quad.mean(axis=0) @ inward)
        envelope = np.float32([upper[0], upper[1], lower[2], lower[3]])
        sides = np.linalg.norm(np.roll(envelope, -1, axis=0) - envelope, axis=1)
        if not cv2.isContourConvex(envelope) or sides.min() < 480:
            continue
        facing = np.float32([*upper[[3, 2]], *lower[:2]])
        transform = cv2.getPerspectiveTransform(envelope, _UNIT_QUAD)
        projected = cv2.perspectiveTransform(facing[None], transform)[0]
        y = projected[:, 1].reshape(2, 2)
        seam = float(y.mean())
        separation = y[1] - y[0]
        if (not .28 <= seam <= .72 or np.any(separation < .003) or np.any(separation > .027)
                or np.ptp(y, axis=1).max() > .018
                or np.max(np.abs(projected[:, 0] - [0, 1, 0, 1])) > .06):
            continue
        width, height = 800, 1040
        scaled = np.diag([width - 1., height - 1., 1.]) @ transform
        aligned = cv2.warpPerspective(image, scaled, (width, height))
        light = cv2.cvtColor(aligned[:, 120:680], cv2.COLOR_BGR2LAB)[..., 0]
        center = round(seam * (height - 1))
        pale = np.percentile(light, 15, axis=1) >= 160
        bounds = np.flatnonzero(np.diff(np.r_[False, pale, False])).reshape(-1, 2)
        crossing = next((int(stop - start) for start, stop in bounds if start <= center < stop), 0)
        neighbors = [stop - start for start, stop in bounds
                     if 70 <= abs((start + stop) / 2 - center) <= 360 and 4 <= stop - start <= 24]
        if len(neighbors) < 2 or not 4 <= crossing <= 24 or not .6 <= crossing / np.median(neighbors) <= 1.5:
            continue
        if _perforation_rails_cross_split(light, center):
            return facing.reshape(2, 2, 2).mean(axis=0)
    return None


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
