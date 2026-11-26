"""Measure a scalloped paper silhouette against a bright scanner backing."""

from dataclasses import dataclass

import cv2
import numpy as np

from app.pipeline.geometry import quad_from_polygon
from app.pipeline.print_edges import _backing_color, _edge_band, _is_pale_stock, _sample_lab, _side_line


SCALLOPED_PRINT_REVIEW = "PRINT_SCALLOPED_OUTLINE_REVIEW"
SCALLOPED_PRINT_CONFLICT = "PRINT_SCALLOPED_OWNERSHIP_UNCERTAIN"
SCALLOPED_PRINT_REVISION = 2


@dataclass(frozen=True)
class ScallopedPrint:
    outline: np.ndarray
    quad: np.ndarray
    stock_colors: np.ndarray


def _edge_profile(
    shape: np.ndarray, start: np.ndarray, stop: np.ndarray, radius: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    edge = stop - start
    length = float(np.linalg.norm(edge))
    inward = np.float32([-edge[1], edge[0]]) / length
    rows = np.linspace(start + .10 * edge, stop - .10 * edge, round(.80 * length) + 1, dtype=np.float32)
    steps = np.arange(-radius, radius + 1, dtype=np.float32)
    grid = rows[:, None] + steps[None, :, None] * inward
    run = cv2.remap(shape, grid[..., 0].copy(), grid[..., 1].copy(), cv2.INTER_NEAREST) > 0
    valid = ~run[:, 0] & run[:, -1]
    if valid.mean() < .97:
        return None
    selected = np.flatnonzero(valid)
    offsets = steps[run[selected].argmax(axis=1)]
    return rows[selected] + offsets[:, None] * inward, inward, np.interp(np.arange(len(rows)), selected, offsets)


def _repeated_cut(profile: np.ndarray, short_side: float) -> bool:
    positions = np.linspace(-1, 1, len(profile))
    residuals = profile - np.polyval(np.polyfit(positions, profile, 2), positions)
    spread = float(np.diff(np.percentile(residuals, [5, 95]))[0])
    if spread < 3.5:
        return False
    frequencies = np.fft.rfftfreq(len(profile))
    spectrum = np.abs(np.fft.rfft(residuals * np.hanning(len(profile)))) ** 2
    band = (frequencies >= 1 / min(140, .15 * short_side)) & (frequencies <= 1 / max(7, .01 * short_side))
    peak = int(np.argmax(np.where(band, spectrum, 0)))
    if not band[peak]:
        return False
    width = max(1, int(np.ceil(.18 * peak)))
    if spectrum[max(1, peak - width):peak + width + 1].sum() < .20 * spectrum[1:].sum():
        return False
    # A local tear can yield a spectral peak without a repeated cut along
    # the side. The periodic component needs support in separated thirds.
    return all(np.ptp(part) >= .52 * spread for part in np.array_split(residuals, 3))


def _verify_perimeter(
    image: np.ndarray, shape: np.ndarray, quad: np.ndarray, origin: np.ndarray,
    backing: np.ndarray, short_side: float,
) -> np.ndarray | None:
    radius = int(np.clip(round(.028 * short_side), 10, 70))
    unit = float(np.clip(.006 * short_side, 2.0, 11.0))
    repeated = 0
    colors_by_side = []
    for start, stop in zip(quad, np.roll(quad, -1, axis=0), strict=True):
        edge = _edge_profile(shape, start, stop, radius)
        if edge is None:
            return None
        points, inward, profile = edge
        positions = points[::max(1, len(points) // 144)] + origin
        samples = positions[:, None] + np.float32([-1.8, -.95, 1.8, 2.6, 3.4, 4.2])[None, :, None] * unit * inward
        if np.any((samples < 0) | (samples > np.array(image.shape[1::-1]) - 1)):
            return None
        colors = _sample_lab(image, samples.reshape(-1, 2)).reshape(-1, 6, 3)
        outside, inside = colors[:, :2], colors[:, 2:]
        exterior = np.linalg.norm(outside - backing, axis=2).max(axis=1) <= 12
        # The cut edge itself can be dark where the exposed fibres cast a
        # narrow shadow. Confirm a continuous material band behind that edge.
        color = next((sample for index in range(3)
                      if (sample := _continuous_stock_band(inside[:, index:index + 2], exterior, backing)) is not None), None)
        if color is None:
            return None
        colors_by_side.append(color)
        repeated += _repeated_cut(profile, short_side)
    return np.asarray(colors_by_side) if repeated >= 2 else None


def _continuous_stock_band(interior: np.ndarray, exterior: np.ndarray, backing: np.ndarray) -> np.ndarray | None:
    stock = np.median(interior, axis=(0, 1))
    coherent = np.linalg.norm(interior - stock, axis=2).max(axis=1) <= 16
    if _is_pale_stock(stock) and np.linalg.norm(stock - backing) >= 16 and np.mean(exterior & coherent) >= .73:
        return stock
    return None


def possible_layer_edge(image: np.ndarray, inner: np.ndarray) -> bool:
    """Detect a shadow on pale material directly outside a nested candidate."""
    strength = []
    for start, stop in zip(inner, np.roll(inner, -1, axis=0), strict=True):
        band = _edge_band(image, start, stop, [-14, -10, -8, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5])
        if band is None:
            return True
        near = (band[:, 3, 0] + band[:, 6, 0]) * .5
        narrow = band[:, 4:10, 0].min(axis=1)
        inside = band[:, 10:, 0].mean(axis=1)
        wide = _is_shadowed_stock(band[:, :2], near)
        trough = (inside - narrow >= 7) & (
            _is_shadowed_stock(band[:, :2], narrow) | _is_shadowed_stock(band[:, 2:4], narrow)
        )
        strength.append(float(np.mean(wide | trough)))
    return max(strength) >= .60 or sum(value >= .30 for value in strength) >= 2


def _is_shadowed_stock(pair: np.ndarray, transition: np.ndarray) -> np.ndarray:
    stock = pair.mean(axis=1)
    pale = (stock[:, 0] >= 185) & (stock[:, 0] <= 239) & (np.max(np.abs(stock[:, 1:] - 128), axis=1) <= 21)
    stable = np.linalg.norm(pair[:, 0] - pair[:, 1], axis=1) <= 8
    return pale & stable & (stock[:, 0] - transition >= 13)


def measure_scalloped_print(image: np.ndarray, seed: np.ndarray, backing: np.ndarray | None = None) -> ScallopedPrint | None:
    """Return a native print contour only when white stock and a repeated cut agree."""
    height, width = image.shape[:2]
    short_side = float(np.linalg.norm(np.roll(seed, -1, axis=0) - seed, axis=1).min())
    if short_side < 150:
        return None
    backing = _backing_color(image) if backing is None else backing
    if backing[0] < 245 or np.linalg.norm(backing[1:] - 128) > 5:
        return None
    pad = round(.11 * short_side)
    origin = np.maximum(np.floor(seed.min(axis=0)).astype(int) - pad, 0)
    end = np.minimum(np.ceil(seed.max(axis=0)).astype(int) + pad + 1, (width, height))
    left, top = origin
    right, bottom = end
    if (right - left) * (bottom - top) > 8_000_000:
        return None
    region = image[top:bottom, left:right]
    lab = cv2.cvtColor(region, cv2.COLOR_BGR2LAB)
    foreground = cv2.inRange(lab, (0, 0, 0), (int(backing[0]) - 12, 255, 255))
    foreground = cv2.morphologyEx(foreground, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(foreground, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    center = tuple(float(value) for value in seed.mean(axis=0) - origin)
    contour = next((points for points in contours if cv2.pointPolygonTest(points, center, False) > 0), None)
    if contour is None or len(contour) < 16:
        return None
    local = contour.reshape(-1, 2).astype(np.float32)
    if np.any(local.min(axis=0) < 3) or np.any(local.max(axis=0) >= end - origin - 3):
        return None
    quad, _ = quad_from_polygon(local)
    seed_area = cv2.contourArea(seed)
    overlap, _ = cv2.intersectConvexConvex(seed - origin.astype(np.float32), quad)
    if overlap / (seed_area + cv2.contourArea(quad) - overlap) < .80:
        return None
    shape = cv2.drawContours(np.zeros(region.shape[:2], np.uint8), [contour], -1, 255, cv2.FILLED)
    colors = _verify_perimeter(image, shape, quad, origin, backing, short_side)
    if colors is None:
        return None
    outline = cv2.approxPolyDP(contour, .50, True).reshape(-1, 2).astype(np.float32)
    return ScallopedPrint(outline + origin.astype(np.float32), quad + origin.astype(np.float32), colors)


def suggest_scalloped_motif(
    image: np.ndarray, quad: np.ndarray, outline: np.ndarray, stock_colors: np.ndarray,
) -> np.ndarray | None:
    """Find a straight motif behind the measured paper using the native stock colours."""
    if stock_colors.shape != (4, 3) or not np.isfinite(stock_colors).all():
        return None
    edges = np.roll(quad, -1, axis=0) - quad
    lengths = np.linalg.norm(edges, axis=1)
    short_side = float(lengths.min())
    normals = np.column_stack((-edges[:, 1], edges[:, 0])) / lengths[:, None]
    offsets = np.arange(round(.01 * short_side), round(.13 * short_side))
    if len(offsets) < 16:
        return None
    radius = int(np.clip(round(.01 * short_side), 7, 18))
    lines = []
    for index, (start, stop) in enumerate(zip(quad, np.roll(quad, -1, axis=0), strict=True)):
        band = _edge_band(image, start, stop, offsets.tolist(), fractions=np.linspace(.16, .84, 112, dtype=np.float32))
        if band is None:
            return None
        distances = np.linalg.norm(band - stock_colors[index], axis=2)
        windows = np.lib.stride_tricks.sliding_window_view(distances, 8, axis=1)
        departures = (windows[..., :3].max(axis=2) <= 15) & (windows[..., 5:].min(axis=2) >= 16)
        votes = np.convolve(departures.sum(axis=0), np.ones(radius + 1), mode="same")
        location = int(np.argmax(votes))
        if votes[location] < .45 * len(band):
            return None
        depth = np.float32(offsets[location + 4]) * normals[index]
        line, evidence = _side_line(image, start + depth, stop + depth, radius)
        if line is None or evidence["support"] < .76 or evidence["corner_support"] < .56:
            return None
        lines.append(line)
    intersections = np.cross(np.roll(lines, 1, axis=0), lines)
    if np.any(np.abs(intersections[:, 2]) < 1e-6):
        return None
    motif = (intersections[:, :2] / intersections[:, 2:3]).astype(np.float32)
    if not cv2.isContourConvex(motif) or cv2.contourArea(motif, oriented=True) <= 0:
        return None
    for index, (start, stop) in enumerate(zip(motif, np.roll(motif, -1, axis=0), strict=True)):
        points = np.linspace(start, stop, 48)
        if any(cv2.pointPolygonTest(outline, tuple(point), True) < 2 for point in points):
            return None
        margin = float(np.clip(.008 * short_side, 5, 13))
        stock = _edge_band(image, start, stop, [-1.5 * margin, -margin])
        if stock is None or np.mean(np.linalg.norm(stock - stock_colors[index], axis=2).max(axis=1) <= 17) < .76:
            return None
    return motif
