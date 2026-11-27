"""Locate an album boundary beneath a frame-wide background proposal."""

from itertools import combinations

import cv2
import numpy as np

from app.pipeline.crop_sampling import _outline_mask
from app.pipeline.geometry import order_quad_cyclic, quad_from_polygon
from app.pipeline.near_capture_review import quarter_turn
from app.pipeline.print_edges import refine_print_boundary


CAPTURE_BACKGROUND_COUNT_REVIEW = "CAPTURE_BACKGROUND_COUNT_REVIEW"
ALBUM_BOUNDARY_TRACES = "album_boundary_traces"
CAPTURE_BACKGROUND_REVIEW_REVISION = 1


def _frame_strip(polygon: np.ndarray, width: int, height: int) -> bool:
    left, top = polygon.min(axis=0)
    right, bottom = polygon.max(axis=0)
    margin = max(3, width * .012)
    return bool(left <= margin and right >= width - 1 - margin and top <= margin
                and .07 * height <= bottom <= .29 * height and bottom - top <= .27 * width)


def _binding_support(image: np.ndarray, bottom: float) -> tuple[np.ndarray, np.ndarray] | None:
    height, width = image.shape[:2]
    step = max(.65, width / 1100.)
    margin = int(np.ceil(26 * step))
    low = max(margin, round(bottom - .10 * width))
    high = min(height - margin, round(bottom + .06 * width))
    if high - low < 7:
        return None
    columns = np.unique(np.rint(np.linspace(.045 * width, .955 * width, 260)).astype(int))
    band = cv2.cvtColor(image[low - margin:high + margin], cv2.COLOR_BGR2LAB)
    rows = np.arange(high - low) + margin

    def levels(offsets: tuple[int, ...]) -> np.ndarray:
        offsets_px = np.rint(np.array(offsets) * step).astype(int)
        return band[rows[:, None, None] + offsets_px[None, :, None], columns].mean(axis=1)

    outside, binding, inside = levels((-14, -10, -7)), levels((0, 2, 4)), levels((16, 20, 24))
    redness = binding[..., 1] - np.maximum(outside[..., 1], inside[..., 1])
    light_change = inside[..., 0] - outside[..., 0]
    matching = ((binding[..., 1] >= 153) & (redness >= 15) & (inside[..., 0] >= 145)
                & (light_change >= 46) & (binding[..., 0] + 15 <= inside[..., 0]))
    scores = np.where(matching, redness + .35 * light_change, -1)
    locations = scores.argmax(axis=0)
    usable = scores[locations, np.arange(len(columns))] >= 0
    if usable.mean() < .81:
        return None
    chosen = np.column_stack((columns, low + locations)).astype(np.float32)
    vx, vy, x0, y0 = cv2.fitLine(chosen[usable], cv2.DIST_WELSCH, 0, .01, .01).ravel()
    if abs(vx) < .98:
        return None
    slope = vy / vx
    fitted = y0 + (columns - x0) * slope
    usable &= np.abs(chosen[:, 1] - fitted) <= max(2.5, 3.5 * step)
    if usable.mean() < .81 or min(chunk.mean() for chunk in np.array_split(usable, 4)) < .70:
        return None
    return np.float32([slope, y0 - x0 * slope]), chosen[usable]


def _has_spiral_binding(image: np.ndarray, line: np.ndarray) -> bool:
    height, width = image.shape[:2]
    step = max(.65, width / 1100.)
    diameter = max(9, round(19 * step) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (diameter, diameter))
    rail_y = float(np.polyval(line, width / 2))
    stop = min(round(.78 * height), round(rail_y + .49 * height))
    for left, right in ((0, round(.16 * width)), (round(.84 * width), width)):
        start = round(np.polyval(line, (left + right) / 2) + 14 * step)
        if stop <= start:
            continue
        gray = cv2.cvtColor(image[start:stop, left:right], cv2.COLOR_BGR2LAB)[..., 0]
        contrast = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
        dark = np.uint8((contrast >= 33) & (gray <= 151))
        _, _, stats, centers = cv2.connectedComponentsWithStats(dark)
        sizes = stats[:, [cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT]]
        area = stats[:, cv2.CC_STAT_AREA]
        components = ((sizes >= 3 * step) & (sizes <= 20 * step)).all(axis=1)
        components &= (area >= max(6, 9 * step**2)) & (area <= 190 * step**2)
        points = centers[components] + [left, start]
        if not 13 <= len(points) <= 128:
            continue
        points = points[np.argsort(points[:, 1])]
        for first, last in combinations(points, 2):
            if first[1] > np.polyval(line, first[0]) + .065 * height:
                break
            dx, dy = last - first
            if dy < .18 * height or abs(dx) > .20 * dy:
                continue
            on_line = np.abs(points[:, 0] - first[0] - (points[:, 1] - first[1]) * dx / dy) <= max(2.5, 4 * step)
            gaps = np.diff(points[on_line, 1])
            plausible = (gaps >= max(10, 14 * step)) & (gaps <= 42 * step)
            if plausible.sum() < 12:
                continue
            spacing = np.median(gaps[plausible])
            regular = np.flatnonzero(plausible & (np.abs(gaps - spacing) <= max(2.5, .18 * spacing)))
            if len(regular) >= 12 and any(len(run) >= 8 for run in np.split(regular, np.flatnonzero(np.diff(regular) > 1) + 1)):
                return True
    return False


def _crosses_binding(polygon: np.ndarray, shape: tuple[int, int], line: np.ndarray) -> bool:
    height, width = shape
    scale = max(.65, width / 1100.)
    rows, columns = np.nonzero(_outline_mask(polygon, (min(height, int(polygon[:, 1].max()) + 2), width)))
    if not len(rows):
        return True
    depth = rows - np.polyval(line, columns)
    return np.mean(depth <= -5 * scale) < .92 or np.mean(depth >= 15 * scale) > .003


def _has_outboard_edge(image: np.ndarray, line: np.ndarray) -> bool:
    width = image.shape[1]
    step = max(.65, width / 1100.)
    columns = np.linspace(.07 * width, .925 * width, 192, dtype=np.float32)
    center = np.polyval(line, columns)
    offsets = np.arange(-39., -7.1, .5, dtype=np.float32) * step
    profile = np.float32([-3., -1.1, 0., 1.1, 3.]) * step
    low = max(0, int(center.min() - 45 * step))
    high = min(image.shape[0], int(np.ceil(center.max())) + 1)
    band = cv2.cvtColor(image[low:high], cv2.COLOR_BGR2LAB).astype(np.float32)
    drifts = np.linspace(-1.5 * step, 1.5 * step, len(columns), dtype=np.float32)
    for drift in (-drifts, np.zeros_like(drifts), drifts):
        rows = center + offsets[:, None, None] + profile[None, :, None] + drift - low
        valid = (rows.min(axis=(1, 2)) >= 0) & (rows.max(axis=(1, 2)) <= len(band) - 1)
        if not valid.any():
            continue
        rows = rows[valid].reshape(-1, len(columns))
        samples = cv2.remap(band, np.broadcast_to(columns, rows.shape).copy(), rows, cv2.INTER_LINEAR)
        samples = samples.reshape(-1, len(profile), len(columns), 3)
        # A paper edge can appear as a colour step or a narrow band.
        for delta in (samples[:, 1] - samples[:, 3], samples[:, 2] - .5 * (samples[:, 0] + samples[:, 4])):
            median = np.median(delta, axis=1)
            strength = np.linalg.norm(median, axis=1)
            projection = (delta * median[:, None]).sum(axis=2) / np.maximum(1., strength[:, None])
            supported = projection >= 4
            weakest_quarter = np.min([chunk.mean(axis=1) for chunk in np.array_split(supported, 4, axis=1)], axis=0)
            agreement = (.7 * supported.mean(axis=1) + .3 * weakest_quarter) * np.minimum(strength / 9, 1)
            if np.any(agreement >= .59):
                return True
    return False


def _inboard_print_edges(image: np.ndarray, contour: np.ndarray) -> bool:
    quad = order_quad_cyclic(quad_from_polygon(contour)[0])
    radius = int(np.clip(.02 * image.shape[1], 6, 32))
    _, sides, _ = refine_print_boundary(image, quad, search_radius=radius)
    supported = [edge["support"] >= .89 and (edge.get("corner_support") or 0) >= .74
                 and not edge["ambiguous"] for edge in sides.values()]
    return any(supported[index - 1] and side for index, side in enumerate(supported))


def album_binding_traces(image: np.ndarray, candidate: np.ndarray, peers: list[np.ndarray]) -> list[np.ndarray]:
    """Return only measured binding spans. All contours and returned spans are normalized."""
    if len(peers) < 2 or min(image.shape[:2]) < 160 or image.shape[0] * image.shape[1] > 12_000_000:
        return []
    for turn in range(4):
        page = np.rot90(image, turn)
        height, width = page.shape[:2]
        scale = np.float32([width - 1, height - 1])
        polygon = quarter_turn(candidate, turn) * scale
        if _frame_strip(polygon, width, height):
            break
    else:
        return []
    witness = _binding_support(page, polygon[:, 1].max())
    if witness is None:
        return []
    line, points = witness
    if (_crosses_binding(polygon, (height, width), line) or not _has_spiral_binding(page, line)
            or _has_outboard_edge(page, line)):
        return []
    inside = []
    for peer in peers:
        contour = quarter_turn(peer, turn) * scale
        if np.min(contour[:, 1] - np.polyval(line, contour[:, 0])) >= max(7, .014 * width):
            if _inboard_print_edges(page, contour):
                center = (contour[:, 0].min() + contour[:, 0].max()) / 2
                inside.append(center)
                if np.ptp(inside) >= .20 * width:
                    break
    if len(inside) < 2 or np.ptp(inside) < .20 * width:
        return []
    # Omit gaps in the measured binding. Endpoints are actual source observations.
    breaks = np.flatnonzero(np.diff(points[:, 0]) > .02 * width) + 1
    return [quarter_turn(run[[0, -1]] / scale, (-turn) % 4)
            for run in np.split(points, breaks) if len(run) > 1 and np.ptp(run[:, 0]) >= .08 * width]
