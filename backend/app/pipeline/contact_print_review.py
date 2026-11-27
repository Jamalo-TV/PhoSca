"""Measure source edges around a printed grid for a held crop comparison."""

import cv2
import numpy as np

from app.pipeline.geometry import normalized_polygon_overlap
from app.pipeline.multipanel_prints import _native_perforations_supported, regular_perforation_grid
from app.pipeline.print_edges import _edge_band
from app.pipeline.print_outline_suggestion import _outline_at_distance


CONTACT_PRINT_EDGE_SCOPE = "contact_print_edge_scope"
CONTACT_PRINT_EDGE_CHECK = "contact_print_edge_check_required"


def _stable_chroma_outline(image: np.ndarray, seed: np.ndarray) -> np.ndarray | None:
    shortest = float(np.linalg.norm(seed - np.roll(seed, -1, axis=0), axis=1).min())
    if shortest < 450:
        return None
    scale = np.float32(image.shape[1::-1]) - 1
    padding = .25 * shortest
    low = np.maximum(0, np.floor(seed.min(axis=0) - padding).astype(int))
    high = np.minimum(scale, np.ceil(seed.max(axis=0) + padding)).astype(int) + 1
    patch = image[low[1]:high[1], low[0]:high[0]]
    if patch.shape[0] * patch.shape[1] > 1_200_000:
        return None
    lab = cv2.cvtColor(patch, cv2.COLOR_BGR2LAB).astype(np.float32)
    chroma = lab[..., 1:]
    interior = cv2.fillConvexPoly(np.zeros(lab.shape[:2], np.uint8), np.rint(seed - low).astype(np.int32), 1)
    pale = (interior != 0) & (lab[..., 0] >= 180) & (np.linalg.norm(chroma - 128, axis=2) <= 16)
    if np.count_nonzero(pale) < .10 * cv2.contourArea(seed):
        return None
    distance = cv2.medianBlur(np.linalg.norm(chroma - np.median(chroma[pale], axis=0), axis=2), 5)
    candidates = [_outline_at_distance(distance, low, seed, scale, limit, max_area_ratio=1.40, minimum_iou=.74)
                  for limit in (8, 10, 12)]
    if any(candidate is None for candidate in candidates):
        return None
    left, central, right = candidates
    return central if max(np.linalg.norm(central - other, axis=1).max() for other in (left, right)) <= 5 else None


def _chroma_line(image: np.ndarray, start: np.ndarray, end: np.ndarray) -> np.ndarray | None:
    stations = np.linspace(.06, .94, 360, dtype=np.float32)
    depths = np.arange(-14, 14.01, .5, dtype=np.float32)
    band = _edge_band(image, start, end, depths, fractions=stations)
    if band is None:
        return None
    chroma = band[..., 1:].astype(np.float32)
    inside = np.median(chroma[:, (depths >= 6) & (depths <= 11)], axis=1)
    outside = np.median(chroma[:, (depths >= -13) & (depths <= -8)], axis=1)
    step = inside - outside
    delta = np.median(step, axis=0)
    magnitude = np.linalg.norm(delta)
    stock = np.median(inside, axis=0)
    if magnitude < 13 or np.linalg.norm(stock - 128) > 17:
        return None
    direction = delta / magnitude
    plateaus = (step @ direction >= 10) & (np.linalg.norm(inside - stock, axis=1) <= 10)
    if plateaus.mean() < .86 or min(block.mean() for block in np.array_split(plateaus, 8)) < .65:
        return None

    values = chroma @ direction
    # Use the half-height crossing between the two observed chroma plateaus.
    search = np.flatnonzero((depths >= -7) & (depths < 5))
    relative = (values[:, search] - (outside @ direction)[:, None]) / np.maximum(1, (step @ direction)[:, None])
    error = abs(relative - .5)
    stations_idx = np.arange(len(stations))
    best = error.argmin(axis=1)
    columns = search[best]
    narrow = values[stations_idx, columns + 4] - values[stations_idx, columns - 4]
    crossing = plateaus & (error[stations_idx, best] <= .29)
    if crossing.mean() < .83:
        return None
    supported = crossing & (narrow >= 6)
    if supported.mean() < .80:
        return None
    fit = np.polyfit(stations[supported], depths[columns[supported]], 1)
    supported &= abs(np.polyval(fit, stations) - depths[columns]) <= 2.2
    if supported.mean() < .78 or min(block.mean() for block in np.array_split(supported, 8)) < .55:
        return None
    offsets = np.polyval(np.polyfit(stations[supported], depths[columns[supported]], 1), [0, 1])
    if np.max(np.abs(offsets)) > 5.5:
        return None
    edge = end - start
    inward = np.float32([-edge[1], edge[0]]) / np.linalg.norm(edge)
    first, last = np.float32([start, end]) + offsets[:, None] * inward
    return np.cross(np.r_[first, 1.], np.r_[last, 1.])


def _continuous_pale_perimeter(image: np.ndarray, outline: np.ndarray) -> bool:
    for start, end in zip(outline, np.roll(outline, -1, axis=0), strict=True):
        length = float(np.linalg.norm(end - start))
        fractions = np.linspace(.012, .988, round(length / .7), dtype=np.float32)
        band = _edge_band(image, start, end, [1.5, 2.5, 3.5, 5., 6.5, 8.], fractions=fractions)
        if band is None:
            return False
        stock = np.median(band[:, 1:], axis=(0, 1))
        distance = np.linalg.norm(band[..., 1:] - stock[1:], axis=2)
        light = np.median(band[..., 0], axis=1)
        matched = (distance <= 11).mean(axis=1) >= .60
        interrupted = (distance[:, :4] > 10).all(axis=1) | ((light < stock[0] - 25) & (distance[:, 1:] > 6).all(axis=1))
        if (matched.mean() < .985 or np.quantile(light, .005) < 175
                or np.any(interrupted[:-1] & interrupted[1:])):
            return False
    return True


def _positive_grid_on_stock(image: np.ndarray, outline: np.ndarray) -> bool:
    lengths = np.linalg.norm(outline - np.roll(outline, -1, axis=0), axis=1)
    if lengths[::2].mean() > lengths[1::2].mean():
        outline = np.roll(outline, -1, axis=0)
    width, height = 800, 1040
    start, end = 120, 680
    target = np.float32([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])
    matrix = cv2.getPerspectiveTransform(outline, target)
    aligned = cv2.warpPerspective(image, matrix, (width, height))
    light = cv2.cvtColor(aligned[:, start:end], cv2.COLOR_BGR2LAB)[..., 0]
    grid = regular_perforation_grid(light)
    if grid is None:
        return False
    # A sleeve can have exactly the same perforations. Its grille alone is not a print cue.
    centers, pitches, polarities = grid
    if np.mean(polarities >= 8) < .80 or np.median(polarities) < 18 or min(polarities) < -10:
        return False
    from_grid = np.linalg.inv(matrix)
    return all(_native_perforations_supported(image, from_grid, start, end, row, int(pitch))
               for row, pitch in zip(centers, pitches, strict=True))


def measure_chroma_outer_outline(image: np.ndarray, seed: np.ndarray, current: np.ndarray) -> np.ndarray | None:
    """Refit a larger color-supported contour in the source, independent of its material."""
    proposal = _stable_chroma_outline(image, seed)
    if proposal is None:
        return None
    lines = [_chroma_line(image, a, b) for a, b in zip(proposal, np.roll(proposal, -1, axis=0), strict=True)]
    if any(line is None for line in lines):
        return None
    corners = np.cross(np.roll(np.asarray(lines), 1, axis=0), lines)
    if np.any(np.abs(corners[:, 2]) < 1e-6):
        return None
    outline = np.float32(corners[:, :2] / corners[:, 2:])
    scale = np.float32(image.shape[1::-1]) - 1
    if (not cv2.isContourConvex(outline) or cv2.contourArea(outline, oriented=True) <= 0
            or np.any((outline < 0) | (outline > scale))
            or np.max(np.linalg.norm(outline - proposal, axis=1)) > 8):
        return None
    match = normalized_polygon_overlap(current / scale, outline / scale, raster_size=1024)
    return outline if .74 <= match.iou <= .94 and match.first_containment >= .985 else None


def contact_print_review_outline(image: np.ndarray, seed: np.ndarray, current: np.ndarray) -> np.ndarray | None:
    """Return a source comparison scope for a held crop, with no ownership decision."""
    outline = measure_chroma_outer_outline(image, seed, current)
    if outline is None:
        return None
    if not _continuous_pale_perimeter(image, outline) or not _positive_grid_on_stock(image, outline):
        return None
    return outline
