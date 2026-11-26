"""Find a provisional print outline after the normal detector finds no print."""

from collections.abc import Iterator
from itertools import combinations, islice, product

import cv2
import numpy as np

from app.pipeline.adjacent_prints import _joined_segments
from app.pipeline.crossing_prints import _local_side_support, _native_line
from app.pipeline.edge_prints import _photo_texture
from app.pipeline.print_edges import _edge_band, refine_print_boundary


PROJECTIVE_PRINT_REVISION = 1
PROJECTIVE_PRINT_PROPOSAL = "possible_projective_print"
PROJECTIVE_PRINT_OUTLINE_REVIEW = "PROJECTIVE_PRINT_OUTLINE_REVIEW"


def _span_coverage(start: np.ndarray, end: np.ndarray, segment: np.ndarray) -> float:
    length = float(np.linalg.norm(end - start))
    positions = np.sort((segment - start) @ ((end - start) / length))
    return max(0., min(length, float(positions[1])) - max(0., float(positions[0]))) / length


def _projective_seeds(image: np.ndarray) -> Iterator[np.ndarray]:
    height, width = image.shape[:2]
    scale = min(1., 1280 / max(height, width))
    if min(height, width) < 640:
        return
    reduced = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else image
    segments = _joined_segments(reduced, min(reduced.shape[:2]))
    if len(segments) > 40:
        return
    pairs = []
    for axis in (0, 1):
        endpoints = [segment.ends / scale for segment in segments if segment.axis == axis]
        endpoints.sort(key=lambda end: float(end[:, 1 - axis].mean()))
        pairs.append(combinations(((end, _native_line(end)) for end in endpoints), 2))
    for ((top, north), (bottom, south)), ((left, west), (right, east)) in product(*pairs):
        sides = np.float32([north, east, south, west])
        intersections = np.cross(np.roll(sides, 1, axis=0), sides)
        if np.min(np.abs(intersections[:, 2])) < .75:
            continue
        quad = intersections[:, :2] / intersections[:, 2, None]
        if (np.any(quad < 12) or np.any(quad > [width - 13, height - 13])
                or not cv2.isContourConvex(quad) or not .075 <= cv2.contourArea(quad) / (height * width) <= .27):
            continue
        ends = np.roll(quad, -1, axis=0)
        lengths = np.linalg.norm(ends - quad, axis=1)
        if not .52 <= lengths[::2].sum() / lengths[1::2].sum() <= 1.95:
            continue
        coverage = [_span_coverage(start, end, segment)
                    for start, end, segment in zip(quad, ends, (top, right, bottom, left), strict=True)]
        if min(coverage) >= .50 and max(coverage) >= .85 and np.median(coverage) >= .74:
            yield quad


def _paper_contacts(image: np.ndarray, quad: np.ndarray) -> bool:
    pale_sides = 0
    for start, end in zip(quad, np.roll(quad, -1, axis=0), strict=True):
        band = _edge_band(image, start, end, [-14, -9, -4, 4, 9, 14],
                          fractions=np.linspace(.05, .95, 104, dtype=np.float32))
        if band is None:
            return False
        exterior, interior = (np.median(part, axis=1) for part in np.split(band, 2, axis=1))
        if np.mean(np.linalg.norm(exterior - interior, axis=1) >= 35) < .53:
            return False
        pale = (exterior[:, 0] >= 167) & (np.linalg.norm(exterior[:, 1:] - 128, axis=1) < 22)
        pale_sides += pale.mean() >= .66
    return pale_sides >= 3


def _terminal_arm(image: np.ndarray, first: np.ndarray, last: np.ndarray, *, at_start: bool) -> bool:
    length = float(np.linalg.norm(last - first))
    positions = np.float32([-30, -24, -18, -12, 7, 11, 16, 22, 32]) / length
    if not at_start:
        positions = 1 - positions
    band = _edge_band(image, first, last, list(range(-14, 15)), fractions=positions)
    if band is None:
        return False
    light = band[..., 0].astype(np.float32)
    pale = light[:, :-4] >= 158
    contrast = np.where(pale, light[:, :-4] - light[:, 4:], 0).max(axis=1)
    inside, beyond = contrast[4:], contrast[:4]
    return bool(np.count_nonzero(inside >= 13) >= 4 and np.median(inside) >= 18 and np.median(beyond) < 12)


def projective_print_outline(image: np.ndarray) -> np.ndarray | None:
    """Find one editable outline supported by native source lines and paper ends."""
    seeds = list(islice(_projective_seeds(image), 17))
    if len(seeds) > 16:
        return None
    outlines = []
    for seed in seeds:
        if not _paper_contacts(image, seed) or not _photo_texture(image, seed):
            continue
        quad, sides, _ = refine_print_boundary(image, seed, search_radius=16)
        if quad is None:
            continue
        ends = np.roll(quad, -1, axis=0)
        if not all(
            (not evidence["ambiguous"] and evidence["support"] >= .88)
            or _local_side_support(image, start, end, 11)
            for (start, end), evidence in zip(zip(quad, ends, strict=True), sides.values(), strict=True)
        ):
            continue
        # A projected intersection can be a T-junction or a rail extending
        # into the background. Require source terminations around a full side.
        corners = [
            _terminal_arm(image, quad[index - 1], corner, at_start=False)
            and _terminal_arm(image, corner, ends[index], at_start=True)
            for index, corner in enumerate(quad)
        ]
        if not any(a and b for a, b in zip(corners, [*corners[1:], corners[0]], strict=True)):
            continue
        if any(np.linalg.norm(quad - prior, axis=1).max() <= 5 for prior in outlines):
            continue
        outlines.append(quad)
    return outlines[0] if len(outlines) == 1 else None
