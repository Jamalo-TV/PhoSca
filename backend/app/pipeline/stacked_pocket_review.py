"""Locate pocket bands outside two distinct photo proposals for count review."""

import cv2
import numpy as np

from app.pipeline.geometry import normalized_polygon_overlap, order_quad_cyclic
from app.pipeline.print_edges import _edge_band


STACKED_POCKET_COUNT_REVIEW = "STACKED_POCKET_COUNT_REVIEW"
STACKED_POCKET_REVIEW_TRACES = "stacked_pocket_review_traces"
STACKED_POCKET_REVIEW_CHILDREN = "stacked_pocket_review_children"
STACKED_POCKET_REVIEW_REVISION = 1


def _has_paper_perimeter(image: np.ndarray, outline: np.ndarray) -> bool:
    quad = order_quad_cyclic(outline)
    short = np.linalg.norm(np.roll(quad, -1, axis=0) - quad, axis=1).min()
    offsets = np.float32([-1.2, -.75, -.40, .40, .75, 1.2]) * max(2., short * .003)
    fractions = np.linspace(.06, .94, 144, dtype=np.float32)
    for start, stop in zip(quad, np.roll(quad, -1, axis=0), strict=True):
        band = _edge_band(image, start, stop, offsets, fractions=fractions)
        if band is None:
            return False
        outside, inside = (np.median(part, axis=1) for part in np.split(band, 2, axis=1))
        strong = np.linalg.norm(inside - outside, axis=1) >= 15
        if strong.mean() < .94 or min(part.mean() for part in np.array_split(strong, 8)) < .82:
            return False
    return True


def _stock_runs(similar: np.ndarray, spacing: float, width: float) -> list[tuple[int, int]]:
    covered = (similar.mean(axis=1) >= .76) & (
        np.min([part.mean(axis=1) for part in np.array_split(similar, 6, axis=1)], axis=0) >= .50
    )
    runs = np.flatnonzero(np.diff(np.r_[False, covered, False])).reshape(-1, 2)
    joined: list[tuple[int, int]] = []
    for start, stop in runs:
        if (stop - start) * spacing < .0055 * width:
            continue
        if joined and (start - joined[-1][1]) * spacing < .012 * width:
            joined[-1] = (joined[-1][0], stop)
        else:
            joined.append((start, stop))
    return [part for part in joined if (part[1] - part[0]) * spacing <= .075 * width]


def _stock_palette(patch: np.ndarray) -> np.ndarray | None:
    warm = (patch[..., 0] >= 155) & (patch[..., 1] >= 129) & (patch[..., 2] >= 134)
    if warm.mean() < .15:
        return None
    colors = patch[warm, 1:]
    buckets = np.uint8(np.clip(colors - 118, 0, 59)) // 6
    a, b = np.divmod(np.bincount(buckets[:, 0] * 10 + buckets[:, 1], minlength=100).argmax(), 10)
    center = np.float32([a, b]) * 6 + 121
    return np.median(colors[(np.abs(colors - center) <= 8).all(axis=1)], axis=0)


def _has_fine_detail(
    image: np.ndarray, tangent: np.ndarray, axis: np.ndarray, columns: np.ndarray, rows: np.ndarray, slope: float,
) -> bool:
    width = float(columns[-1] - columns[0])
    height = float(rows[-1] - rows[0])
    lateral = np.linspace(columns[0] + .05 * width, columns[-1] - .05 * width, 960, dtype=np.float32)
    along = np.linspace(rows[0] + .15 * height, rows[-1] - .15 * height,
                        int(np.clip(height / width * 1040, 25, 160)), dtype=np.float32)
    depth = along[:, None] + slope * (lateral - columns.mean())
    points = lateral[None, :, None] * tangent + depth[..., None] * axis
    patch = cv2.remap(image, points[..., 0].copy(), points[..., 1].copy(), cv2.INTER_LINEAR)
    light = cv2.cvtColor(patch, cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32)
    residual = np.abs(light - cv2.GaussianBlur(light, (0, 0), 1.3))
    return sum(np.percentile(part, 80) >= 3 for part in np.array_split(residual, 12, axis=1)) >= 10


def _pocket_bands(
    image: np.ndarray, tangent: np.ndarray, axis: np.ndarray, lateral: tuple[float, float], depth: tuple[float, float],
) -> list[np.ndarray]:
    start, end = lateral
    top, bottom = depth
    width = end - start
    if not .20 * width < bottom - top < .90 * width:
        return []
    columns = np.linspace(start, end, 156, dtype=np.float32)
    rows = np.linspace(top, bottom, int(np.clip(round((bottom - top) * 1.05), 145, 1360)), dtype=np.float32)
    spacing = (bottom - top) / (len(rows) - 1)
    centered = columns - columns.mean()
    padding = int(np.ceil(.076 * np.max(np.abs(centered)) / spacing)) + 5
    depths = top + np.arange(-padding, len(rows) + padding, dtype=np.float32) * spacing
    points = columns[None, :, None] * tangent + depths[:, None, None] * axis
    if np.any((points < 0) | (points > np.float32(image.shape[1::-1]) - 1)):
        return []
    patch = cv2.remap(image, points[..., 0].copy(), points[..., 1].copy(), cv2.INTER_LINEAR)
    lab = cv2.cvtColor(patch, cv2.COLOR_BGR2LAB).astype(np.float32)
    palette = _stock_palette(lab[padding:padding + len(rows)])
    if palette is None:
        return []
    stock = (np.linalg.norm(lab[..., 1:] - palette, axis=2) <= 9) & (lab[..., 0] >= 155)
    scan_rows = np.arange(len(rows))[:, None] + padding
    scan_columns = np.arange(len(columns))
    best = (0, 0, 0.)
    traces = []
    best_gaps = []
    best_slope = 0.
    for slope in np.linspace(-.075, .075, 31, dtype=np.float32):
        indices = scan_rows + np.rint(slope * centered / spacing).astype(int)
        similar = stock[indices, scan_columns]
        runs = _stock_runs(similar, spacing, width)
        if len(runs) < 3:
            continue
        gap_borders = set()
        gaps = []
        for index, ((_, lower), (upper, _)) in enumerate(zip(runs, runs[1:])):
            if not .026 * width <= (upper - lower) * spacing <= .28 * width:
                continue
            strip = lab[indices[lower:upper], scan_columns, 0]
            across = np.diff(np.quantile(strip, [.12, .88], axis=1), axis=0)
            if (similar[lower:upper].mean() <= .58 and np.ptp(np.quantile(strip, [.05, .95])) >= 55
                    and np.median(across) >= 41):
                gaps.append(rows[lower:upper])
                gap_borders.update((index, index + 1))
        if len(gaps) < 2:
            continue
        score = (len(gaps), len(runs), sum(float(similar[a:b].mean()) for a, b in runs))
        if score <= best:
            continue
        # Show three actual runs around the textured intervals. Do not draw
        # across missing stations or extrapolate the rails to either side.
        observed = []
        for index in sorted(gap_borders):
            first, last = runs[index]
            row = first + int(np.argmax(similar[first:last].sum(axis=1)))
            sections = np.flatnonzero(np.diff(np.r_[False, similar[row], False])).reshape(-1, 2)
            begin, end = max(sections, key=lambda pair: pair[1] - pair[0])
            if end - begin >= .40 * len(columns):
                ends = [begin, end - 1]
                observed.append(columns[ends, None] * tangent + (rows[row] + slope * centered[ends, None]) * axis)
        if len(observed) >= 3:
            best = score
            traces = [observed[0], observed[len(observed) // 2], observed[-1]]
            best_gaps = gaps
            best_slope = float(slope)
    # Smooth paint and printed rules can pass the band and contrast check.
    # Require fine texture across the interiors of two separate intervals.
    textured = sum(_has_fine_detail(image, tangent, axis, columns, gap, best_slope) for gap in best_gaps)
    return traces if textured >= 2 else []


def stacked_pocket_review_traces(
    image: np.ndarray, carriers: list[np.ndarray], photos: list[np.ndarray],
) -> list[np.ndarray]:
    """Return measured source marks for two broad, disputed carrier contours."""
    first, second = carriers
    native_size = np.float32(image.shape[1::-1]) - 1
    outer = [outline / native_size for outline in carriers]
    inner = [outline / native_size for outline in photos]
    if (not .49 <= normalized_polygon_overlap(*outer, raster_size=1024).iou <= .98
            or normalized_polygon_overlap(*inner, raster_size=1024).iou > .015):
        return []
    for carrier in outer:
        overlaps = [normalized_polygon_overlap(carrier, photo, raster_size=1024) for photo in inner]
        if (any(row.second_containment < .58 or not .12 <= row.first_containment <= .45 for row in overlaps)
                or not .44 <= sum(row.first_containment for row in overlaps) <= .84):
            return []
    if _has_paper_perimeter(image, first) or _has_paper_perimeter(image, second):
        return []
    a, b = (cv2.convexHull(photo)[:, 0] for photo in photos)
    found = []
    for near, far in ((a, b), (b, a)):
        axis = far.mean(axis=0) - near.mean(axis=0)
        length = np.linalg.norm(axis)
        if length <= 1:
            continue
        axis /= length
        tangent = np.float32([axis[1], -axis[0]])
        left = max((near @ tangent).min(), (far @ tangent).min())
        right = min((near @ tangent).max(), (far @ tangent).max())
        width = right - left
        if width < 240:
            continue
        stop = (near @ axis).min() - .035 * width
        start = max(max((quad @ axis).min() for quad in carriers) - .03 * width, stop - .59 * width)
        lines = _pocket_bands(image, tangent, axis, (left + .035 * width, right - .035 * width), (start, stop))
        if lines and not any(
            cv2.pointPolygonTest(polygon, tuple(point), False) >= 0
            for polygon in (a, b) for line in lines for point in np.linspace(*line, 32)
        ):
            found.append(lines)
    return found[0] if len(found) == 1 else []
