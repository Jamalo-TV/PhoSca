"""Find a separate print beside an observed print on a pale album leaf."""

from dataclasses import dataclass
from itertools import combinations

import cv2
import numpy as np

from app.pipeline.edge_prints import _photo_texture
from app.pipeline.geometry import order_quad_cyclic
from app.pipeline.print_edges import _edge_band, _replace_quad_side, refine_print_boundary


ADJACENT_PRINT_REVISION = 1
ADJACENT_PRINT_REVIEW = "ADJACENT_PRINT_OUTLINE_REVIEW"


@dataclass(frozen=True)
class _Segment:
    ends: np.ndarray
    tangent: np.ndarray
    length: float
    axis: int
    samples: np.ndarray


def _segment(ends: np.ndarray, axis: int, samples: np.ndarray) -> _Segment:
    ends = ends[ends[:, axis].argsort()]
    direction = ends[1] - ends[0]
    length = float(np.linalg.norm(direction))
    return _Segment(ends, direction / length, length, axis, samples)


def _joined_segments(image: np.ndarray, short: int) -> list[_Segment]:
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    detector = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    observations = []
    for index in range(3):
        lines = detector.detect(lab[:, :, index])[0]
        if lines is None:
            continue
        for ends in lines[:, 0].reshape(-1, 2, 2):
            direction = np.abs(ends[1] - ends[0])
            if np.linalg.norm(direction) >= .08 * short and direction.min() < .36 * direction.max():
                observations.append(_segment(ends, int(direction.argmax()), ends))
    if len(observations) > 120:
        return []
    joined: list[_Segment] = []
    for item in sorted(observations, key=lambda line: -line.length):
        for index, previous in enumerate(joined):
            if previous.axis != item.axis:
                continue
            normal = np.float32([-previous.tangent[1], previous.tangent[0]])
            relative = item.ends - previous.ends[0]
            projected = relative @ previous.tangent
            if (np.max(np.abs(relative @ normal)) > 7 or projected.min() > previous.length + .13 * short
                    or projected.max() < -.13 * short):
                continue
            samples = np.concatenate([previous.samples, item.samples])
            vx, vy, x, y = cv2.fitLine(samples, cv2.DIST_HUBER, 0, .01, .01).ravel()
            relative = samples - [x, y]
            if np.quantile(np.abs(relative @ [-vy, vx]), .90) > 5.6:
                continue
            positions = relative @ [vx, vy]
            ends = np.float32([x, y]) + positions[[positions.argmin(), positions.argmax()], None] * [vx, vy]
            joined[index] = _segment(ends.astype(np.float32), item.axis, samples)
            break
        else:
            joined.append(item)
    return [item for item in joined if item.length >= .14 * short]


def _collar_outline(image: np.ndarray, quad: np.ndarray, side: int) -> np.ndarray | None:
    """Check the bright strip between image ink and the edge of the album leaf."""
    start, end = quad[side], quad[(side + 1) % 4]
    positions = np.linspace(.065, .935, 112, dtype=np.float32)
    band = _edge_band(image, start, end, list(range(-38, 29)), fractions=positions)
    if band is None:
        return None
    offsets = np.arange(-11, 17)
    columns = offsets[:, None] + 38
    pale = band[:, columns - [2, 3, 4, 5]].mean(axis=2)
    ink = band[:, columns + [2, 3, 4], 0].mean(axis=2)
    exterior = band[:, columns - [9, 13, 17, 21, 25], 0].min(axis=2)
    drop, rise = pale[:, :, 0] - ink, pale[:, :, 0] - exterior
    supported = (drop >= 19) & (rise >= 16) & (pale[:, :, 0] >= 174) & (np.linalg.norm(pale[:, :, 1:] - 128, axis=2) < 14)
    response = np.where(supported, .65 * drop + .35 * rise, 0)
    best = response.argmax(axis=1)
    found = supported.any(axis=1)
    if found.mean() < .80 or min(part.mean() for part in np.array_split(found, 4)) < .69:
        return None
    direction = end - start
    tangent = direction / np.linalg.norm(direction)
    inward = np.float32([-tangent[1], tangent[0]])
    samples = start + positions[found, None] * direction + offsets[best[found], None] * inward
    vx, vy, x, y = cv2.fitLine(samples.astype(np.float32), cv2.DIST_HUBER, 0, .01, .01).ravel()
    normal = np.float32([-vy, vx])
    residuals = np.abs((samples - [x, y]) @ normal)
    if np.median(residuals) > 2.8 or np.quantile(residuals, .90) > 8:
        return None
    return _replace_quad_side(quad, side, np.r_[normal, -normal @ [x, y]], 20, image.shape)


def _near_staggered_neighbor(quad: np.ndarray, neighbor: np.ndarray) -> bool:
    overlap, _ = cv2.intersectConvexConvex(quad, neighbor)
    if overlap > .025 * min(cv2.contourArea(quad), cv2.contourArea(neighbor)):
        return False
    for start, end in zip(neighbor, np.roll(neighbor, -1, axis=0), strict=True):
        along = end - start
        length = np.linalg.norm(along)
        tangent = along / length
        normal = np.float32([-tangent[1], tangent[0]])
        if (quad.mean(axis=0) - start) @ normal >= 0:
            continue
        for first, last in zip(quad, np.roll(quad, -1, axis=0), strict=True):
            projected = (np.float32([first, last]) - start) @ tangent
            lo, hi = np.sort(projected)
            shared_start, shared_end = max(0, lo), min(length, hi)
            shortest = min(length, hi - lo)
            if shortest <= 0 or not .30 * shortest <= shared_end - shared_start <= .87 * shortest:
                continue
            offsets = (np.float32([first, last]) - start) @ normal
            near = np.interp([shared_start, shared_end], np.sort(projected), offsets[projected.argsort()])
            if np.all((-15 <= near) & (near <= 7)):
                return True
    return False


def adjacent_album_edge_outlines(image: np.ndarray, neighbor: np.ndarray) -> list[np.ndarray]:
    scale = min(1., 1280 / max(image.shape[:2]))
    working = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    height, width = working.shape[:2]
    short = min(height, width)
    if short < 128 or not cv2.isContourConvex(neighbor):
        return []
    segments = _joined_segments(working, short)
    if len(segments) > 48:
        return []
    candidates = []
    for axis in (0, 1):
        arms = [rail for rail in segments if rail.axis == axis]
        crossbars = [rail for rail in segments if rail.axis != axis]
        for first, second in combinations(arms, 2):
            for middle in crossbars:
                if max(abs(middle.tangent @ arm.tangent) for arm in (first, second)) > .34:
                    continue
                lines = [np.cross(*np.column_stack((rail.ends, np.ones(2)))) for rail in (middle, first, second)]
                corners = np.cross(lines[0], lines[1:])
                joints = (corners[:, :2] / corners[:, 2:]).astype(np.float32)
                projection = np.sort((joints - middle.ends[0]) @ middle.tangent)
                if (projection[0] < -.10 * middle.length or projection[1] > 1.10 * middle.length
                        or np.ptp(projection) < .50 * middle.length
                        or np.min(np.linalg.norm(middle.ends[:, None] - joints, axis=2)) > .19 * middle.length):
                    continue
                for endpoint in (0, 1):
                    if any(np.linalg.norm(point - arm.ends[endpoint]) > .13 * arm.length
                           for point, arm in zip(joints, (first, second), strict=True)):
                        continue
                    last = np.float32([first.ends[1 - endpoint], second.ends[1 - endpoint]])
                    seed = order_quad_cyclic(np.vstack([joints, last[::-1]]))
                    lengths = np.linalg.norm(np.roll(seed, -1, axis=0) - seed, axis=1)
                    if (not cv2.isContourConvex(seed) or np.any(seed < 4) or np.any(seed > [width - 5, height - 5])
                            or lengths.min() < .17 * short or not .055 <= cv2.contourArea(seed) / (width * height) <= .36):
                        continue
                    if not .53 <= lengths[::2].sum() / lengths[1::2].sum() <= 2.:
                        continue
                    length = np.linalg.norm(last[1] - last[0])
                    tangent = (last[1] - last[0]) / length
                    normal = np.float32([-tangent[1], tangent[0]])
                    stations = np.linspace(0, length, 64)
                    observed = np.zeros(64, bool)
                    for rail in crossbars:
                        positions = rail.ends - last[0]
                        if rail is middle or np.max(np.abs(positions @ normal)) > 12:
                            continue
                        lo, hi = np.sort(positions @ tangent)
                        observed |= (stations >= lo) & (stations <= hi)
                    native = seed / scale
                    if observed.mean() >= .69 and _near_staggered_neighbor(native, neighbor):
                        candidates.append(native)
                        if len(candidates) > 8:
                            return []
    result = []
    for seed in candidates:
        quad, evidence, reasons = refine_print_boundary(image, seed, search_radius=18)
        if quad is None or reasons != ["PRINT_BOUNDARY_UNSUPPORTED"]:
            continue
        weak = [i for i, side in enumerate(evidence.values()) if side["support"] < .55 and not side["ambiguous"]]
        if len(weak) != 1 or any(side["support"] < .90 or (side["corner_support"] or 0) < .70
                                for index, side in enumerate(evidence.values()) if index != weak[0]):
            continue
        measured = _collar_outline(image, quad, weak[0])
        if measured is not None and _near_staggered_neighbor(measured, neighbor) and _photo_texture(image, measured):
            result.append(measured)
    return result
