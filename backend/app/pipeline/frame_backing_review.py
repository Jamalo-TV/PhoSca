"""Source evidence that a proposal spans the album edge at the capture frame."""

import cv2
import numpy as np

from app.pipeline.crossing_prints import _quarter_turn
from app.pipeline.geometry import order_quad_cyclic
from app.pipeline.print_edges import _sample_lab, _side_line


FRAME_BACKING_PROPOSAL = "possible_frame_backing"
FRAME_BACKING_COUNT_REVIEW = "FRAME_BACKING_COUNT_REVIEW"


def _surface_transition(
    image: np.ndarray, quad: np.ndarray, owner: np.ndarray,
) -> tuple[np.ndarray, np.ndarray] | None:
    depth = float(np.mean(quad[2:, 1] - quad[:2, 1]))
    left = max(0, round(quad[0, 0] - 130))
    right = min(image.shape[1] - 1, round(quad[1, 0] + 130))
    first = max(14, round(.045 * depth))
    last = min(image.shape[0] - 12, round(.22 * depth))
    if right - left < 250 or not 24 <= last - first <= 840:
        return None

    columns = np.linspace(left, right, 192, dtype=np.float32)
    grid = np.stack(np.meshgrid(columns, np.arange(first - 10, last + 10, dtype=np.float32)), axis=-1)
    lightness = _sample_lab(image, grid.reshape(-1, 2))[:, 0].reshape(grid.shape[:2])
    size = last - first
    step = lightness[11:11 + size] - lightness[9:9 + size]
    above, below = (np.median([lightness[10 + offset:10 + offset + size] for offset in offsets], axis=0)
                    for offsets in ((-4, -7, -10), (4, 7, 10)))
    votes = np.uint8((step >= 8) & (below >= 190) & (below - above >= 16))
    support = cv2.dilate(votes, np.ones((5, 1), np.uint8))
    rows, indices = np.nonzero(support)
    if len(rows) < 100:
        return None

    # Nominate lines with source votes at every tested shallow slope. Validate
    # the nominated line independently at both crossings and outside them.
    center = float(columns[len(columns) // 2])
    centered_columns = columns - center
    rail_tips = quad[[0, 1]]
    rails = quad[[3, 2]] - rail_tips
    best_score = 0.
    best = None
    for slope in np.linspace(-.05, .05, 101):
        centers = np.rint(rows - slope * centered_columns[indices]).astype(int)
        valid = centers[(centers >= 0) & (centers < size)]
        histogram = np.bincount(valid, minlength=size)
        for peak in np.argpartition(histogram, -3)[-3:]:
            if histogram[peak] < .72 * len(columns):
                continue
            locations = first + peak + slope * centered_columns
            offsets = np.rint(locations - first).astype(int)
            if offsets.min() < 0 or offsets.max() >= size:
                continue
            depth_at_tips = first + peak + slope * (rail_tips[:, 0] - center) - rail_tips[:, 1]
            intersections = rail_tips + (depth_at_tips / (rails[:, 1] - slope * rails[:, 0]))[:, None] * rails
            a, b = intersections[:, 0]
            interior = (columns > a + 22) & (columns < b - 22)
            zones = (columns < a - 7, np.abs(columns - a) <= 34, interior,
                     np.abs(columns - b) <= 34, columns > b + 7)
            if min(zone.sum() for zone in zones) < 8:
                continue
            visible = support[offsets, np.arange(len(columns))] > 0
            weakest = min(visible[zone].mean() for zone in zones)
            if weakest < .75:
                continue
            gaps = np.flatnonzero(np.diff(np.r_[False, ~visible[interior], False])).reshape(-1, 2)
            if len(gaps) and np.diff(gaps, axis=1).max() * (columns[1] - columns[0]) > 32:
                continue
            score = weakest + .18 * visible.mean()
            if score <= best_score:
                continue
            stations = np.column_stack((columns, locations)).astype(np.float32)
            if any(cv2.pointPolygonTest(owner, tuple(point), True) > -12 for point in stations[::4]):
                continue
            best_score = score
            best = stations[[0, -1]], intersections
    return best


def _continues_past_lateral_sides(
    image: np.ndarray, quad: np.ndarray, owner: np.ndarray, intersections: np.ndarray,
) -> bool:
    mean_depth = float(intersections[:, 1].mean())
    for start, stop, corner in zip(quad[:2], quad[[3, 2]], intersections, strict=True):
        edge = stop - start
        depth = np.linspace(max(8, .16 * mean_depth), mean_depth + min(110, .12 * edge[1]), 96, dtype=np.float32)
        stations = start + (depth - start[1])[:, None] * edge / edge[1]
        exposed = np.array([cv2.pointPolygonTest(owner, tuple(point), True) < -20 for point in stations])
        clear = exposed & (np.abs(depth - corner[1]) > 12)
        if clear.mean() < .50:
            return False
        normal = np.float32([-edge[1], edge[0]]) / np.linalg.norm(edge)
        samples = stations[:, None] + np.float32([-11, -8, -4, 4, 8, 11])[None, :, None] * normal
        if np.any((samples < 0) | (samples > np.array(image.shape[1::-1]) - 1)):
            return False
        lab = _sample_lab(image, samples.reshape(-1, 2)).reshape(len(stations), 6, 3)
        first, last = (np.median(part, axis=1) for part in np.split(lab, 2, axis=1))
        matches = np.linalg.norm(first - last, axis=1) <= 8
        if matches[clear].mean() < .78:
            return False
        if any(selection.sum() >= 7 and seen[selection].mean() <= .69
               for selection, seen in zip(np.array_split(clear, 4), np.array_split(matches, 4), strict=True)):
            return False
    return True


def _borrows_owner_side(image: np.ndarray, quad: np.ndarray, owner: np.ndarray) -> bool:
    line, evidence = _side_line(image, owner[2], owner[3], 12, collect_curve_segments=True)
    if evidence["support"] < .87:
        return False
    if line is None:
        alternatives = evidence.get("competing_print_edges") or []
        if len(alternatives) != 2:
            return False
        lines = [np.cross([*start, 1.], [*end, 1.]) for start, end in alternatives]
    else:
        lines = [line]
    stations = quad[2] + np.linspace(.1, .9, 112, dtype=np.float32)[:, None] * (quad[3] - quad[2])
    return all(np.mean(np.abs(stations @ rail[:2] + rail[2]) <= 16 * np.linalg.norm(rail[:2])) >= .70
               for rail in lines)


def _pale_album_material(image: np.ndarray, quad: np.ndarray, owner: np.ndarray, surface: np.ndarray) -> bool:
    stride = 4
    low = np.maximum(0, np.floor(quad.min(axis=0)).astype(int))
    high = np.minimum(image.shape[1::-1], np.ceil(quad.max(axis=0)).astype(int) + 1)
    left, top = low
    right, bottom = high
    region = image[top:bottom:stride, left:right:stride]
    inside = np.zeros(region.shape[:2], np.uint8)
    claimed = np.zeros_like(inside)
    cv2.fillConvexPoly(inside, np.rint((quad - low) / stride).astype(np.int32), 1)
    cv2.fillConvexPoly(claimed, np.rint((owner - low) / stride).astype(np.int32), 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    slope = (surface[1, 1] - surface[0, 1]) / (surface[1, 0] - surface[0, 0])
    boundary = surface[0, 1] + slope * (np.arange(left, right, stride) - surface[0, 0])
    depth = np.arange(top, bottom, stride)[:, None] - boundary
    clear = (cv2.erode(inside, kernel) != 0) & (cv2.dilate(claimed, kernel) == 0) & (depth >= 23)
    lab = cv2.cvtColor(region, cv2.COLOR_BGR2LAB).astype(np.float32)
    pale = clear & (depth <= 54) & (lab[:, :, 0] >= 190)
    if clear.sum() < 320 or pale.sum() < 80:
        return False
    tone = np.median(lab[pale], axis=0)
    difference = np.linalg.norm(lab - tone, axis=2)
    distinct = np.uint8(clear & (difference > 5.5))
    _, _, stats, _ = cv2.connectedComponentsWithStats(distinct, connectivity=8)
    return not np.any(stats[1:, cv2.CC_STAT_AREA] * stride ** 2 >= 1000)


def frame_backing_trace(image: np.ndarray, proposal: np.ndarray, owner: np.ndarray) -> np.ndarray | None:
    """Locate the exposed album transition for an individually reviewable proposal."""
    height, width = image.shape[:2]
    orientations = [
        (turn, quad) for turn in range(4)
        if np.max((quad := order_quad_cyclic(_quarter_turn(proposal, turn, width, height)))[:2, 1]) <= 2
    ]
    if len(orientations) != 1:
        return None
    turn, quad = orientations[0]
    rotated_owner = order_quad_cyclic(_quarter_turn(owner, turn, width, height))
    if np.linalg.norm(np.roll(quad, -1, axis=0) - quad, axis=1).min() < 300:
        return None
    image = np.ascontiguousarray(np.rot90(image, turn))
    if not _borrows_owner_side(image, quad, rotated_owner):
        return None
    evidence = _surface_transition(image, quad, rotated_owner)
    if evidence is None:
        return None
    surface, intersections = evidence
    if not _continues_past_lateral_sides(image, quad, rotated_owner, intersections):
        return None
    if not _pale_album_material(image, quad, rotated_owner, surface):
        return None
    return _quarter_turn(surface, (-turn) % 4, image.shape[1], image.shape[0])
