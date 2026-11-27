"""Recover measured motif lines when a paper proposal has the wrong slant."""

from dataclasses import dataclass

import cv2
import numpy as np

from app.pipeline.geometry import normalized_polygon_overlap, order_quad_cyclic
from app.pipeline.print_edges import _edge_band, _line_endpoints, _quad_from_lines, _side_line


MEASURED_MOTIF_COMPARISON = "measured_motif_comparison"
MOTIF_COMPARISON_PAPER_EDIT = "motif_comparison_requires_paper_edit"
MOTIF_COMPARISON_REVISION = 1


@dataclass(frozen=True)
class MotifComparison:
    outline: np.ndarray
    motif_traces: list[np.ndarray]
    paper_traces: list[np.ndarray]


def _candidate_quads(lightness, origin, scale, polygon, area, sigma, settings):
    blurred = cv2.GaussianBlur(lightness, (0, 0), sigma)
    candidates = {}
    for low, high, closing in settings:
        edges = cv2.Canny(blurred, low, high)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((closing, closing), np.uint8))
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if not .45 * area <= cv2.contourArea(contour) <= 1.05 * area:
                continue
            hull = cv2.convexHull(contour)
            corners = cv2.approxPolyDP(hull, .012 * cv2.arcLength(hull, True), True).reshape(-1, 2)
            if (len(corners) != 4 or not cv2.isContourConvex(corners) or np.any(corners < 3)
                    or np.any(corners > np.array(lightness.shape[1::-1]) - 4)):
                continue
            corners = order_quad_cyclic(np.float32(corners + origin))
            overlap = normalized_polygon_overlap(corners / scale, polygon, raster_size=512)
            if .62 <= overlap.iou <= .86 and overlap.first_containment >= .965:
                candidates[tuple(corners.flat)] = corners
    return list(candidates.values())


def _fit_scene(image: np.ndarray, seed: np.ndarray, radius: int) -> np.ndarray | None:
    lines = []
    for start, end in zip(seed, np.roll(seed, -1, axis=0), strict=True):
        line, evidence = _side_line(image, start, end, radius)
        if (line is None or evidence["ambiguous"] or evidence["support"] < .88
                or (evidence.get("corner_support") or 0) < .68):
            return None
        lines.append(line)
    outline = _quad_from_lines(np.asarray(lines))
    if (outline is None or not cv2.isContourConvex(outline) or not np.isfinite(outline).all()
            or np.linalg.norm(outline - seed, axis=1).max() > 1.6 * radius):
        return None
    return outline


def _scene_traces(image: np.ndarray, outline: np.ndarray) -> list[np.ndarray]:
    fractions = np.linspace(.04, .96, 112, dtype=np.float32)
    traces = []
    for start, end in zip(outline, np.roll(outline, -1, axis=0), strict=True):
        band = _edge_band(image, start, end, [-7, -5, -3, -2, 2, 3, 5, 7], fractions=fractions)
        if band is None:
            return []
        outside, inside = (np.median(part, axis=1) for part in (band[:, :4], band[:, 4:]))
        stock = np.median(outside, axis=0)
        delta = outside - inside
        transition = (delta[:, 0] >= 12) | ((delta[:, 0] >= 2) & (np.linalg.norm(delta[:, 1:], axis=1) >= 13))
        pale = ((outside[:, 0] >= 133) & (np.linalg.norm(outside[:, 1:] - 128, axis=1) <= 16)
                & (np.linalg.norm(outside - stock, axis=1) <= 19))
        supported = np.flatnonzero(transition & pale)
        if len(supported) < .70 * len(fractions):
            return []
        # Missing texture on light clothing can interrupt a line. Mark the
        # measured runs and leave extended gaps visible in the source.
        breaks = np.flatnonzero(np.diff(supported) > 4) + 1
        runs = [run for run in np.split(supported, breaks) if len(run) >= 10]
        if not runs or sum(map(len, runs)) < .58 * len(fractions):
            return []
        traces.extend(start + fractions[run[[0, -1]], None] * (end - start) for run in runs)
    return traces


def _visible_paper_traces(image: np.ndarray, outline: np.ndarray, proposal: np.ndarray, shortest: float) -> list[np.ndarray]:
    traces = []
    for index, (start, end) in enumerate(zip(outline, np.roll(outline, -1, axis=0), strict=True)):
        delta = end - start
        inward = np.float32([-delta[1], delta[0]]) / np.linalg.norm(delta)
        prior = proposal[[index, (index + 1) % 4]]
        prior_side = prior[1] - prior[0]
        if (abs(prior_side @ inward) < .10 * np.linalg.norm(prior_side)
                or min((prior - [start, end]) @ inward) > -.035 * shortest):
            continue
        for radius in (round(.062 * shortest), round(.09 * shortest)):
            line, evidence = _side_line(image, start, end, radius, outward_only=True)
            if (line is None or evidence["ambiguous"] or evidence["support"] < .87
                    or (evidence.get("corner_support") or 0) < .67):
                continue
            first, last = np.float32(_line_endpoints(start, end, line))
            gaps = (np.float32([start, end]) - [first, last]) @ inward
            span = last - first
            if (gaps.min() < max(4, .022 * shortest) or np.ptp(gaps) > .025 * shortest
                    or np.linalg.norm(span) < .9 * np.linalg.norm(delta)):
                continue
            trace = np.float32([first + .12 * span, last - .12 * span])
            band = _edge_band(image, *trace, [-5, -3, -2, 2, 3, 5], fractions=np.linspace(.05, .95, 64, dtype=np.float32))
            if band is None:
                continue
            change = np.median(band[:, 3:], axis=1) - np.median(band[:, :3], axis=1)
            direction = np.median(change, axis=0)
            strength = np.linalg.norm(direction)
            if strength < 2.4 or np.mean(change @ direction >= 1.5 * strength) < .55:
                continue
            traces.append(trace)
            break
    return traces


def _pale_extension_has_marks(image: np.ndarray, outline: np.ndarray) -> bool:
    """Veto a preview if a bright exterior has small, locally darker marks."""
    edges = np.linalg.norm(np.roll(outline, -1, axis=0) - outline, axis=1)
    width, height = (round(edges[::2].mean()), round(edges[1::2].mean()))
    pad = int(np.clip(round(.16 * min(width, height)), 16, 65))
    target = np.float32([[pad, pad], [pad + width - 1, pad], [pad + width - 1, pad + height - 1],
                         [pad, pad + height - 1]])
    mapping = cv2.getPerspectiveTransform(outline, target)
    patch = cv2.warpPerspective(image, mapping, (width + 2 * pad, height + 2 * pad))
    lab = cv2.cvtColor(patch, cv2.COLOR_BGR2LAB)
    lightness = lab[..., 0]
    closed = cv2.morphologyEx(lightness, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    neutral = np.linalg.norm(lab[..., 1:].astype(np.float32) - 128, axis=2) <= 17
    pale_marks = np.uint8(neutral & (closed >= 215) & (closed - lightness >= 3))
    # The motif edge and interpolation outside the search collar cannot be marks.
    pale_marks[pad - 3:pad + height + 3, pad - 3:pad + width + 3] = 0
    pale_marks[:2] = pale_marks[-2:] = 0
    pale_marks[:, :2] = pale_marks[:, -2:] = 0
    _, _, stats, _ = cv2.connectedComponentsWithStats(pale_marks)
    dimensions = stats[1:, [cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT]]
    areas = stats[1:, cv2.CC_STAT_AREA]
    return bool(np.any((dimensions >= [2, 3]).all(axis=1) & (dimensions <= [42, 35]).all(axis=1)
                       & (areas >= 8) & (areas <= 550) & (areas < .88 * dimensions.prod(axis=1))))


def measured_motif_comparison(image: np.ndarray, polygon: np.ndarray, proposal: np.ndarray) -> MotifComparison | None:
    """Nominate a held preview from two native contour searches and signed printed edges."""
    scale = np.float32(image.shape[1::-1]) - 1
    shortest = float(np.linalg.norm(np.roll(proposal, -1, axis=0) - proposal, axis=1).min())
    if not 110 <= shortest <= 780:
        return None
    points = polygon * scale
    pad = .12 * shortest
    low = np.maximum(0, np.floor(points.min(axis=0) - pad).astype(int))
    high = np.minimum(np.int32(scale) + 1, np.ceil(points.max(axis=0) + pad).astype(int))
    if (high - low).prod() > 1_200_000:
        return None
    lightness = cv2.cvtColor(image[low[1]:high[1], low[0]:high[0]], cv2.COLOR_BGR2LAB)[..., 0]
    area = cv2.contourArea(points)
    radius = int(np.clip(round(.018 * shortest), 4, 8))
    groups = []
    for sigma, settings in (
        (.65, ((15, 34, 3), (18, 37, 5), (22, 42, 7))),
        (.85, ((12, 28, 5), (15, 34, 5), (18, 37, 7))),
        (1.30, ((9, 25, 7), (12, 28, 7))),
    ):
        candidates = _candidate_quads(lightness, low, scale, polygon, area, sigma, settings)
        fits = [fit for seed in candidates if (fit := _fit_scene(image, seed, radius)) is not None
                and _scene_traces(image, fit)]
        if fits:
            groups.append(fits)
        if len(groups) == 2:
            break
    if len(groups) != 2:
        return None
    fits = np.asarray(groups[0] + groups[1])
    outline = np.median(fits, axis=0).astype(np.float32)
    if np.linalg.norm(fits - outline, axis=-1).max() > max(3., .011 * shortest):
        return None
    traces = _scene_traces(image, outline)
    if not traces:
        return None

    # Use the enclosing fitted line on each side. A small outside allowance
    # preserves native boundary centres across the independent refits.
    directions = np.roll(outline, -1, axis=0) - outline
    normals = np.column_stack((-directions[:, 1], directions[:, 0]))
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    gaps = np.einsum("kij,ij->ki", fits - outline, normals)
    next_gaps = np.einsum("kij,ij->ki", np.roll(fits - outline, -1, axis=1), normals)
    edge_offsets = np.minimum(gaps.min(axis=0), next_gaps.min(axis=0)) - 2.
    lines = np.column_stack((normals, -(normals * outline).sum(axis=1) - edge_offsets))
    envelope = _quad_from_lines(lines)
    if envelope is None or np.any(envelope < 0) or np.any(envelope > scale):
        return None
    if _pale_extension_has_marks(image, envelope):
        return None
    return MotifComparison(envelope, traces, _visible_paper_traces(image, outline, proposal, shortest))
