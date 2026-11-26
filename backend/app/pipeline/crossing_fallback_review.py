"""Locate original-image evidence for a disputed classical count proposal."""

import cv2
import numpy as np

from app.pipeline.edge_prints import _has_independent_content
from app.pipeline.fusion_ownership import _borrowed_side
from app.pipeline.geometry import order_quad_cyclic, quad_from_polygon
from app.pipeline.print_edges import _edge_band, _line_endpoints, _side_line, refine_print_boundary


CROSSING_FALLBACK_TRACES = "crossing_fallback_review_traces"
CROSSING_FALLBACK_COUNT_REVIEW = "CROSSING_FALLBACK_COUNT_REVIEW"


def _scene_crossing(
    gray: np.ndarray, start: np.ndarray, end: np.ndarray, peers: list[np.ndarray],
) -> tuple[int, np.ndarray] | None:
    edge = end - start
    length = np.linalg.norm(edge)
    tangent = edge / length
    normal = np.float32([-tangent[1], tangent[0]])
    stations = start + np.arange(0, length, dtype=np.float32)[:, None] * tangent
    distances = np.array([[cv2.pointPolygonTest(peer, tuple(point), True) for point in stations] for peer in peers])
    results = []
    for index in range(len(peers)):
        in_scene = (distances[index] >= 26) & (np.delete(distances, index, axis=0).max(axis=0) < -26)
        intervals = np.flatnonzero(np.diff(np.r_[False, in_scene, False])).reshape(-1, 2)
        for first, last in intervals:
            if last - first < max(200, .15 * length):
                continue
            points = stations[None, first + 6:last - 6] + np.arange(-24, 25, dtype=np.float32)[:, None, None] * normal
            if np.any((points < 0) | (points > np.float32(gray.shape[1::-1]) - 1)):
                continue
            strip = cv2.remap(gray, points[..., 0].copy(), points[..., 1].copy(), cv2.INTER_LINEAR)
            edges = cv2.Canny(strip, 35, 80)
            _, labels = cv2.connectedComponents(edges, connectivity=8)
            crossing = np.intersect1d(labels[:17], labels[33:])
            crossing = crossing[crossing != 0]
            if len(crossing) < 8:
                continue
            columns = np.nonzero(np.isin(labels[20:29], crossing))[1]
            if (len(np.unique(columns // 12)) >= 9 and np.ptp(columns) >= .40 * strip.shape[1]):
                results.append((index, stations[[first, last - 1]]))
    return results[0] if len(results) == 1 else None


def _shared_neighbor_edge(
    image: np.ndarray, span: np.ndarray, peer: np.ndarray,
) -> bool:
    fractions = np.linspace(.08, .92, 80, dtype=np.float32)
    stations = span[0] + fractions[:, None] * (span[1] - span[0])
    close = np.array([abs(cv2.pointPolygonTest(peer, tuple(point), True)) <= 24 for point in stations])
    if close.mean() < .90:
        return False
    seed, method = quad_from_polygon(peer)
    if method == "minimum_area_rect":
        return False
    shortest = np.linalg.norm(np.roll(seed, -1, axis=0) - seed, axis=1).min()
    quad, edges, _ = refine_print_boundary(image, seed, search_radius=int(np.clip(round(.042 * shortest), 7, 42)))
    if quad is None or (side := _borrowed_side(stations, quad)) is None:
        return False
    evidence = list(edges.values())[side]
    return evidence["support"] >= .90 and not evidence["ambiguous"]


def _pale_step_polarity(image: np.ndarray, start: np.ndarray, end: np.ndarray, *, coverage: float) -> int:
    band = _edge_band(image, start, end, [-14, -10, -6, -3, 3, 6, 10, 14])
    if band is None:
        return 0
    left, right = (np.median(section, axis=1) for section in np.split(band, 2, axis=1))
    delta = left[:, 0] - right[:, 0]
    for polarity, pale in ((1, left), (-1, right)):
        support = ((polarity * delta >= 26) & (pale[:, 0] >= 175)
                   & (np.linalg.norm(pale[:, 1:] - 128, axis=1) < 16))
        if support.mean() >= coverage:
            return polarity
    return 0


def _continuing_album_edge(
    image: np.ndarray, span: np.ndarray, peers: list[np.ndarray],
) -> np.ndarray | None:
    length = np.linalg.norm(span[1] - span[0])
    for first, last in (span, span[::-1]):
        tangent = (last - first) / length
        start, end = last + 12 * tangent, last + min(.14 * length, 192.) * tangent
        if np.linalg.norm(end - start) < 64:
            continue
        stations = start + np.linspace(.05, .95, 32, dtype=np.float32)[:, None] * (end - start)
        if (np.any((stations < 0) | (stations > np.float32(image.shape[1::-1]) - 1))
                or any(cv2.pointPolygonTest(peer, tuple(point), True) > -16 for peer in peers for point in stations)):
            continue
        line, evidence = _side_line(image, start, end, 11)
        if line is None or evidence["ambiguous"] or evidence["support"] < .92:
            continue
        border = np.float32(_line_endpoints(start, end, line))
        if np.linalg.norm(border - [start, end], axis=1).max() > 8:
            continue
        beyond = _pale_step_polarity(image, *border, coverage=.88)
        if beyond and beyond == _pale_step_polarity(image, last - 45 * tangent, last - 12 * tangent, coverage=.66):
            return border
    return None


def crossing_fallback_review_traces(
    image: np.ndarray, proposal: np.ndarray, peers: list[np.ndarray],
    *, extension_claims: list[np.ndarray] | None = None,
) -> list[np.ndarray]:
    """Trace four observed spans while leaving the candidate's source claim active."""
    quad = order_quad_cyclic(proposal)
    shortest = np.linalg.norm(np.roll(quad, -1, axis=0) - quad, axis=1).min()
    if shortest < 320:
        return []
    radius = int(np.clip(round(.032 * shortest), 7, 36))
    strong, weak = [], []
    for start, end in zip(quad, np.roll(quad, -1, axis=0), strict=True):
        line, evidence = _side_line(image, start, end, radius)
        if evidence["ambiguous"]:
            return []
        if line is None and evidence["support"] <= .36:
            weak.append((start, end))
        elif line is not None and evidence["support"] >= .80:
            strong.append(np.float32(_line_endpoints(start, end, line)))
    if len(strong) != 2 or len(weak) != 2:
        return []

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    scenes = [_scene_crossing(gray, start, end, peers) for start, end in weak]
    if any(scene is None for scene in scenes):
        return []
    (first, first_trace), (second, second_trace) = scenes
    if first == second:
        return []
    matches = [index for index, span in enumerate(strong) if any(
        _shared_neighbor_edge(image, span, peer) for owner, peer in enumerate(peers) if owner not in (first, second)
    )]
    if len(matches) != 1:
        return []
    shared, = matches
    continuation = _continuing_album_edge(image, strong[1 - shared], [*peers, *(extension_claims or [])])
    if continuation is None:
        return []
    covered = np.zeros(image.shape[:2], np.uint8)
    for peer in peers:
        cv2.fillPoly(covered, [np.rint(peer).astype(np.int32)], 1)
    covered = cv2.dilate(covered, np.ones((7, 7), np.uint8))
    return [] if _has_independent_content(image, covered, quad) else [
        strong[shared], continuation, first_trace, second_trace,
    ]
