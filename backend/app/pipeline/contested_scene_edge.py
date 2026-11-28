"""Measure open warm-to-cool scene transitions without inferring paper ownership."""

import cv2
import numpy as np

from app.pipeline.geometry import normalized_polygon_overlap, normalized_vertices_to_pixels, order_quad_cyclic, quad_from_polygon
from app.pipeline.near_capture_review import _claimed_contours
from app.pipeline.print_edges import _edge_band


CONTESTED_SCENE_EDGE_TRACES = "contested_scene_edge_traces"
_SOURCE_SHOULDER = 8


def _warm_rail(source: np.ndarray, side: np.ndarray) -> np.ndarray | None:
    start, end = side
    length = np.linalg.norm(end - start)
    if length < 100:
        return None
    fractions = np.linspace(.09, .91, 180, dtype=np.float32)
    band = _edge_band(source, start, end, range(-28, 36), fractions=fractions)
    if band is None:
        return None
    columns = np.arange(8, 56)
    outside = np.median(band[:, columns[:, None] + [-7, -5, -3]], axis=2)
    inside = np.median(band[:, columns[:, None] + [2, 4, 6]], axis=2)
    wide = (outside - inside)[..., 2]
    step = band[:, columns - 1] - band[:, columns + 1]
    warm = ((outside[..., 2] >= 135) & (outside[..., 0] >= 145) & (wide >= 8) & (step[..., 2] >= 2)
            & (np.linalg.norm(step, axis=2) >= 4))
    shifts, slopes = np.meshgrid(np.arange(-20, 27.1, .5), np.arange(-.095, .096, .0025))
    end_offsets = shifts.ravel()[:, None] + slopes.ravel()[:, None] * length * np.float32([-.5, .5])
    offsets = end_offsets[:, :1] + np.diff(end_offsets, axis=1) * fractions
    inside_band = (offsets.min(axis=1) >= -20) & (offsets.max(axis=1) <= 27)
    samples = np.rint(offsets[inside_band]).astype(np.intp) + 20
    at = (np.arange(len(fractions)), samples)
    signed = warm[at]
    support = signed.mean(axis=1)
    weakest = np.min([chunk.mean(axis=1) for chunk in np.array_split(signed, 8, axis=1)], axis=0)
    scores = (.48 * support + .28 * weakest + .14 * np.clip(wide[at] / 25, 0, 1).mean(axis=1)
              + .10 * np.clip(step[..., 2][at] / 10, 0, 1).mean(axis=1))
    best = np.argmax(scores)
    if support[best] < .50:
        return None
    normal = np.flip(end - start) * [-1, 1] / length
    return np.float32(side + end_offsets[inside_band][best, :, None] * normal)


def _observed_spans(source: np.ndarray, rail: np.ndarray) -> list[np.ndarray]:
    start, end = rail
    edge = end - start
    length = np.linalg.norm(edge)
    fractions = np.linspace(.12, .88, round(.76 * length), dtype=np.float32)
    offsets = np.arange(-7, 7.01, .25, dtype=np.float32)
    band = _edge_band(source, start, end, offsets, fractions=fractions)
    if band is None:
        return []
    columns = np.flatnonzero(np.abs(offsets) <= 3)
    step = band[:, columns - 4, 2] - band[:, columns + 4, 2]
    wide = band[:, columns - 12, 2] - band[:, columns + 12, 2]
    plausible = (band[:, columns - 8, 2] >= 135) & (step >= 3) & (wide >= 7)
    supported = plausible.any(axis=1)
    if supported.mean() < .80:
        return []
    peaks = offsets[columns[np.argmax(np.where(plausible, step + .28 * wide, -1), axis=1)]]
    stations = fractions * length
    vx, vy, x, y = cv2.fitLine(np.column_stack((stations[supported], peaks[supported])), cv2.DIST_HUBER, 0, .01, .01).ravel()
    if abs(vx) < .1:
        return []
    fitted = y + vy / vx * (stations - x)
    agreed = supported & (np.abs(fitted - peaks) <= 1.25)
    if agreed.mean() < .80 or min(chunk.mean() for chunk in np.array_split(agreed, 4)) < .65:
        return []
    positions = start + fractions[:, None] * edge + fitted[:, None] * np.float32([-edge[1], edge[0]]) / length
    hits = np.flatnonzero(agreed)
    return [positions[run[[0, -1]]] for run in np.split(hits, np.flatnonzero(np.diff(hits) > 3) + 1) if len(run) >= 15]


def _common_spans(first: list[np.ndarray], second: list[np.ndarray], claims: list[np.ndarray]) -> list[np.ndarray]:
    result = []
    for a, b in first:
        delta = b - a
        length = np.linalg.norm(delta)
        direction = delta / length
        normal = np.float32([-direction[1], direction[0]])
        for other in second:
            offsets = other - a
            if np.max(np.abs(offsets @ normal)) > 1.25:
                continue
            left, right = sorted(offsets @ direction)
            left, right = max(0., left), min(length, right)
            if right - left < 15:
                continue
            start, end = a + left * direction, a + right * direction
            if not _span_intersects_claim(start, end, claims):
                result.append(np.float32([start, end]))
    return result


def _span_intersects_claim(start: np.ndarray, end: np.ndarray, claims: list[np.ndarray]) -> bool:
    delta = end - start
    length = np.linalg.norm(delta)
    if length == 0:
        return False
    along = _SOURCE_SHOULDER * delta / length
    across = np.float32([-along[1], along[0]])
    collar = np.float32([start - along - across, end + along - across,
                         end + along + across, start - along + across])
    return any(cv2.intersectConvexConvex(collar, claim)[0] > .01 for claim in claims)


def contested_scene_guide_conflicts(outline: list[dict], traces: list, width: int, height: int) -> bool:
    """Retire the guide as soon as a saved claim enters its native sampling band."""
    claims = _claimed_contours([{"polygon": outline}], width, height, 0)
    return any(_span_intersects_claim(*normalized_vertices_to_pixels([(p["x"], p["y"]) for p in trace], width, height), claims)
               for trace in traces if len(trace) == 2)


def contested_scene_edge_traces(source: np.ndarray, polygons: list[np.ndarray], claims: list[dict]) -> list[np.ndarray]:
    """Return a single independently observed side as open pieces in source pixels."""
    if len(polygons) != 2 or normalized_polygon_overlap(*polygons, raster_size=512).iou < .90:
        return []
    height, width = source.shape[:2]
    quads = [order_quad_cyclic(quad_from_polygon(normalized_vertices_to_pixels(polygon, width, height))[0])
             for polygon in polygons]
    contours = _claimed_contours(claims, width, height, 0)
    sides = []
    for index in range(4):
        vertices = [index, (index + 1) % 4]
        rails = [_warm_rail(source, quad[vertices]) for quad in quads]
        if any(rail is None for rail in rails):
            continue
        first, second = (_observed_spans(source, rail) for rail in rails)
        shared = _common_spans(first, second, contours)
        if sum(np.linalg.norm(end - start) for start, end in shared) >= 90:
            sides.append(shared)
    return sides[0] if len(sides) == 1 else []
