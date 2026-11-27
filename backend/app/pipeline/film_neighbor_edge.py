"""Identify the far edge of a source corridor beside a reviewed film sleeve."""

import cv2
import numpy as np

from app.pipeline.film_carrier_review import refitted_film_carrier
from app.pipeline.geometry import normalized_vertices_to_pixels, order_quad_cyclic, pixel_vertices_to_normalized
from app.pipeline.print_edges import _sample_lab


FILM_NEIGHBOR_EDGE = "film_neighbor_edge"


def _matching_side(trace: np.ndarray, outline: np.ndarray, tolerance: float) -> np.ndarray | None:
    matches = []
    for start, end in zip(outline, np.roll(outline, -1, axis=0), strict=True):
        edge = end - start
        length = np.linalg.norm(edge)
        tangent = edge / length
        normal = np.float32([-tangent[1], tangent[0]])
        depth = (trace - start) @ normal
        span = (trace - start) @ tangent / length
        if np.max(np.abs(depth)) <= tolerance and span.min() >= 0 and span.max() <= 1 and np.ptp(span) >= .80:
            matches.append(np.float32([start, end]))
    return matches[0] if len(matches) == 1 else None


def _corridor_scope(image: np.ndarray, paper_side: np.ndarray, film_side: np.ndarray) -> np.ndarray | None:
    start, end = paper_side
    film_start, film_end = film_side
    paper_edge = end - start
    film_edge = film_end - film_start
    outward = np.float32([paper_edge[1], -paper_edge[0]]) / np.linalg.norm(paper_edge)
    film_outward = np.float32([film_edge[1], -film_edge[0]]) / np.linalg.norm(film_edge)
    alignment = float(outward @ film_outward)
    if alignment > -.98:
        return None
    fractions = np.linspace(.075, .925, 256, dtype=np.float32)
    stations = start + fractions[:, None] * paper_edge
    gap = ((film_start - stations) @ film_outward) / alignment
    across = stations + gap[:, None] * outward
    film_span = (across - film_start) @ film_edge / (film_edge @ film_edge)
    if gap.min() < 7 or gap.max() > 18 or film_span.min() < 0 or film_span.max() > 1:
        return None
    scale = np.float32(image.shape[1::-1]) - 1
    bounds = np.concatenate([stations - 5.5 * outward, across + 5.5 * outward])
    if np.any((bounds < 0) | (bounds > scale)):
        return None

    def sample(depths):
        points = stations[:, None] + depths[..., None] * outward
        return _sample_lab(image, points.reshape(-1, 2)).reshape(*points.shape[:2], 3)[..., 1:]

    depths = np.float32([2.5, 3.5, 4.5, 5.5])
    paper_stock = np.median(sample(-depths), axis=1)
    film_stock = np.median(sample(gap[:, None] + depths), axis=1)
    backing_samples = sample(gap[:, None] * np.float32([.29, .42, .58, .71]))
    backing = np.median(backing_samples, axis=1)
    difference = np.stack([paper_stock - backing, film_stock - backing])
    magnitude = np.linalg.norm(difference, axis=2)
    normal = difference / np.maximum(1, magnitude[..., None])
    continuity = (magnitude.min(axis=0) >= 12) & ((normal[0] * normal[1]).sum(axis=1) >= .80)
    continuity &= np.linalg.norm(backing_samples - backing[:, None], axis=2).max(axis=1) <= 8
    continuity &= np.linalg.norm(paper_stock - film_stock, axis=1) <= 12

    search = np.arange(-2.5, 2.501, .25, dtype=np.float32)
    for index, positions in enumerate((search, gap[:, None] + search)):
        values = ((sample(positions) - backing[:, None]) * normal[index, :, None]).sum(axis=2)
        fraction = values / np.maximum(magnitude[index, :, None], 1)
        best = np.abs(fraction - .5).argmin(axis=1)
        station_ids = np.arange(len(stations))
        sharpness = np.abs(values[station_ids, np.minimum(len(search) - 1, best + 4)]
                           - values[station_ids, np.maximum(0, best - 4)])
        continuity &= (np.abs(search[best]) <= 1.5) & (np.abs(fraction[station_ids, best] - .5) <= .25) & (sharpness >= 5)

    if (continuity.mean() < .90 or min(part.mean() for part in np.array_split(continuity, 8)) < .75
            or np.any(~(continuity[:-3] | continuity[1:-2] | continuity[2:-1] | continuity[3:]))):
        return None
    return np.float32([stations[0], stations[-1], across[-1], across[0]])


def film_neighbor_edge(
    image: np.ndarray, paper_quad: list[dict], traces: list[list[dict]], group: list[dict], neighbors: list[dict],
) -> dict | None:
    """Bind an existing alternative trace to the far side of a separate source corridor."""
    height, width = image.shape[:2]

    def pixels(points):
        return normalized_vertices_to_pixels([(point["x"], point["y"]) for point in points], width, height)

    parents = [mask for mask in group if mask.get("source") == "classical_border_quad"
               and mask.get("detector_source") == "contrast_region"]
    children = [mask for mask in group if mask.get("source") == "yolo_seg_mask" and mask.get("detector_source") == "yolo_onnx"]
    if len(parents) != 1 or len(children) != 2 or len(paper_quad) != 4 or len(traces) != 2:
        return None
    parent, = parents
    marks = ("film_grid_review_seam", "film_grid_review_rim")
    if any(len(parent.get(key) or []) != 2 or any(child.get(key) != parent[key] for child in children) for key in marks):
        return None

    # The original broad proposal can include the gap. Check ownership on the measured contour.
    fitted = refitted_film_carrier(image, order_quad_cyclic(pixels(parent["polygon"])),
                                 [pixels(child["polygon"]) for child in children], [])
    if fitted is None:
        return None
    carrier, seam, rim = fitted
    if any(np.linalg.norm(line - pixels(parent[key]), axis=1).max() > .25 for line, key in zip((seam, rim), marks, strict=True)):
        return None
    paper = order_quad_cyclic(pixels(paper_quad))
    if cv2.intersectConvexConvex(paper, carrier)[0] > 1:
        return None
    pair = [pixels(trace) for trace in traces]
    if any(len(trace) != 2 for trace in pair):
        return None
    paper_sides = [_matching_side(trace, paper, 1.) for trace in pair]
    film_sides = [_matching_side(trace, carrier, 1.5) for trace in pair]
    matches = [index for index in range(2) if film_sides[index] is not None and paper_sides[1 - index] is not None]
    if len(matches) != 1:
        return None
    index, = matches
    scope = _corridor_scope(image, paper_sides[1 - index], film_sides[index])
    if scope is None:
        return None
    for neighbor in neighbors:
        for key in ("polygon", "visible_polygon", "warp_quad"):
            if len(points := neighbor.get(key) or []) >= 3:
                claim = cv2.convexHull(pixels(points))
                if any(cv2.intersectConvexConvex(outline, claim)[0] > 1 for outline in (scope, carrier)):
                    return None
    return {
        "trace_index": index,
        "source_scope": [dict(x=float(x), y=float(y)) for x, y in pixel_vertices_to_normalized(scope, width, height)],
        "carrier_outline": [dict(x=float(x), y=float(y)) for x, y in pixel_vertices_to_normalized(carrier, width, height)],
    }
