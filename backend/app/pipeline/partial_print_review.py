"""Measure open guides below a partial print claim without extending that claim."""

import cv2
import numpy as np

from app.pipeline.fragmented_prints import _image_continues_across
from app.pipeline.print_edges import COMPETING_PRINT_EDGES, _edge_band, _line_endpoints, _side_line


PARTIAL_PRINT_EXTENT_REVIEW = "PARTIAL_PRINT_EXTENT_REVIEW"
PARTIAL_PRINT_EXTENT_TRACES = "partial_print_extent_traces"


def _dark_cool_transition(image: np.ndarray, span: np.ndarray) -> bool:
    band = _edge_band(image, *span, [-3, -2, -1, 1, 2, 3], fractions=np.linspace(.10, .90, 96, dtype=np.float32))
    if band is None:
        return False
    delta = np.median(band[:, 3:], axis=1) - np.median(band[:, :3], axis=1)
    support = (delta[:, 0] <= -20) & (delta[:, 2] <= -3)
    return support.mean() >= .80 and min(part.mean() for part in np.array_split(support, 4)) >= .65


def _continuing_flank(
    image: np.ndarray, a: np.ndarray, b: np.ndarray, distances: np.ndarray, *, extends_after: bool,
) -> np.ndarray | None:
    vector = b - a
    length = np.linalg.norm(vector)
    tangent = vector / length
    # Leave room on each side of the disputed cut and the potential corners.
    pad = .06 * length
    observed = [distances[1] + pad, length - pad] if extends_after else [pad, distances[0] - pad]
    if observed[1] - observed[0] < max(80, .28 * length):
        return None
    span = np.float32(a + np.float32(observed)[:, None] * tangent)
    line, evidence = _side_line(image, *span, 7)
    if line is None or evidence["ambiguous"] or evidence["support"] < .90 or evidence["corner_support"] < .80:
        return None
    inner = np.float32(a + np.float32([distances[0] + pad, distances[1] - pad])[:, None] * tangent)
    inner_line, inner_evidence = _side_line(image, *inner, 7)
    if inner_line is None or inner_evidence["ambiguous"] or inner_evidence["support"] < .85:
        return None
    traced = np.float32(_line_endpoints(*span, line))
    if np.max(np.linalg.norm(np.float32(_line_endpoints(*span, inner_line)) - traced, axis=1)) > 2:
        return None
    return traced if _dark_cool_transition(image, traced) else None


def partial_print_extent_traces(image: np.ndarray, claim: np.ndarray, tile: np.ndarray) -> list[np.ndarray]:
    """Return two measured continuations and two competing terminal transitions."""
    delta = np.roll(tile, -1, axis=0) - tile
    lengths = np.linalg.norm(delta, axis=1)
    if lengths.min() < 200:
        return []
    tangents = delta / lengths[:, None]
    normals = np.column_stack((-tangents[:, 1], tangents[:, 0]))
    offsets = np.stack((claim - tile, np.roll(claim, -1, axis=0) - tile), axis=1)
    distances = np.einsum("ijk,ik->ij", offsets, tangents)
    aligned = (np.abs(np.einsum("ijk,ik->ij", offsets, normals)).max(axis=1) <= 4)
    aligned &= (distances.min(axis=1) >= -4) & (distances.max(axis=1) <= lengths + 4)
    unmatched = np.flatnonzero(~aligned)
    if len(unmatched) != 1:
        return []
    side = int(unmatched[0])
    cut = claim[side]
    end = claim[(side + 1) % 4]
    line, evidence = _side_line(image, cut, end, 8)
    if line is not None or evidence["ambiguous"] or evidence["support"] > .50:
        return []
    # Image features must continue across the cut at several sample scales.
    # An aligned overlay may still reproduce them, so this remains a review cue.
    cut_vector = end - cut
    normal = np.float32([-cut_vector[1], cut_vector[0]]) / np.linalg.norm(cut_vector)
    first, last = cut + .10 * cut_vector, end - .10 * cut_vector
    for depth, minimum in ((4, 4), (8, 3)):
        band = np.float32([first - depth * normal, last - depth * normal, last + depth * normal, first + depth * normal])
        if sum(_image_continues_across(image, band, width) for width in (220, 400, 600, 768, 1000)) < minimum:
            return []

    before, after = (side - 1) % 4, (side + 1) % 4
    continuations = [
        _continuing_flank(image, tile[i], tile[(i + 1) % 4], distances[i], extends_after=i == before)
        for i in (before, after)
    ]
    if any(span is None for span in continuations):
        return []
    _, terminal = _side_line(image, tile[side], tile[(side + 1) % 4], 11, collect_curve_segments=True)
    competing = np.float32(terminal.get(COMPETING_PRINT_EDGES, []))
    if competing.shape != (2, 2, 2) or min(np.linalg.norm(competing[:, 1] - competing[:, 0], axis=1)) < .80 * lengths[side]:
        return []
    separation = np.abs((competing[0] - competing[1]) @ normals[side])
    if (separation.min() < 2.4 or separation.max() > 8
            or sum(_dark_cool_transition(image, span) for span in competing) != 1):
        return []
    return [*continuations, *competing]
