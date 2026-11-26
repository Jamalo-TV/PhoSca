"""Measure an exposed curved side without replacing the saved rectangular warp."""

from dataclasses import dataclass

import cv2
import numpy as np

from app.pipeline.print_edges import _PRINT_SIDES, _edge_band


POSSIBLE_BOWED_PRINT_EDGE = "possible_bowed_print_edge"


@dataclass(frozen=True)
class BowedPrintPreview:
    side: int
    traces: list[np.ndarray]
    outline: np.ndarray
    search_band: np.ndarray


def _trace_side(
    image: np.ndarray, start: np.ndarray, end: np.ndarray, fractions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    depths = np.arange(-8., 30.01, .5, dtype=np.float32)
    band = _edge_band(image, start, end, (-depths).tolist(), fractions=fractions)
    if band is None:
        return None
    centers = np.arange(7, 39)
    step = band[:, centers + 2] - band[:, centers - 2]
    contrast = np.linalg.norm(step, axis=2)
    exterior = band[:, centers[:, None] + np.arange(6, 37)]
    backing = np.median(exterior, axis=2)
    difference = np.linalg.norm(exterior - backing[:, :, None], axis=3)
    near = np.linalg.norm(band[:, centers + 2] - backing, axis=2)
    pale = (backing[:, :, 0] >= 190) & (np.linalg.norm(backing[:, :, 1:] - 128, axis=2) <= 16)
    energy = np.maximum(0, contrast - .65 * near - .5 * np.percentile(difference, 75, axis=2))
    energy *= pale & (contrast >= 9) & (step[:, :, 0] >= 5) & (near <= 13)

    initial = energy.argmax(axis=1)
    observed = energy.max(axis=1) > 0
    offsets = depths[centers]
    groups = np.array_split(np.arange(len(fractions)), 8)
    anchors = []
    for group in groups:
        hits = group[observed[group]]
        if len(hits) >= .42 * len(group):
            anchors.append((np.median(fractions[hits]), np.median(offsets[initial[hits]])))
    if len(anchors) < 6:
        return None
    positions, ordinates = np.array(anchors).T
    model = np.polynomial.Polynomial.fit(positions, ordinates, 2)
    for _ in range(3):
        separation = model(fractions)[:, None] - offsets
        score = energy * np.exp(-.16 * separation ** 2) * (np.abs(separation) <= 2.15)
        chosen = score.argmax(axis=1)
        weights = score.max(axis=1)
        observed = weights > 0
        if observed.mean() < .87 or min(observed[g].mean() for g in groups) < .65:
            return None
        measured = offsets[chosen]
        model = np.polynomial.Polynomial.fit(fractions[observed], measured[observed], 2,
                                             w=np.sqrt(np.minimum(weights[observed], 35)))
    fitted = model(fractions)
    median_error, tail_error = np.quantile(np.abs(measured[observed] - fitted[observed]), [.50, .95])
    line = np.polynomial.Polynomial.fit(fractions[observed], measured[observed], 1)
    line_error = np.quantile(np.abs(measured[observed] - line(fractions[observed])), .95)
    bow = abs(model(.5) - (model(0) + model(1)) / 2)
    if (not 2 <= bow <= 8 or median_error > .60 or tail_error > 1.35
            or line_error < tail_error + .55 or np.mean(measured[observed] > 1.75) < .23):
        return None

    # Compare the whole exterior band as well as the transition itself. A
    # nearby piece of stock can produce an equally strong inner peak.
    collar = difference[np.arange(len(fractions)), chosen]
    if np.quantile(collar[observed].max(axis=1), .95) > 10:
        return None
    return measured, fitted, observed


def suggest_bowed_print_preview(
    image: np.ndarray, quad: np.ndarray, evidence: dict, neighbors: list[np.ndarray],
) -> BowedPrintPreview | None:
    sides = [evidence.get(name) or {} for name in _PRINT_SIDES]
    if (np.linalg.norm(np.roll(quad, -1, axis=0) - quad, axis=1).min() < 300
            or sum(side.get("support", 0) >= .90 for side in sides) < 2):
        return None
    fractions = np.linspace(.025, .975, 228, dtype=np.float32)
    results = []
    for index, side in enumerate(sides):
        if (side.get("ambiguous") or not .55 <= side.get("support", 0) <= .83
                or (side.get("outward_search") or {}).get("print_rim_supported")):
            continue
        start, end = quad[[index, (index + 1) % 4]]
        tangent = end - start
        length = np.linalg.norm(tangent)
        if length < 500:
            continue
        outward = np.float32([tangent[1], -tangent[0]]) / length
        band = np.float32([start - 8 * outward, end - 8 * outward, end + 30 * outward, start + 30 * outward])
        if any(cv2.intersectConvexConvex(band, cv2.convexHull(peer))[0] > 1. for peer in neighbors):
            continue
        trace = _trace_side(image, start, end, fractions)
        if trace is None:
            continue
        measured, fitted, observed = trace
        stations = start + fractions[:, None] * tangent
        hits = np.flatnonzero(observed)
        paths = np.split(hits, np.flatnonzero((np.diff(hits) != 1) | (np.abs(np.diff(measured[hits])) > 2)) + 1)
        traces = [cv2.approxPolyDP(np.float32(stations[path] + measured[path, None] * outward), .35, False)[:, 0]
                  for path in paths if len(path) >= 6]
        if sum(np.linalg.norm(path[-1] - path[0]) for path in traces) < .70 * length:
            continue

        # Expand locally, with a one-pixel support margin on either side of
        # each station. Where evidence ends, the preview returns to the old cut.
        verified = np.where(observed, np.clip(np.minimum(fitted, measured) - 1.25, 0, None), 0)
        depth = np.minimum.reduce([verified, np.r_[0., verified[:-1]], np.r_[verified[1:], 0.]])
        if depth.sum() * length * (fractions[1] - fractions[0]) < 280:
            continue
        expanded = cv2.approxPolyDP(np.float32(stations + depth[:, None] * outward), .18, False)[:, 0]
        outline = np.vstack([quad[:index + 1], expanded, quad[index + 1:]])
        results.append(BowedPrintPreview(index, traces, outline, band))
    return results[0] if len(results) == 1 else None
