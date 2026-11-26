"""Measure an exposed curved side without replacing the saved rectangular warp."""

from dataclasses import dataclass

import cv2
import numpy as np

from app.pipeline.print_edges import _PRINT_SIDES, _edge_band, _sample_lab
from app.pipeline.protected_regions import region_box_in_pixels


POSSIBLE_BOWED_PRINT_EDGE = "possible_bowed_print_edge"
POSSIBLE_DISPLACED_PRINT_EDGES = "possible_displaced_print_edges"
POSSIBLE_ALBUM_LAYER_EDGE = "possible_album_layer_edge"
POSSIBLE_LOCAL_PRINT_EDGE = "possible_local_print_edge"
POSSIBLE_MOUNTED_EDGE_STRIP = "possible_mounted_edge_strip"
MEASURED_PRINT_PREVIEW_SOURCES = (
    POSSIBLE_BOWED_PRINT_EDGE, POSSIBLE_DISPLACED_PRINT_EDGES, POSSIBLE_ALBUM_LAYER_EDGE, POSSIBLE_LOCAL_PRINT_EDGE,
    POSSIBLE_MOUNTED_EDGE_STRIP,
)


@dataclass(frozen=True)
class MeasuredPrintPreview:
    side: int
    traces: list[np.ndarray]
    outline: np.ndarray
    search_band: np.ndarray
    inset_band: np.ndarray | None = None


def _fit_edge_candidates(
    fractions: np.ndarray, offsets: np.ndarray, energy: np.ndarray,
) -> tuple[np.polynomial.Polynomial, np.ndarray, np.ndarray] | None:
    initial = energy.argmax(axis=1)
    observed = energy.max(axis=1) > 0
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
    return model, chosen, observed


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
    offsets = depths[centers]
    curve = _fit_edge_candidates(fractions, offsets, energy)
    if curve is None:
        return None
    model, chosen, observed = curve
    measured = offsets[chosen]
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
) -> MeasuredPrintPreview | None:
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
        results.append(MeasuredPrintPreview(index, traces, outline, band))
    return results[0] if len(results) == 1 else None


def _displaced_edge(
    image: np.ndarray, start: np.ndarray, end: np.ndarray, fractions: np.ndarray, bounds: tuple[int, int],
) -> tuple[np.polynomial.Polynomial, np.ndarray, np.ndarray] | None:
    depths = np.arange(bounds[0] - 10, bounds[1] + 13, .5, dtype=np.float32)
    band = _edge_band(image, start, end, (-depths).tolist(), fractions=fractions)
    if band is None:
        return None
    centers = np.flatnonzero((depths >= bounds[0]) & (depths <= bounds[1]))
    step = band[:, centers + 3, 0] - band[:, centers - 3, 0]
    exterior = band[:, centers[:, None] + np.arange(7, 21)]
    backing = np.median(exterior, axis=2)
    variation = np.percentile(np.linalg.norm(exterior - backing[:, :, None], axis=3), 80, axis=2)
    inside = np.median(band[:, centers[:, None] + np.arange(-18, -6), 0], axis=2)
    contrast = backing[..., 0] - inside
    pale = (backing[..., 0] >= 169) & (np.linalg.norm(backing[..., 1:] - 128, axis=2) <= 20)
    supported = pale & (step >= 17) & (contrast >= 21) & (variation <= 14)
    energy = np.where(supported, step + .25 * np.minimum(160, contrast) - .7 * variation, 0)
    offsets = depths[centers]
    curve = _fit_edge_candidates(fractions, offsets, energy)
    if curve is None:
        return None
    model, chosen, observed = curve
    measured = offsets[chosen]
    if (np.quantile(np.abs(measured[observed] - model(fractions[observed])), .95) > 1.35
            or np.median(step[np.arange(len(chosen))[observed], chosen[observed]]) < 40):
        return None
    return model, measured, observed


def _backing_continues_at_corner(
    image: np.ndarray, start: np.ndarray, edge: np.ndarray, outward: np.ndarray, model: np.polynomial.Polynomial,
) -> bool:
    fractions = np.float32([-.045, -.032, -.022, -.012, .012, .022, .032, .045])
    depths = np.float32([-6, -4, 4, 6, 13, 16, 19])
    for positions in (fractions, 1 - fractions):
        points = start + positions[:, None, None] * edge + (model(positions)[:, None, None]
                   + depths[None, :, None]) * outward
        if ((points < 0) | (points > np.array(image.shape[1::-1]) - 1)).any():
            continue
        rows = _sample_lab(image, points.reshape(-1, 2).astype(np.float32)).reshape(8, len(depths), 3)
        beyond, inside = rows[:4], rows[4:]
        local_step = inside[:, 2:4, 0].mean(1) - inside[:, :2, 0].mean(1)
        continuing_step = beyond[:, 2:4, 0].mean(1) - beyond[:, :2, 0].mean(1)
        inside_stock = np.median(inside[:, 4:], axis=(0, 1))
        adjacent_stock = np.median(beyond[:, 4:], axis=(0, 1))
        if ((local_step >= 35).sum() >= 3 and np.max(np.abs(continuing_step)) <= 12
                and np.min(beyond[..., 0]) >= 174 and np.linalg.norm(inside_stock - adjacent_stock) <= 9):
            return True
    return False


def _has_competing_outer_rim(
    image: np.ndarray, stations: np.ndarray, outward: np.ndarray,
) -> bool:
    # A bright stock rim can terminate well beyond the strongest image
    # transition. Look for a distributed second darkening step in that strip.
    depths = np.arange(-.5, 27.25, .25, dtype=np.float32)
    points = stations[:, None] + depths[None, :, None] * outward
    if ((points < 0) | (points > np.array(image.shape[1::-1]) - 1)).any():
        return True
    light = _sample_lab(image, points.reshape(-1, 2).astype(np.float32))[:, 0].reshape(len(stations), -1)
    drops = light[:, :-12].astype(np.float32) - light[:, 12:]
    # Subtract the stronger adjacent three-pixel drop. This exposes a narrow
    # reversal on a gradual brightening ramp, including a pale paper rim.
    reversal = drops[:, 12:-12] - np.maximum(drops[:, :-24], drops[:, 24:])
    for signal in (drops[:, 12:-12], reversal):
        local = np.lib.stride_tricks.sliding_window_view(signal, 7, axis=1).max(axis=2)
        quartile, median = np.quantile(local, [.25, .50], axis=0)
        if np.any((quartile >= 2) & (median >= 3.5)):
            return True
    return False


def _keep_protected_content(
    stations: np.ndarray, outward: np.ndarray, depth: np.ndarray, regions: list[dict], width: int, height: int,
) -> None:
    inset = stations - depth[:, None] * outward
    strip = np.stack([stations[:-1], stations[1:], inset[:-1], inset[1:]], axis=1)
    lower, upper = strip.min(axis=1), strip.max(axis=1)
    for region in regions:
        box = region_box_in_pixels(region, width, height)
        if box is None:
            continue
        x1, y1, x2, y2 = box
        # Leave an extra native pixel for outline simplification and sampling.
        pad = float(region.get("margin_ratio", .08)) * max(x2 - x1, y2 - y1) + 1
        intersects = ((lower < [x2 + pad, y2 + pad]) & (upper > [x1 - pad, y1 - pad])).all(axis=1)
        depth[np.r_[intersects, False] | np.r_[False, intersects]] = 0


def suggest_displaced_print_preview(
    image: np.ndarray, quad: np.ndarray, evidence: dict, neighbors: list[np.ndarray], *,
    protected_regions: list[dict] | None = None,
) -> MeasuredPrintPreview | None:
    """Compare a recovered image edge and an inset transition on adjacent sides."""
    sides = [evidence.get(name) or {} for name in _PRINT_SIDES]
    weak = [i for i, side in enumerate(sides) if side.get("support", 0) <= .36
            and .65 <= (side.get("outward_search") or {}).get("support", 0) <= .91]
    disputed = [i for i, side in enumerate(sides) if side.get("ambiguous") and side.get("support", 0) >= .66]
    if (len(weak) != 1 or len(disputed) != 1 or (weak[0] - disputed[0]) % 4 not in (1, 3)
            or np.linalg.norm(np.roll(quad, -1, axis=0) - quad, axis=1).min() < 520
            or any((side.get("support", 0) < .94 or (side.get("corner_support") or 0) < .85)
                   for i, side in enumerate(sides) if i not in {*weak, *disputed})):
        return None
    fractions = np.linspace(.03, .97, 256, dtype=np.float32)
    traces, sections, bands = [], {}, []
    for side, bounds in ((weak[0], (2, 21)), (disputed[0], (-38, -7))):
        start, end = quad[[side, (side + 1) % 4]]
        edge = end - start
        outward = np.float32([edge[1], -edge[0]]) / np.linalg.norm(edge)
        near, far = (-2, 34) if bounds[0] > 0 else (-38, 0)
        proof = np.float32([start + near * outward, end + near * outward,
                            end + far * outward, start + far * outward])
        if any(cv2.intersectConvexConvex(proof, cv2.convexHull(peer))[0] > 1 for peer in neighbors):
            return None
        result = _displaced_edge(image, start, end, fractions, bounds)
        if result is None:
            return None
        model, measured, observed = result
        fitted = model(fractions)
        stations = start + fractions[:, None] * edge
        if bounds[0] > 0:
            bow = abs(model(.5) - (model(0) + model(1)) / 2)
            if not 2.5 <= bow <= 10 or np.quantile(measured[observed], .10) < 3.2:
                return None
            verified = np.maximum(0, np.minimum(measured, fitted) - 1.5)
        else:
            middle = observed & (fractions > .10) & (fractions < .89)
            rim_stations = stations[middle] + fitted[middle, None] * outward
            if (np.quantile(measured[observed], .90) > -10
                    or not _backing_continues_at_corner(image, start, edge, outward, model)
                    or _has_competing_outer_rim(image, rim_stations, outward)):
                return None
            # Keep a native collar outside the strong scene transition.
            verified = np.maximum(0, -np.maximum(measured, fitted) - 2.5)
        verified[~observed] = 0
        depth = np.minimum.reduce([verified, np.r_[0., verified[:-1]], np.r_[verified[1:], 0.]])
        if bounds[0] < 0 and protected_regions:
            _keep_protected_content(stations, outward, depth, protected_regions, *image.shape[1::-1])
        if depth.sum() * np.linalg.norm(edge) * (fractions[1] - fractions[0]) < 280:
            return None
        path = stations + (depth if bounds[0] > 0 else -depth)[:, None] * outward
        sections[side] = cv2.approxPolyDP(np.float32(path), .22, False)[:, 0]
        hits = np.flatnonzero(observed)
        breaks = (np.diff(hits) != 1) | (np.abs(np.diff(measured[hits])) > 2)
        measured_line = stations + measured[:, None] * outward
        traced = [cv2.approxPolyDP(np.float32(measured_line[run]), .35, False)[:, 0]
                  for run in np.split(hits, np.flatnonzero(breaks) + 1) if len(run) >= 6]
        if sum(np.linalg.norm(span[-1] - span[0]) for span in traced) < .75 * np.linalg.norm(edge):
            return None
        traces.extend(traced)
        bands.append(proof)
    outline = np.vstack([points for i, corner in enumerate(quad)
                         for points in (corner[None], sections.get(i, np.empty((0, 2), dtype=np.float32)))])
    return MeasuredPrintPreview(weak[0], traces, outline, bands[0], bands[1])
