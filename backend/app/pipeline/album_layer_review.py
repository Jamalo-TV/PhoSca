"""Compare a held print crop against an observed continuing album layer."""

import cv2
import numpy as np

from app.pipeline.bowed_edge_review import (
    MeasuredPrintPreview, _displaced_edge, _fit_edge_candidates, _keep_protected_content,
)
from app.pipeline.print_edges import _PRINT_SIDES, _edge_band, _sample_lab


def _layer_steps(band: np.ndarray, columns: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    light = band[..., 0].astype(np.float32)
    step = light[:, columns + 3] - light[:, columns - 3]
    dark = np.median(band[:, columns[:, None] + np.arange(-14, -5)], axis=2)
    pale = np.median(band[:, columns[:, None] + np.arange(8, 25)], axis=2)
    rise = pale[..., 0] - dark[..., 0]
    supported = ((step >= 16) & (rise >= 21) & (pale[..., 0] >= 174)
                 & (np.linalg.norm(pale[..., 1:] - 128, axis=2) < 19))
    return np.where(supported, step + .25 * rise, 0), pale, pale[..., 1:] - dark[..., 1:]


def _another_backing_layer(
    image: np.ndarray, stations: np.ndarray, inward: np.ndarray, seam_depth: np.ndarray,
    dark: np.ndarray, chroma_direction: float,
) -> bool:
    # A shadow can reproduce the first edge. Require a separate light layer
    # beyond the dark strip on the side where the photograph has ended.
    distances = np.arange(8., 35.5, .5, dtype=np.float32)
    points = stations[:, None] + (seam_depth[:, None, None] - distances[None, :, None]) * inward
    limit = np.array(image.shape[1::-1]) - 1
    inside = ((points >= 0) & (points <= limit)).all(axis=2) & (distances <= seam_depth[:, None])
    values = _sample_lab(image, np.clip(points, 0, limit).reshape(-1, 2).astype(np.float32))
    values = values.reshape(len(stations), len(distances), 3)
    bright = inside & (values[..., 0] >= np.maximum(dark[:, :1] + 24, 170))
    first = bright.argmax(axis=1)
    offsets = np.minimum(first[:, None] + [3, 5, 7, 9], len(distances) - 1)
    rows = np.arange(len(stations))[:, None]
    sustained = bright.any(axis=1) & (first + 9 < len(distances)) & bright[rows, offsets].all(axis=1)
    if sustained.mean() < .75:
        return False
    following = np.median(values[rows, offsets], axis=1)
    shift = np.median((following - dark)[sustained, 1:], axis=0)
    return bool(shift[0] * chroma_direction >= 3. and np.linalg.norm(shift) >= 3.)


def _continuing_album_layer(
    image: np.ndarray, start: np.ndarray, edge: np.ndarray, inward: np.ndarray,
    model: np.polynomial.Polynomial, scene: np.polynomial.Polynomial, stock: np.ndarray, chroma_direction: float,
    fractions: np.ndarray, measured_layer: np.ndarray, supported_layer: np.ndarray,
) -> tuple[list[np.ndarray], tuple[float, float]] | None:
    # Compare the source outside either end of the print. A line contained
    # inside the print may be a frame or a feature of its printed image.
    extension = np.linspace(-.22, -.025, 54, dtype=np.float32)
    offsets = np.arange(-14., 44., .5, dtype=np.float32)
    columns = np.flatnonzero(np.abs(offsets) <= 6)
    for positions, extent in ((extension, (-.22, 1.)), (1 - extension, (0., 1.22))):
        middle = start + positions[:, None] * edge
        predicted = model(positions)
        points = middle[:, None] + (predicted[:, None, None] + offsets[None, :, None]) * inward
        if ((points < 0) | (points > np.array(image.shape[1::-1]) - 1)).any():
            continue
        band = _sample_lab(image, points.reshape(-1, 2).astype(np.float32)).reshape(len(positions), len(offsets), 3)
        energy, pale, chroma = _layer_steps(band, columns)
        valid = energy > 0
        valid &= np.linalg.norm(pale - stock, axis=2) <= 19
        chosen = np.argmax(np.where(valid, energy, 0), axis=1)
        observed = valid.any(axis=1)
        # The inner contrast must be local to the photograph. The surrounding
        # backing stays light where that inner transition would continue.
        scene_depth = -scene(positions) - predicted
        vicinity = np.abs(offsets - scene_depth[:, None]) <= 4
        clear_backing = (vicinity & (band[..., 0] >= 174)).sum(axis=1) >= 12
        observed &= clear_backing
        if observed.mean() < .85 or min(part.mean() for part in np.array_split(observed, 4)) < .75:
            continue
        rows = np.flatnonzero(observed)
        shift = np.median(chroma[rows, chosen[observed]], axis=0)
        if shift[0] * chroma_direction < 1.5 or np.linalg.norm(shift) < 2.:
            continue
        measured = offsets[columns[chosen]]
        if np.ptp(np.quantile(measured[observed], [.05, .95])) > 4.5:
            continue
        selected = columns[chosen[observed]]
        dark = np.median(band[rows[:, None], selected[:, None] + np.arange(-14, -5)], axis=1)
        depth = predicted + measured
        if not _another_backing_layer(image, middle[observed], inward, depth[observed], dark, chroma_direction):
            continue
        # The curve fits on both sides of the corner must agree. A small
        # raised rim or mount can replace only the first part of the layer.
        corner = 0. if extent[0] < 0 else 1.
        near_inside = supported_layer & (np.abs(fractions - corner) <= .105)
        near_beyond = observed & (np.abs(positions - corner) <= .145)
        if min(np.count_nonzero(near_inside), np.count_nonzero(near_beyond)) < 12:
            continue
        inner = np.polynomial.Polynomial.fit(fractions[near_inside], measured_layer[near_inside], 1)
        outer = np.polynomial.Polynomial.fit(positions[near_beyond], depth[near_beyond], 1)
        if abs(inner(corner) - outer(corner)) > 2.5:
            continue
        return _observed_spans(middle, inward, depth, observed), extent
    return None


def _observed_spans(
    stations: np.ndarray, inward: np.ndarray, depth: np.ndarray, valid: np.ndarray,
) -> list[np.ndarray]:
    hits = np.flatnonzero(valid)
    breaks = (np.diff(hits) != 1) | (np.abs(np.diff(depth[hits])) > 2)
    measured = stations + depth[:, None] * inward
    return [cv2.approxPolyDP(np.float32(measured[run]), .35, False)[:, 0]
            for run in np.split(hits, np.flatnonzero(breaks) + 1) if len(run) >= 6]


def suggest_album_layer_preview(
    image: np.ndarray, quad: np.ndarray, evidence: dict, neighbors: list[np.ndarray], *,
    protected_regions: list[dict] | None = None,
) -> MeasuredPrintPreview | None:
    edges = np.roll(quad, -1, axis=0) - quad
    lengths = np.linalg.norm(edges, axis=1)
    sides = [evidence.get(name) or {} for name in _PRINT_SIDES]
    if lengths.min() < 520 or sum(side.get("support", 0) >= .94 for side in sides) < 2:
        return None
    image_limit = np.array(image.shape[1::-1]) - 1
    fractions = np.linspace(.03, .97, 256, dtype=np.float32)
    grid = np.arange(2., 58.5, .5, dtype=np.float32)
    columns = np.flatnonzero((grid >= 13) & (grid <= 44))
    offsets = grid[columns]
    candidates = []
    for index, (side, start, edge, length) in enumerate(zip(sides, quad, edges, lengths, strict=True)):
        if (side.get("ambiguous") or not .70 <= side.get("support", 0) <= .92
                or not .15 <= (side.get("corner_support") or 0) <= .65
                or max(side.get("start_support") or 0, side.get("end_support") or 0) < .80):
            continue
        inward = np.float32([-edge[1], edge[0]]) / length
        axis = int(np.argmax(np.abs(inward)))
        frame = (start + np.float32([[0.], [1.]]) * edge)[:, axis]
        if inward[axis] < 0:
            frame = image_limit[axis] - frame
        if abs(inward[axis]) < .96 or frame.min() < 3 or frame.max() > min(48, .065 * length):
            continue
        scene = _displaced_edge(image, start, start + edge, fractions, (-83, -43))
        if scene is None:
            continue
        scene_model, _, scene_supported = scene
        band = _edge_band(image, start, start + edge, grid.tolist(), fractions=fractions)
        if band is None:
            continue
        energy, pale, chroma = _layer_steps(band, columns)
        gap = -scene_model(fractions)[:, None] - offsets
        energy *= (gap >= 15) & (gap <= 44) & scene_supported[:, None]
        fit = _fit_edge_candidates(fractions, offsets, energy)
        if fit is None:
            continue
        layer, chosen, supported = fit
        measured = offsets[chosen]
        fitted = layer(fractions)
        rows = np.flatnonzero(supported)
        peaks = columns[chosen[supported]]
        light = band[..., 0].astype(np.float32)
        narrow = light[rows, peaks + 3] - light[rows, peaks - 3]
        wide = light[rows, peaks + 8] - light[rows, peaks - 8]
        shift = np.median(chroma[rows, chosen[supported]], axis=0)
        if (np.quantile(np.abs(measured[supported] - fitted[supported]), .95) > 1.55
                or np.quantile(narrow / np.maximum(wide, 1.), .25) < .53
                or abs(shift[0]) < .75 or np.linalg.norm(shift) < 2.
                or np.ptp(np.quantile((-scene_model(fractions) - fitted)[supported], [.05, .95])) > 7):
            continue
        stock = np.median(pale[rows, chosen[supported]], axis=0)
        extension = _continuing_album_layer(
            image, start, edge, inward, layer, scene_model, stock, np.sign(shift[0]), fractions, measured, supported,
        )
        if extension is None:
            continue
        extension_traces, (first, last) = extension
        span_start, span_end = start + first * edge, start + last * edge
        proof = np.float32([span_start, span_end, span_end + 83 * inward, span_start + 83 * inward])
        if any(cv2.intersectConvexConvex(proof, cv2.convexHull(peer))[0] > 1 for peer in neighbors):
            continue
        stations = start + fractions[:, None] * edge
        verified = np.where(supported, np.maximum(0, np.minimum(measured, fitted) - 3.), 0)
        depth = np.minimum.reduce([verified, np.r_[0., verified[:-1]], np.r_[verified[1:], 0.]])
        if protected_regions:
            _keep_protected_content(stations, -inward, depth, protected_regions, *image.shape[1::-1])
        if depth.sum() * length * (fractions[1] - fractions[0]) < 1_500:
            continue
        traces = _observed_spans(stations, inward, measured, supported)
        if not extension_traces or sum(np.linalg.norm(s[-1] - s[0]) for s in traces) < .75 * length:
            continue
        path = cv2.approxPolyDP(np.float32(stations + depth[:, None] * inward), .22, False)[:, 0]
        outline = np.vstack([quad[:index + 1], path, quad[index + 1:]])
        candidates.append(MeasuredPrintPreview(index, [*traces, *extension_traces], outline, proof))
    return candidates[0] if len(candidates) == 1 else None
