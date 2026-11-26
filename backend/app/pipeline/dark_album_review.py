"""Measure disputed claims and missed mounts on dark album backing."""

from dataclasses import dataclass
from itertools import combinations

import cv2
import numpy as np

from app.pipeline.edge_prints import _photo_texture
from app.pipeline.geometry import normalized_polygon_overlap, order_quad_cyclic
from app.pipeline.print_edges import _edge_band, _line_endpoints, _sample_lab, _side_line


DARK_ALBUM_REVIEW_REVISION = 2
DARK_ALBUM_PROPOSAL = "possible_dark_album_backing"
DARK_BACKING_OUTLINE_REVIEW = "DARK_BACKING_OUTLINE_REVIEW"

_UNIT = np.float32([[0, 0], [1, 0], [1, 1], [0, 1]])
_CORE_AXIS = np.linspace(.08, .92, 104, dtype=np.float32)
_CORE_GRID = np.stack(np.meshgrid(_CORE_AXIS, _CORE_AXIS), axis=-1).reshape(1, -1, 2)


@dataclass(frozen=True)
class DarkBackingEvidence:
    backing_traces: list[np.ndarray]
    borrowed_traces: list[np.ndarray]


def _red_mount_centers(image: np.ndarray) -> list[np.ndarray]:
    light, red, yellow = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2LAB))
    height, width = light.shape
    inside = np.s_[height // 20:-(height // 20), width // 20:-(width // 20)]
    dark = ((light <= 52) & (np.abs(red.astype(np.int16) - 128) <= 9)
            & (np.abs(yellow.astype(np.int16) - 128) <= 9))
    if dark[inside].mean() < .30:
        return []

    # Use compact red material as corner locations. A connected border
    # printed around artwork would produce one component, not four.
    mask = np.uint8((light >= 32) & (red >= 146))
    _, _, regions, centers = cv2.connectedComponentsWithStats(mask)
    return [
        center for (_, _, w, h, area), center in zip(regions[1:], centers[1:], strict=True)
        if 6 <= min(w, h) and max(w, h) <= 44 and .15 * w * h <= area <= .50 * w * h
    ]


def _dark_mounted_side_traces(image: np.ndarray, outline: np.ndarray) -> list[np.ndarray]:
    traces = []
    supported_sides = 0
    for start, end in zip(outline, np.roll(outline, -1, axis=0), strict=True):
        direction = end - start
        radius = int(np.clip(round(.052 * np.linalg.norm(direction)), 10, 96))
        spans = []
        for low, high in ((.12, .37), (.37, .63), (.63, .88)):
            first, last = start + low * direction, start + high * direction
            line, evidence = _side_line(image, first, last, radius)
            if line is None or evidence["ambiguous"] or evidence["support"] < .85:
                continue
            measured = np.float32(_line_endpoints(first, last, line))
            band = _edge_band(image, *measured, [-7, -5, -3, 2, 4, 6])
            if band is None:
                continue
            backing, face = (np.median(part, axis=1) for part in np.split(band, 2, axis=1))
            if np.mean((backing[:, 0] <= 55) & (face[:, 0] - backing[:, 0] >= 24)) >= .80:
                spans.append(measured)
        supported_sides += len(spans) >= 2
        traces.extend(spans)
    return traces if supported_sides >= 3 else []


def find_dark_mounted_print_review(
    image: np.ndarray, claims: list[np.ndarray],
) -> tuple[np.ndarray, list[np.ndarray]] | None:
    """Offer one possible independent print with contrasting mounts covering its tips."""
    height, width = image.shape[:2]
    if max(height, width) < 640 or min(height, width) < 380:
        return None
    reduction = min(1., 800 / max(height, width))
    working = cv2.resize(image, None, fx=reduction, fy=reduction, interpolation=cv2.INTER_AREA) if reduction < 1 else image
    centers = _red_mount_centers(working)
    if not 4 <= len(centers) <= 12:
        return None
    working_scale = np.float32(working.shape[1::-1]) - 1
    native_scale = np.float32([width - 1, height - 1])
    choices = []
    for indices in combinations(range(len(centers)), 4):
        try:
            seed = order_quad_cyclic(np.float32([centers[index] for index in indices]))
        except ValueError:
            continue
        if not .10 <= cv2.contourArea(seed) / (working.shape[0] * working.shape[1]) <= .34:
            continue
        directions = np.roll(seed, -1, axis=0) - seed
        lengths = np.linalg.norm(directions, axis=1)
        if lengths.min() < 60 or not .48 <= lengths[::2].sum() / lengths[1::2].sum() <= 2.1:
            continue
        unit = directions / lengths[:, None]
        if np.max(np.abs((unit * np.roll(unit, 1, axis=0)).sum(axis=1))) > .25:
            continue
        # A mount at an interior join can belong to a separate print.
        if any(cv2.pointPolygonTest(seed, tuple(center), True) >= -6
               for index, center in enumerate(centers) if index not in indices):
            continue
        normalized = seed / working_scale
        if any(normalized_polygon_overlap(normalized, claim).first_containment > .025 for claim in claims):
            continue
        if not _photo_texture(working, seed):
            continue
        outline = normalized * native_scale
        spans = _dark_mounted_side_traces(image, outline)
        if spans:
            choices.append((outline, spans))
            if len(choices) > 1:
                return None
    return choices[0] if choices else None


def _dark_core(image: np.ndarray, quad: np.ndarray) -> bool:
    points = cv2.perspectiveTransform(_CORE_GRID, cv2.getPerspectiveTransform(_UNIT, quad))[0]
    lab = _sample_lab(image, points)
    low, middle, upper, high, peak = np.quantile(lab[:, 0], [.10, .50, .90, .95, .99])
    chroma = np.linalg.norm(np.median(lab[:, 1:], axis=0) - 128)
    return bool(middle <= 40 and upper - low <= 36 and high <= 58 and peak <= 73 and chroma <= 8)


def _has_narrow_seam(image: np.ndarray, stations: np.ndarray, inward: np.ndarray) -> bool:
    """Strong, nearly collinear changes can be a separate dark sheet."""
    offsets = np.arange(-56, 57, 2, dtype=np.float32)
    points = stations[:, None] + offsets[None, :, None] * inward
    valid = ((points >= 0) & (points <= np.array(image.shape[1::-1]) - 1)).all(axis=(1, 2))
    if valid.mean() < .80:
        return True
    lab = _sample_lab(image, points[valid].reshape(-1, 2)).reshape(-1, len(offsets), 3)
    changes = np.linalg.norm(lab[:, 3:] - lab[:, :-3], axis=2) >= 9
    extent = np.linspace(-.5, .5, len(changes))
    rows = np.arange(len(changes))[:, None]
    centre = np.arange(5, len(offsets) - 8)[None, :]
    for drift in (-9, -6, -3, 0, 3, 6, 9):
        columns = centre + np.rint(extent * drift / 2).astype(int)[:, None]
        supported = changes[rows, columns]
        for column in np.flatnonzero(supported.mean(axis=0) >= .61):
            if min(part.mean() for part in np.array_split(supported[:, column], 4)) >= .44:
                return True
    return False


def _visible_spans(support: np.ndarray, stations: np.ndarray, length: float) -> list[np.ndarray]:
    intervals = np.flatnonzero(np.diff(np.r_[False, support, False])).reshape(-1, 2)
    spans = [stations[[first, end - 1]] for first, end in intervals if end - first >= 8]
    return [span for span in spans if np.linalg.norm(span[1] - span[0]) >= max(18., .05 * length)]


def inspect_dark_album_backing(image: np.ndarray, quad: np.ndarray, peers: list[np.ndarray]) -> DarkBackingEvidence | None:
    """Keep an ambiguous, weak proposal identifiable without calling it a print."""
    edges = np.roll(quad, -1, axis=0) - quad
    if not peers or np.linalg.norm(edges, axis=1).min() < 320 or not _dark_core(image, quad):
        return None
    limit = np.array(image.shape[1::-1]) - 1
    backing_count = borrowed_count = 0
    backing_traces = []
    borrowed_traces = []
    for start, direction in zip(quad, edges, strict=True):
        length = float(np.linalg.norm(direction))
        inward = np.float32([-direction[1], direction[0]]) / length
        stations = start + np.linspace(.10, .90, 128, dtype=np.float32)[:, None] * direction
        points = stations[:, None] + np.float32([-42, -24, -12, -6, 6, 12, 24, 42])[None, :, None] * inward
        valid = ((points >= 0) & (points <= limit)).all(axis=(1, 2))
        if valid.sum() < 80:
            continue
        band = _sample_lab(image, points[valid].reshape(-1, 2)).reshape(-1, 8, 3)
        exterior, interior = (np.median(part, axis=1) for part in np.split(band, 2, axis=1))
        lightness = exterior[:, 0] - interior[:, 0]
        if np.mean(lightness <= -35) >= .20:
            return None

        matches = np.linalg.norm(exterior - interior, axis=1) <= 8
        if matches.mean() >= .71 and min(part.mean() for part in np.array_split(matches, 4)) >= .52:
            if _has_narrow_seam(image, stations[valid], inward):
                return None
            backing_count += 1
            complete = valid.copy()
            complete[valid] = matches
            backing_traces.extend(_visible_spans(complete, stations, length))

        reversed_step = (lightness >= 35) & (exterior[:, 0] >= 83)
        possible = np.flatnonzero(reversed_step)
        outside = points[valid, 0]
        near_peer = np.zeros(len(band), dtype=bool)
        for peer in peers:
            for index in possible:
                if cv2.pointPolygonTest(peer, tuple(outside[index]), True) >= -16:
                    near_peer[index] = True
        if near_peer.mean() >= .60:
            borrowed_count += 1
            complete = valid.copy()
            complete[valid] = near_peer
            borrowed_traces.extend(_visible_spans(complete, stations, length))
    if backing_count + borrowed_count < 3 or not backing_traces or not borrowed_traces:
        return None
    return DarkBackingEvidence(backing_traces, borrowed_traces)
