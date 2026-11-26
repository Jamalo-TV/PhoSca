"""Measure dark backing and borrowed photo edges for count review."""

from dataclasses import dataclass

import cv2
import numpy as np

from app.pipeline.print_edges import _sample_lab


DARK_ALBUM_REVIEW_REVISION = 1
DARK_ALBUM_PROPOSAL = "possible_dark_album_backing"
DARK_BACKING_OUTLINE_REVIEW = "DARK_BACKING_OUTLINE_REVIEW"

_UNIT = np.float32([[0, 0], [1, 0], [1, 1], [0, 1]])
_CORE_AXIS = np.linspace(.08, .92, 104, dtype=np.float32)
_CORE_GRID = np.stack(np.meshgrid(_CORE_AXIS, _CORE_AXIS), axis=-1).reshape(1, -1, 2)


@dataclass(frozen=True)
class DarkBackingEvidence:
    backing_traces: list[np.ndarray]
    borrowed_traces: list[np.ndarray]


def dark_album_neighbor_masks(segmentation: dict) -> list[dict]:
    """Keep pending areas in the source coverage checked by adjacent crops."""
    pending = (segmentation.get("fusion") or {}).get("deferred_print_proposals") or []
    return [offer["source_mask"] for offer in pending if offer.get("reason") == DARK_ALBUM_PROPOSAL
            and isinstance(offer.get("source_mask"), dict)]


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
