"""Refine visible print masks with the registered local ONNX model."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from hashlib import sha256
from pathlib import Path

import cv2
import numpy as np

from app.pipeline.album_sleeve import sleeve_reflection_sides
from app.pipeline.geometry import normalized_polygon_overlap
from app.pipeline.model_registry import ModelArtifactError, verify_model_artifact

_INPUT_SIZE = 320
_MAX_CLAIM_PIXELS = 48_000_000
_MAX_DISPUTED_PIXELS = 2_000_000


@dataclass
class _Claim:
    left: int
    top: int
    mask: np.ndarray

    @property
    def region(self) -> tuple[slice, slice]:
        height, width = self.mask.shape
        return slice(self.top, self.top + height), slice(self.left, self.left + width)


@lru_cache(maxsize=2)
def _load_session(path: Path, digest: str):
    import onnxruntime as ort

    model = path.read_bytes()
    if sha256(model).hexdigest() != digest:
        raise ModelArtifactError("visible mask model changed during loading")
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    session = ort.InferenceSession(model, sess_options=options, providers=["CPUExecutionProvider"])
    inputs, outputs = session.get_inputs(), session.get_outputs()
    if (
        len(inputs) != 1 or inputs[0].name != "image_and_mask" or inputs[0].type != "tensor(float)"
        or tuple(inputs[0].shape[1:]) != (4, _INPUT_SIZE, _INPUT_SIZE)
        or len(outputs) != 1 or outputs[0].name != "mask_logits" or outputs[0].type != "tensor(float)"
        or tuple(outputs[0].shape[1:]) != (1, _INPUT_SIZE, _INPUT_SIZE)
    ):
        raise ModelArtifactError("unsupported visible mask model input or output")
    return session


def _largest_outline(mask: np.ndarray) -> np.ndarray | None:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    outline = cv2.approxPolyDP(contour, .70, True).reshape(-1, 2)
    return outline if len(outline) >= 4 and cv2.contourArea(outline) > 0 else None


def _local_outline(session, image: np.ndarray, polygon: np.ndarray) -> np.ndarray | None:
    height, width = image.shape[:2]
    scale = np.array([width - 1, height - 1])
    points = polygon * scale
    lower, upper = points.min(axis=0), points.max(axis=0)
    padding = np.maximum(12, .74 * (upper - lower))
    left, top = np.maximum(0, np.floor(lower - padding).astype(int))
    right, bottom = np.minimum([width, height], np.ceil(upper + padding + 1).astype(int))
    extent = np.array([right - left, bottom - top])
    if extent.min() < 5:
        return None

    origin = np.array([left, top])
    coordinates = (points - origin + .5) * (_INPUT_SIZE / extent) - .5
    prompt = np.zeros((_INPUT_SIZE, _INPUT_SIZE), dtype=np.uint8)
    cv2.fillPoly(prompt, [np.rint(coordinates).astype(np.int32)], 255)
    bgr = cv2.resize(image[top:bottom, left:right], (_INPUT_SIZE, _INPUT_SIZE), interpolation=cv2.INTER_AREA)
    tensor = np.ascontiguousarray(np.dstack((bgr, prompt)).transpose(2, 0, 1)[None], dtype=np.float32) / 255
    logits, = session.run(["mask_logits"], {"image_and_mask": tensor})
    if logits.shape != (1, 1, _INPUT_SIZE, _INPUT_SIZE) or not np.isfinite(logits).all():
        raise ModelArtifactError("invalid visible mask logits")
    interpolated = cv2.resize(logits[0, 0], tuple(extent), interpolation=cv2.INTER_LINEAR)
    outline = _largest_outline((interpolated > 0).astype(np.uint8))
    if outline is None:
        return None
    candidate = (outline + origin) / scale
    overlap = normalized_polygon_overlap(polygon, candidate)
    # Restrict changes to the same print even when the original mask found only its motif.
    return candidate if overlap.first_containment >= .72 and overlap.second_containment >= .45 else None


def _distances_to_seed(points: np.ndarray, columns: np.ndarray, rows: np.ndarray) -> np.ndarray:
    left = min(int(points[:, 0].min()), int(columns.min()))
    top = min(int(points[:, 1].min()), int(rows.min()))
    right = max(int(points[:, 0].max()), int(columns.max()))
    bottom = max(int(points[:, 1].max()), int(rows.max()))
    field = np.ones((bottom - top + 1, right - left + 1), dtype=np.uint8)
    cv2.fillPoly(field, [points - np.int32([left, top])], 0)
    distance = cv2.distanceTransform(field, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    return distance[rows - top, columns - left]


def resolve_mask_ownership(
    image_shape: tuple[int, ...], originals: list[np.ndarray], candidates: list[np.ndarray],
) -> list[np.ndarray]:
    """Assign contested source pixels only where one detector core is clearly closer."""
    if len(candidates) < 2:
        return candidates
    height, width = image_shape[:2]
    scale = np.array([width - 1, height - 1])
    vertices = [np.rint(outline * scale).astype(np.int32) for outline in candidates]
    bounds = [(points.min(axis=0), points.max(axis=0)) for points in vertices]
    overlapping = {
        index for index, (low, high) in enumerate(bounds)
        if any(index != peer and np.all(low <= other_high) and np.all(other_low <= high)
               for peer, (other_low, other_high) in enumerate(bounds))
    }
    if not overlapping:
        return candidates
    if sum(int(np.prod(bounds[index][1] - bounds[index][0] + 1)) for index in overlapping) > _MAX_CLAIM_PIXELS:
        raise ModelArtifactError("visible mask claims exceed the page budget")

    claims = {}
    coverage = np.zeros((height, width), dtype=np.uint8)
    for index in overlapping:
        (left, top), (right, bottom) = bounds[index]
        mask = np.zeros((bottom - top + 1, right - left + 1), dtype=np.uint8)
        cv2.fillPoly(mask, [vertices[index] - [left, top]], 1)
        claim = _Claim(left, top, mask)
        claims[index] = claim
        target = coverage[claim.region]
        cv2.add(target, mask, dst=target)
    disputed = coverage > 1
    if np.count_nonzero(disputed) > _MAX_DISPUTED_PIXELS:
        raise ModelArtifactError("visible mask overlap exceeds the page budget")
    flat_indices = np.flatnonzero(disputed)
    if not len(flat_indices):
        return candidates
    best = np.full(len(flat_indices), np.inf, dtype=np.float32)
    runner_up = best.copy()
    owners = np.full(len(flat_indices), -1, dtype=np.int32)
    participating = []
    for index, claim in claims.items():
        rows, columns = np.nonzero(disputed[claim.region] & claim.mask)
        if not len(rows):
            continue
        x, y = columns + claim.left, rows + claim.top
        indices = np.searchsorted(flat_indices, y * width + x)
        participating.append(index)
        seed = np.rint(originals[index] * scale).astype(np.int32)
        distances = _distances_to_seed(seed, x, y)
        nearer = distances < best[indices]
        runner_up[indices] = np.where(nearer, best[indices], np.minimum(distances, runner_up[indices]))
        best[indices[nearer]] = distances[nearer]
        owners[indices[nearer]] = index

    resolved = candidates.copy()
    separated = runner_up - best > 1
    for index in participating:
        claim = claims[index]
        rows, columns = np.nonzero(disputed[claim.region] & claim.mask)
        indices = np.searchsorted(flat_indices, (rows + claim.top) * width + columns + claim.left)
        claim.mask[rows, columns] = (owners[indices] == index) & separated[indices]
        outline = _largest_outline(claim.mask)
        if outline is not None:
            resolved[index] = (outline + [claim.left, claim.top]) / scale
    return resolved


def refine_visible_masks(
    image: np.ndarray, originals: list[np.ndarray], model_path: Path, manifest_path: Path,
) -> tuple[list[np.ndarray], str]:
    artifact = verify_model_artifact(model_path, manifest_path)
    digest = artifact["sha256"]
    if not originals:
        return [], digest
    session = _load_session(model_path, digest)
    proposals = [_local_outline(session, image, polygon) for polygon in originals]
    candidates = [original if proposal is None else proposal for original, proposal in zip(originals, proposals, strict=True)]
    if all(proposal is None for proposal in proposals):
        return originals, digest
    resolved = resolve_mask_ownership(image.shape, originals, candidates)
    scale = np.array(image.shape[1::-1]) - 1
    occupied = [np.float32(polygon * scale) for polygon in (*originals, *resolved)]
    return [
        _remove_sleeve_fold(image, original, refined, occupied)
        for original, refined in zip(originals, resolved, strict=True)
    ], digest


def _remove_sleeve_fold(
    image: np.ndarray, original: np.ndarray, refined: np.ndarray, occupied: list[np.ndarray],
) -> np.ndarray:
    if np.array_equal(original, refined):
        return refined
    sides = sleeve_reflection_sides(image, original, refined, occupied)
    if not sides:
        return refined
    scale = np.array(image.shape[1::-1]) - 1
    proposed = np.rint(refined * scale).astype(np.int32)
    primary = np.rint(original * scale).astype(np.int32)
    lower, upper = proposed.min(axis=0), proposed.max(axis=0)
    mask = np.zeros(tuple((upper - lower + 1)[::-1]), np.uint8)
    original_mask = cv2.fillPoly(np.zeros_like(mask), [primary - lower], 1)
    cv2.fillPoly(mask, [proposed - lower], 1)
    yy, xx = np.ogrid[lower[1] : upper[1] + 1, lower[0] : upper[0] + 1]
    for start, end in sides:
        edge = end - start
        side = edge[0] * (yy - start[1]) - edge[1] * (xx - start[0])
        mask[(original_mask == 0) & (side <= 3 * np.linalg.norm(edge))] = 0
    outline = _largest_outline(mask)
    return refined if outline is None else (outline + lower) / scale
