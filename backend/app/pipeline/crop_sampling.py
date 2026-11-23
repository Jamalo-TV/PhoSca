"""Keep a bilinear crop's edge samples on its side of the fitted outline."""

import cv2
import numpy as np


OWNED_BILINEAR = "owned_bilinear"
_PHASES = cv2.INTER_TAB_SIZE
_TAP_OFFSETS = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
_CANDIDATE_OFFSETS = np.array([(x, y) for y in range(-2, 3) for x in range(-2, 3)])


def _edge_strips(lines: np.ndarray, width: int, height: int):
    # Each inset half-plane delineates pixels where all bilinear taps are safe.
    # Bound the affected strips at both ends, since the inset lines may slant.
    top, right, bottom, left = lines
    slopes = np.array([top[1], -right[0], -bottom[1], left[0]])
    if np.any(slopes <= 1e-12):
        return [(0, 0, width, height)]
    x, y = width - 1, height - 1
    at_ends = np.array([
        min(top[2], top[0] * x + top[2]),
        min(right[0] * x + right[2], right[0] * x + right[1] * y + right[2]),
        min(bottom[1] * y + bottom[2], bottom[0] * x + bottom[1] * y + bottom[2]),
        min(left[2], left[1] * y + left[2]),
    ])
    top, right, bottom, left = np.clip(
        np.ceil(-at_ends / slopes), 0, [height, width, height, width],
    ).astype(int)
    middle_end = max(top, height - bottom)
    right_start = max(left, width - right)
    return [(0, 0, width, top), (0, middle_end, width, height),
            (0, top, left, middle_end), (right_start, top, width, middle_end)]


def _owned(taps: np.ndarray, lines: np.ndarray, extent: np.ndarray) -> np.ndarray:
    return (
        ((taps >= 0) & (taps < extent)).all(axis=-1)
        & (taps @ lines[:, :2].T + lines[:, 2] >= -1e-5).all(axis=-1)
    )


def _limit_edge_samples(image, output, map_to_source, lines, strips) -> bool:
    extent = np.array(image.shape[:2][::-1])
    unresolved = False
    source = image[..., None] if image.ndim == 2 else image
    target = output[..., None] if output.ndim == 2 else output
    for left, top, right, bottom in strips:
        width = right - left
        size = width * (bottom - top)
        for start in range(0, size, 4096):
            indices = np.arange(start, min(size, start + 4096))
            x, y = left + indices % width, top + indices // width
            samples = np.column_stack((x, y)) @ map_to_source[:, :2].T + map_to_source[:, 2]
            samples = samples[:, :2] / samples[:, 2:]
            quantized = np.rint(samples * _PHASES).astype(np.int64)
            anchor, phase = np.divmod(quantized, _PHASES)
            taps = anchor[:, None, :] + _TAP_OFFSETS
            weights = np.where(_TAP_OFFSETS, phase[:, None, :], _PHASES - phase[:, None, :]).prod(axis=2)
            keep = _owned(taps, lines, extent)
            affected = np.any(~keep, axis=1)
            floor = np.floor(samples).astype(np.int64)
            rounded_up = (floor != anchor).any(axis=1) & ~affected
            affected[rounded_up] = ~_owned(floor[rounded_up, None] + _TAP_OFFSETS, lines, extent).all(axis=1)
            if not affected.any():
                continue
            taps = taps[affected]
            weights = (weights[affected] * keep[affected]).astype(np.float64)
            x, y = x[affected], y[affected]
            centers = samples[affected]
            empty = ~weights.any(axis=1)
            if empty.any():
                # The interpolation table can round a small owned contribution to zero.
                cells = floor[affected][empty]
                fractions = (centers[empty] - cells)[:, None]
                taps[empty] = cells[:, None] + _TAP_OFFSETS
                exact = np.where(_TAP_OFFSETS, fractions, 1 - fractions).prod(axis=2)
                weights[empty] = exact * _owned(taps[empty], lines, extent)
            totals = weights.sum(axis=1)
            supported = totals > 0
            owned_taps = np.clip(taps[supported], 0, extent - 1)
            values = source[owned_taps[..., 1], owned_taps[..., 0]]
            average = (values * weights[supported, :, None]).sum(axis=1) / totals[supported, None]
            if np.issubdtype(image.dtype, np.integer):
                average = np.floor(average + .5)
            target[y[supported], x[supported]] = average

            if not supported.all():
                # A subpixel corner may have no owned tap in its bilinear cell.
                # Use only the nearest owned sample within two source pixels.
                missing = centers[~supported]
                neighbors = np.rint(missing).astype(np.int64)[:, None, :] + _CANDIDATE_OFFSETS
                distances = ((neighbors - missing[:, None, :]) ** 2).sum(axis=2)
                distances[~_owned(neighbors, lines, extent)] = np.inf
                nearest = distances.argmin(axis=1)
                rows = np.arange(len(nearest))
                valid = distances[rows, nearest] <= 4.0
                coordinates = neighbors[rows[valid], nearest[valid]]
                target[y[~supported][valid], x[~supported][valid]] = source[coordinates[:, 1], coordinates[:, 0]]
                unresolved |= not valid.all()
    return not unresolved


def warp_owned_crop(image: np.ndarray, matrix: np.ndarray, size: tuple[int, int]) -> tuple[np.ndarray, bool]:
    """Preserve the warp and replace only samples whose taps cross the boundary.

    Unsupported samples leave the usual warp as a review preview. The flag
    tells the caller whether all edge samples have an owned source value.
    """
    output = cv2.warpPerspective(image, matrix, size)
    width, height = size
    inverse = np.linalg.inv(matrix)
    output_corners = np.float64([[0, 0, 1], [width - 1, 0, 1], [width - 1, height - 1, 1], [0, height - 1, 1]])
    corners = output_corners @ inverse.T
    if np.all(corners[:, 2] < 0):
        inverse, corners = -inverse, -corners
    if not np.isfinite(corners).all() or not np.all(corners[:, 2] > 0):
        return output, False
    vertices = corners[:, :2] / corners[:, 2:]
    edges = np.roll(vertices, -1, axis=0) - vertices
    lengths = np.linalg.norm(edges, axis=1)
    if not np.all(lengths > 1e-6):
        return output, False
    normals = np.column_stack((-edges[:, 1], edges[:, 0])) / lengths[:, None]
    if np.dot(normals[0], vertices.mean(axis=0) - vertices[0]) < 0:
        normals = -normals
    lines = np.column_stack((normals, -(normals * vertices).sum(axis=1)))
    # The safe inset covers the full square of possible taps and phase rounding.
    inset = lines.copy()
    inset[:, 2] -= np.abs(normals).sum(axis=1) * (1 + .5 / _PHASES)
    strips = _edge_strips(inset @ inverse, width, height)
    return output, _limit_edge_samples(image, output, inverse, lines, strips)
