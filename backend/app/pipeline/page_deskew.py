"""Estimate an in-plane rotation from repeated album boundaries."""

from __future__ import annotations

import cv2
import numpy as np


def album_rotation(image: np.ndarray) -> float | None:
    scale = min(1., 1080 / max(image.shape[:2]))
    sample = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else image
    height, width = sample.shape[:2]
    short_side = min(width, height)
    if short_side < 48:
        return None
    gray = cv2.cvtColor(sample, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (3, 3), .65), 32, 85)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 720, max(12, round(.06 * short_side)),
        minLineLength=max(18, round(.09 * short_side)), maxLineGap=max(2, round(.014 * short_side)),
    )
    if lines is None:
        return None
    segments = lines.reshape(-1, 4).astype(float)
    direction = segments[:, 2:] - segments[:, :2]
    angles = (np.degrees(np.arctan2(direction[:, 1], direction[:, 0])) + 45) % 90 - 45
    length = np.linalg.norm(direction, axis=1)
    near_axis = np.abs(angles) <= 13
    candidates = np.arange(-13., 13.01, .1)
    votes = ((np.abs(candidates[:, None] - angles) <= .65) * length).sum(axis=1)
    supported = np.abs(angles - candidates[np.argmax(votes)]) <= .7
    if np.count_nonzero(supported) < 12 or length[supported].sum() < .63 * length[near_axis].sum():
        return None
    angle = float(np.average(angles[supported], weights=length[supported]))
    if not .85 <= abs(angle) <= 11.5:
        return None

    midpoint = (segments[:, :2] + segments[:, 2:]) * .5
    tangent = np.tan(np.deg2rad(angle))
    horizontal = np.abs(direction[:, 0]) >= np.abs(direction[:, 1])
    y = (midpoint[:, 1] - tangent * (midpoint[:, 0] - (width - 1) / 2)) / height
    x = (midpoint[:, 0] + tangent * (midpoint[:, 1] - (height - 1) / 2)) / width
    axis_angles, upright_perimeter = [], []
    for family, position, span in ((horizontal, y, width), (~horizontal, x, height)):
        agreeing = family & supported
        positions = np.sort(position[agreeing])
        if len(positions) < 3 or positions[-1] - positions[0] < .44:
            return None
        bands = [positions[0]]
        for position_px in positions[1:]:
            if position_px - bands[-1] >= .055:
                bands.append(position_px)
        if len(bands) < 3 or length[agreeing].sum() < .52 * length[family & near_axis].sum():
            return None
        axis_angles.append(float(np.average(angles[agreeing], weights=length[agreeing])))
        upright_perimeter.append(np.any(
            family & (np.abs(angles) <= .55) & ((position <= .15) | (position >= .85)) & (length >= .50 * span)
        ))
    # Oblique lines inside prints do not determine the page orientation.
    if abs(axis_angles[0] - axis_angles[1]) > .5 or all(upright_perimeter):
        return None
    return angle


def deskewed_view(image: np.ndarray, angle: float) -> tuple[np.ndarray, np.ndarray]:
    """Return a bounded inference view and its pixel-centre map back to the capture."""
    height, width = image.shape[:2]
    scale = min(1., 2048 / max(width, height))
    rotation = cv2.getRotationMatrix2D(((width - 1) / 2, (height - 1) / 2), angle, scale)
    bounds = np.float64([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])
    rotated = cv2.transform(bounds[None], rotation)[0]
    origin = np.floor(rotated.min(axis=0))
    extent = np.ceil(rotated.max(axis=0)) - origin + 1
    rotation[:, 2] -= origin
    edges = np.concatenate((image[0], image[-1], image[:, 0], image[:, -1]))
    padding = tuple(int(value) for value in np.percentile(edges, 60, axis=0))
    view = cv2.warpAffine(image, rotation, tuple(extent.astype(int)), borderValue=padding)
    return view, cv2.invertAffineTransform(rotation)
