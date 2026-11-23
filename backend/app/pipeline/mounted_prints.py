from __future__ import annotations

import cv2
import numpy as np


def _find_mounted_prints(image: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]] | None:
    """Find rectangular openings in a continuous, strongly colored album mount."""
    height, width = image.shape[:2]
    scale = min(1.0, 1500 / max(height, width))
    if min(height, width) * scale < 80:
        return None
    preview = cv2.resize(image, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA)
    height, width = preview.shape[:2]
    hue, saturation, value = cv2.split(cv2.cvtColor(preview, cv2.COLOR_BGR2HSV))
    frame = np.zeros((height, width), dtype=bool)
    left, right = round(width * 0.045), round(width * 0.955)
    top, bottom = round(height * 0.045), round(height * 0.955)
    frame[top:bottom, left:round(width * 0.10)] = True
    frame[top:bottom, round(width * 0.90):right] = True
    frame[top:round(height * 0.10), left:right] = True
    frame[round(height * 0.90):bottom, left:right] = True

    chromatic = (saturation >= 78) & (value >= 25)
    frame_pixels = np.count_nonzero(frame)
    if np.count_nonzero(frame & chromatic) < 0.56 * frame_pixels:
        return None
    hist = np.bincount(hue[frame & chromatic], minlength=180)
    center = int(np.argmax(sum(np.roll(hist, shift) for shift in range(-6, 7))))
    hue_distance = abs((hue.astype(np.int16) - center + 90) % 180 - 90)
    backing = chromatic & (hue_distance <= 14)
    mount_samples = backing & frame
    if np.count_nonzero(mount_samples) < 0.56 * frame_pixels:
        return None
    backing &= saturation >= max(78, 0.57 * np.median(saturation[mount_samples]))

    backing = cv2.morphologyEx(backing.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    count, members, stats, _ = cv2.connectedComponentsWithStats(backing)
    if count < 2:
        return None
    mount = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x, y, w, h, area = map(int, stats[mount])
    if w < width * 0.83 or h < height * 0.83 or area < width * height * 0.15:
        return None

    inside = members[y:y + h, x:x + w]
    foreground = (inside != mount).astype(np.uint8)
    foreground = cv2.morphologyEx(foreground, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(foreground, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    quads = []
    for contour in contours:
        contour_area = cv2.contourArea(contour)
        if not 0.022 <= contour_area / (w * h) <= 0.55:
            continue
        cx, cy, cw, ch = cv2.boundingRect(contour)
        if cx <= 2 or cy <= 2 or cx + cw >= w - 2 or cy + ch >= h - 2:
            continue
        hull = cv2.convexHull(contour)
        points = cv2.approxPolyDP(hull, 0.022 * cv2.arcLength(hull, True), True).reshape(-1, 2)
        if len(points) != 4:
            points = cv2.boxPoints(cv2.minAreaRect(hull))
        points = points.astype(np.float32)
        if contour_area < 0.83 * cv2.contourArea(points):
            continue
        lengths = np.linalg.norm(points - np.roll(points, 1, axis=0), axis=1)
        if min(lengths) <= 0 or max(lengths) > 2.5 * min(lengths):
            continue
        quads.append((points + np.array([x, y], dtype=np.float32)) / np.array([width - 1, height - 1], dtype=np.float32))

    # A colored photograph by itself is weak evidence of an album mount.
    if len(quads) < 2 and area < 0.91 * w * h:
        return None
    background = np.ones((height, width), dtype=np.uint8)
    background[y:y + h, x:x + w] = inside == mount
    return background, quads


def mounted_print_evidence(
    image: np.ndarray, polygons: list[list[dict[str, float]]]
) -> tuple[dict[int, str], list[np.ndarray]]:
    mount = _find_mounted_prints(image)
    if mount is None:
        return {}, []
    backing, quads = mount
    height, width = backing.shape
    scale = np.array([width - 1, height - 1], dtype=np.float32)
    prints = np.zeros_like(backing)
    for quad in quads:
        cv2.fillConvexPoly(prints, np.rint(quad * scale).astype(np.int32), 1)
    explained = cv2.bitwise_or(backing, prints)
    rejected = {}
    mask = np.zeros_like(backing)
    for index, polygon in enumerate(polygons):
        points = np.array([(point["x"], point["y"]) for point in polygon], dtype=np.float32)
        if len(points) < 3:
            continue
        mask.fill(0)
        cv2.fillPoly(mask, [np.rint(points * scale).astype(np.int32)], 1)
        area = cv2.countNonZero(mask)
        if not area:
            continue
        if cv2.countNonZero(cv2.bitwise_and(mask, backing)) >= 0.72 * area:
            rejected[index] = "album_mount_region"
        elif (
            cv2.countNonZero(cv2.bitwise_and(mask, prints)) >= 0.28 * area
            and cv2.countNonZero(cv2.bitwise_and(mask, explained)) >= 0.92 * area
        ):
            rejected[index] = "replaced_by_mount_outline"
    return rejected, quads
