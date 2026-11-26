"""Extend a held preview only between the old crop and observed paper."""

import cv2
import numpy as np

from app.pipeline.bowed_edge_review import MeasuredPrintPreview
from app.pipeline.crop_sampling import _outline_mask
from app.pipeline.dark_album_review import find_dark_mounted_print_review
from app.pipeline.print_edges import _sample_lab
from app.pipeline.protected_regions import region_box_in_pixels


def _outward_rail(quad: np.ndarray, traces: list[np.ndarray]) -> tuple[int, np.ndarray] | None:
    choices = []
    for index, (start, stop) in enumerate(zip(quad, np.roll(quad, -1, axis=0), strict=True)):
        direction = stop - start
        length = np.linalg.norm(direction)
        tangent = direction / length
        inward = np.float32([-tangent[1], tangent[0]])
        for span in traces:
            positions = (span - start) @ tangent / length
            depths = -(span - start) @ inward
            along = abs(positions[1] - positions[0])
            if (positions.min() >= .075 and positions.max() <= .925 and along >= .15
                    and depths.min() >= 32 and depths.max() <= .18 * length
                    and abs(depths[1] - depths[0]) <= .20 * along * length):
                choices.append((float(depths.mean() * along * length), index, span))
    choices.sort(key=lambda entry: entry[0], reverse=True)
    if not choices or (len(choices) > 1 and choices[0][0] < 1.35 * choices[1][0]):
        return None
    _, index, span = choices[0]
    return index, span


def _paper_depths(image: np.ndarray, stations: np.ndarray, inward: np.ndarray, reference: np.ndarray) -> np.ndarray:
    offsets = np.arange(-34, 35, dtype=np.float32)
    points = stations[:, None] + (reference[:, None, None] + offsets[None, :, None]) * inward
    if np.any((points < 0) | (points > np.float32(image.shape[1::-1]) - 1)):
        return np.zeros(len(stations), np.float32)
    colors = _sample_lab(image, points.reshape(-1, 2)).reshape(*points.shape[:-1], 3)
    peaks = np.arange(19, 50)
    outer = colors[:, peaks[:, None] - np.arange(6, 16)]
    backing = np.median(outer, axis=2)
    paper = np.median(colors[:, peaks[:, None] + np.arange(3, 8), 0], axis=2)
    jump = colors[:, peaks + 2, 0] - colors[:, peaks - 2, 0]
    lift = paper - backing[..., 0]
    supported = ((backing[..., 0] <= 54) & (outer[..., 0].max(axis=2) <= 75)
                 & (np.abs(backing[..., 1:] - 128).max(axis=2) <= 13) & (lift >= 28) & (jump >= 11))
    strength = np.where(supported, jump + .2 * lift, 0)
    best = strength.argmax(axis=1)
    maximum = strength.max(axis=1)
    competitors = np.where(np.abs(peaks[best, None] - peaks) > 4, strength, 0).max(axis=1)
    shift = offsets[peaks[best]]
    depths = reference + shift + 1.5
    measured = (maximum > 0) & (competitors < .8 * maximum) & (np.abs(shift) <= 13) & (depths < -12)
    close = np.abs(np.diff(depths)) <= 3
    measured &= np.r_[False, measured[:-1] & close] & np.r_[measured[1:] & close, False]
    return np.where(measured, depths, 0)


def _overlaps_protected_addition(
    added: np.ndarray, origin: np.ndarray, regions: list[dict], width: int, height: int,
) -> bool:
    for region in regions:
        box = region_box_in_pixels(region, width, height)
        if box is None:
            continue
        x1, y1, x2, y2 = box
        pad = float(region.get("margin_ratio", .08)) * max(x2 - x1, y2 - y1)
        left, top = np.maximum(np.floor([x1 - pad, y1 - pad]).astype(int) - origin, 0)
        right, bottom = np.minimum(np.ceil([x2 + pad, y2 + pad]).astype(int) + 1 - origin, added.shape[::-1])
        if right > left and bottom > top and added[top:bottom, left:right].any():
            return True
    return False


def _separate_mounts(
    image: np.ndarray, corners: np.ndarray, coverage: np.ndarray, added: np.ndarray, origin: np.ndarray,
) -> bool:
    """Use the coloured mount footprints only as vetoes for the comparison."""
    size = np.array(image.shape[1::-1])
    radius = max(15, round(.028 * max(size)))
    for x, y in np.rint(corners).astype(int):
        low = np.maximum([x - radius, y - radius], 0)
        high = np.minimum([x + radius + 1, y + radius + 1], size)
        patch = image[low[1]:high[1], low[0]:high[0]]
        lab = cv2.cvtColor(patch, cv2.COLOR_BGR2LAB)
        red = np.uint8((lab[..., 1] >= 145) & (lab[..., 0] >= 26))
        contours, hierarchy = cv2.findContours(red, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if hierarchy is None:
            return False
        ring = []
        for index, (_, _, child, parent) in enumerate(hierarchy[0]):
            contour = contours[index]
            if parent != -1 or child == -1 or cv2.pointPolygonTest(contour, tuple(np.float32([x, y] - low)), False) < 0:
                continue
            area = cv2.contourArea(contour)
            hole = max((cv2.contourArea(contours[j]) for j in np.flatnonzero(hierarchy[0, :, 3] == index)), default=0)
            if area >= .025 * patch.shape[0] * patch.shape[1] and .30 <= hole / area <= .76:
                ring.append(contour)
        if len(ring) != 1:
            return False
        holder = np.zeros(patch.shape[:2], np.uint8)
        cv2.fillPoly(holder, ring, 1)
        # The comparison cannot acquire even the one-pixel transition
        # around a holder. The old field retains its measured coverage.
        halo = cv2.dilate(holder, np.ones((3, 3), np.uint8))
        for forbidden, samples in ((holder, coverage), (halo, added)):
            yy, xx = np.nonzero(forbidden)
            xx = xx + low[0] - origin[0]
            yy = yy + low[1] - origin[1]
            inside = (xx >= 0) & (xx < samples.shape[1]) & (yy >= 0) & (yy < samples.shape[0])
            if samples[yy[inside], xx[inside]].any():
                return False
    return True


def suggest_mounted_strip_preview(
    image: np.ndarray, quad: np.ndarray, traces: list[np.ndarray], neighbors: list[np.ndarray], regions: list[dict],
) -> MeasuredPrintPreview | None:
    rail = _outward_rail(quad, traces)
    if rail is None:
        return None
    side, trace = rail
    start, stop = quad[[side, (side + 1) % 4]]
    direction = stop - start
    length = np.linalg.norm(direction)
    tangent = direction / length
    inward = np.float32([-tangent[1], tangent[0]])
    trace_range = np.sort((trace - start) @ tangent / length)
    begin, end = max(.09, trace_range[0] - .28), min(.91, trace_range[1] + .28)
    fractions = np.linspace(begin, end, max(16, round((end - begin) * length / 1.25)), dtype=np.float32)
    stations = start + fractions[:, None] * direction
    normal = np.float32([trace[0, 1] - trace[1, 1], trace[1, 0] - trace[0, 0]])
    if abs(normal @ inward) < 1e-6:
        return None
    reference = -((stations - trace[0]) @ normal) / (normal @ inward)
    depths = _paper_depths(image, stations, inward, reference)
    transitions = np.flatnonzero(np.diff(np.r_[False, depths < 0, False])).reshape(-1, 2)
    anchors = []
    for first, end in transitions:
        if end - first < 16:
            depths[first:end] = 0
            continue
        anchors.append(fractions[[first, end - 1]])
    if not any(np.minimum(stop, trace_range[1]) - np.maximum(start, trace_range[0]) >= .14 for start, stop in anchors):
        return None
    edge = cv2.approxPolyDP(stations + depths[:, None] * inward, .15, False)[:, 0]
    outline = np.vstack((quad[:side + 1], edge, quad[side + 1:]))
    hull = cv2.convexHull(outline)
    if any(cv2.intersectConvexConvex(hull, cv2.convexHull(peer))[0] > 1 for peer in neighbors):
        return None
    origin = np.floor(outline.min(axis=0)).astype(int)
    maximum = np.ceil(outline.max(axis=0)).astype(int) + 1
    shape = tuple((maximum - origin)[::-1])
    old = _outline_mask(np.float64(quad) - origin, shape)
    expanded = _outline_mask(np.float64(outline) - origin, shape)
    if np.any(old > expanded):
        return None
    added = (expanded > old)
    height, width = image.shape[:2]
    if added.sum() < .04 * old.sum() or _overlaps_protected_addition(added, origin, regions, width, height):
        return None
    scale = np.float32([width - 1, height - 1])
    evidence = find_dark_mounted_print_review(image, [peer / scale for peer in neighbors], partial_claim=quad / scale)
    if evidence is None or not _separate_mounts(image, evidence[0], expanded, added, origin):
        return None
    endpoints = np.flatnonzero(depths < 0)[[0, -1]]
    exterior = stations[endpoints] + (reference[endpoints, None] - 15) * inward
    band = np.vstack((stations[endpoints], exterior[::-1]))
    return MeasuredPrintPreview(side, [], outline, band)
