"""Find visible prints and apertures in a bound spread with pale album stock."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from app.pipeline.geometry import order_quad_cyclic


ALBUM_WINDOW_SOURCE = "album_window"
ALBUM_WINDOW_OCCLUDED = "ALBUM_WINDOW_OCCLUDED"
ALBUM_WINDOW_STOCK_UNCERTAIN = "ALBUM_WINDOW_STOCK_UNCERTAIN"


@dataclass(frozen=True)
class AlbumLeaf:
    outline: np.ndarray
    print_outline: np.ndarray | None
    aperture_quad: np.ndarray | None
    decoration: np.ndarray | None


def _visible_print(image: np.ndarray, quad: np.ndarray) -> AlbumLeaf | None:
    edges = np.linalg.norm(np.roll(quad, -1, axis=0) - quad, axis=1)
    width, height = round(max(edges[::2])), round(max(edges[1::2]))
    if min(width, height) < 80:
        return None
    rectangle = np.float32([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])
    matrix = cv2.getPerspectiveTransform(quad, rectangle)
    inverse = np.linalg.inv(matrix)
    patch = cv2.warpPerspective(image, matrix, (width, height))
    lab = cv2.cvtColor(patch, cv2.COLOR_BGR2LAB).astype(np.float32)
    inset, band = round(.035 * min(width, height)), round(.10 * min(width, height))
    paper = np.concatenate([
        lab[inset:band, band:-band].reshape(-1, 3), lab[-band:-inset, band:-band].reshape(-1, 3),
        lab[band:-band, inset:band].reshape(-1, 3), lab[band:-band, -band:-inset].reshape(-1, 3),
    ])
    stock = np.median(paper, axis=0)
    variation = np.percentile(np.linalg.norm(paper - stock, axis=1), 90)
    if stock[0] < 150 or variation > 13:
        return None
    unlike = np.linalg.norm(lab - stock, axis=2) > max(12, variation + 7)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(unlike.astype(np.uint8))
    decoration = None
    prints = []
    unexplained = 0
    dimensions = np.float32(image.shape[1::-1]) - 1
    for index in range(1, count):
        x, y, w, h, area = stats[index]
        if area <= .002 * width * height:
            continue
        if x <= inset or y <= inset or x + w >= width - inset or y + h >= height - inset:
            unexplained += np.count_nonzero(labels[band:-band, band:-band] == index)
            continue
        occupancy = area / (w * h)
        member = np.uint8(labels[y:y + h, x:x + w] == index)
        contours, _ = cv2.findContours(member, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour = max(contours, key=cv2.contourArea)
        if (w >= .50 * width and h >= .50 * height and .018 <= occupancy <= .22
                and cv2.contourArea(contour) >= .55 * w * h):
            outline = cv2.approxPolyDP(contour, .8, True).reshape(-1, 2) + [x, y]
            decoration = cv2.perspectiveTransform(np.float32(outline)[None], inverse)[0] / dimensions
            continue
        if not (.055 <= area / (width * height) <= .57 and occupancy >= .80
                and w >= .24 * width and h >= .24 * height):
            unexplained += area
            continue
        outline = cv2.approxPolyDP(contour, .65, True).reshape(-1, 2) + [x, y]
        if len(outline) < 4:
            continue
        # A distinct pale print rim can be wider than the normal border search.
        depth = max(4, round(.065 * min(w, h)))
        rim = np.concatenate((lab[y + 3:y + depth, x + depth:x + w - depth].reshape(-1, 3),
                              lab[y + h - depth:y + h - 3, x + depth:x + w - depth].reshape(-1, 3),
                              lab[y + depth:y + h - depth, x + 3:x + depth].reshape(-1, 3),
                              lab[y + depth:y + h - depth, x + w - depth:x + w - 3].reshape(-1, 3)))
        rim_color = np.median(rim, axis=0)
        hull = cv2.convexHull(contour)
        distinct_rim = (
            rim_color[0] >= stock[0] + 9 and np.linalg.norm(rim_color - stock) >= 12
            and np.percentile(np.linalg.norm(rim - rim_color, axis=1), 90) <= 8
            and len(cv2.approxPolyDP(hull, .018 * cv2.arcLength(contour, True), True)) == 4
        )
        native = cv2.perspectiveTransform(np.float32(outline)[None], inverse)[0] / dimensions
        box = np.float32([[x, y], [x + w - 1, y], [x + w - 1, y + h - 1], [x, y + h - 1]])
        aperture = None if distinct_rim else cv2.perspectiveTransform(box[None], inverse)[0] / dimensions
        prints.append((native, aperture))
    if len(prints) > 1 or unexplained > .018 * width * height:
        return None
    if prints and (decoration is not None or prints[0][1] is None):
        return AlbumLeaf(quad / dimensions, *prints[0], decoration)
    return AlbumLeaf(quad / dimensions, None, None, decoration) if decoration is not None else None


def album_window_leaves(image: np.ndarray) -> list[AlbumLeaf]:
    scale = min(1.0, 1100 / max(image.shape[:2]))
    if min(image.shape[:2]) * scale < 160:
        return []
    preview = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else image
    height, width = preview.shape[:2]
    frame_y, frame_x = height // 22, width // 22
    outer = np.concatenate((preview[:frame_y:3, ::3].ravel(), preview[-frame_y::3, ::3].ravel(),
                            preview[::3, :frame_x:3].ravel(), preview[::3, -frame_x::3].ravel()))
    if np.percentile(outer, 70) > 105:
        return []
    lab = cv2.cvtColor(preview, cv2.COLOR_BGR2LAB).astype(np.float32)
    pale = (lab[..., 0] >= 155) & (lab[..., 2] >= 134)
    if np.mean(pale) < .21:
        return []
    paper = np.median(lab[pale], axis=0)
    stock = np.uint8(np.linalg.norm(lab - paper, axis=2) <= 15)
    contours, _ = cv2.findContours(stock, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    leaves = []
    for contour in contours:
        if not .13 <= cv2.contourArea(contour) / (width * height) <= .48:
            continue
        hull = cv2.convexHull(contour)
        quad = cv2.approxPolyDP(hull, .028 * cv2.arcLength(hull, True), True).reshape(-1, 2)
        if len(quad) == 4 and cv2.contourArea(contour) >= .88 * cv2.contourArea(quad):
            leaves.append(order_quad_cyclic(quad))
    if len(leaves) != 2:
        return []
    left, right = sorted(leaves, key=lambda quad: quad[:, 0].mean())
    first, second = left[[1, 2]], right[[0, 3]]
    gap = second - first
    if np.any(gap[:, 0] < -1) or np.any(gap[:, 0] > .038 * width) or np.any(abs(gap[:, 1]) > .065 * height):
        return []
    join = (first + second) / 2
    centers = np.rint(join[0] + np.linspace(.15, .85, 48)[:, None] * (join[1] - join[0])).astype(int)
    if np.mean(lab[centers[:, 1], centers[:, 0], 0] < paper[0] - 34) < .65:
        return []
    result = []
    for quad in (left, right):
        crop = _visible_print(preview, quad)
        if crop is None:
            return []
        result.append(crop)
    return result
