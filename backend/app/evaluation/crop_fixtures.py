"""Reproducible planar and obstructed photo-album captures with exact labels."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from app.evaluation.cropping import Polygon


@dataclass(frozen=True)
class CropReference:
    mask: np.ndarray
    corners: np.ndarray
    shape: tuple[int, int]
    source_landmarks: np.ndarray
    landmarks: np.ndarray
    recoverable: bool
    image: np.ndarray


@dataclass(frozen=True)
class PrintFixture:
    visible_polygon: Polygon
    reference_modes: dict[str, CropReference]


@dataclass(frozen=True)
class PageFixture:
    name: str
    difficulty: str
    image: np.ndarray
    camera_matrix: np.ndarray
    prints: list[PrintFixture]


_PAGE_SIZE = (840, 640)
_CAMERA = np.array([[930, 0, 419.5], [0, 930, 319.5], [0, 0, 1]], dtype=np.float64)
_LANDMARKS = np.array([[.15, .12], [.73, .19], [.46, .53], [.23, .81], [.83, .74]], dtype=np.float32)
_CASES = {
    "borderless": "clean",
    "white_margin": "clean",
    "portrait": "clean",
    "white_on_white": "hard",
    "white_motif": "hard",
    "cast_shadow": "hard",
    "glare": "hard",
    "camera_tilt": "hard",
    "touching": "hard",
    "overlap": "hard",
    "touching_same_color": "review",
    "photo_corners": "review",
    "clipped": "review",
    "curl": "review",
    "negative": "negative",
    "paper_corner_mount": "hard",
    "dense_gutters": "hard",
}


def _rectangle(x1: float, y1: float, x2: float, y2: float) -> np.ndarray:
    return np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)


def _photograph(height: int, width: int, rng: np.random.Generator, *, white: bool) -> np.ndarray:
    # The asymmetric landscape also makes flips apparent on the contact sheet.
    y, x = np.mgrid[:height, :width].astype(np.float32)
    x /= width
    y /= height
    texture = cv2.resize(rng.normal(0, 9, (28, 40)).astype(np.float32), (width, height))
    horizon = .48 + .09 * np.sin(x * 11) + .045 * np.cos(x * 37)
    sky = np.stack([210 - 54*y, 180 - 61*y, 140 - 48*y], axis=-1)
    ground = np.stack([75 + 31*x, 104 + 29*x, 137 - 42*y], axis=-1)
    photo = np.where((y < horizon)[..., None], sky, ground) + texture[..., None]
    if white:
        photo[:, : width // 3] = 246 + texture[:, : width // 3, None] / 15
    return np.clip(photo, 0, 255).astype(np.uint8)


def _projected_corners(
    width: int, height: int, center: tuple[float, float], angles: tuple[float, float, float],
) -> np.ndarray:
    xyz = np.c_[_rectangle(0, 0, width - 1, height - 1), np.zeros(4)]
    xyz[:, :2] -= [(width - 1) / 2, (height - 1) / 2]
    distance = _CAMERA[0, 0]
    translation = np.array([center[0] - _CAMERA[0, 2], center[1] - _CAMERA[1, 2], distance])
    projected, _ = cv2.projectPoints(xyz, np.deg2rad(angles), translation, _CAMERA, None)
    return projected.reshape(4, 2).astype(np.float32)


def _warp(image: np.ndarray, matrix: np.ndarray, interpolation: int) -> np.ndarray:
    return cv2.warpPerspective(image, matrix, _PAGE_SIZE, flags=interpolation)


def _curl(image: np.ndarray, displacement: np.ndarray, interpolation: int) -> np.ndarray:
    yy, xx = np.indices(image.shape[:2], dtype=np.float32)
    return cv2.remap(image, xx, yy - displacement, interpolation)


def _layer(
    case: str,
    rng: np.random.Generator,
    *,
    width: int,
    height: int,
    center: tuple[float, float],
    angles: tuple[float, float, float],
    margin: int,
    paper_tone: int | None = None,
) -> tuple[np.ndarray, dict[str, CropReference]]:
    tone = paper_tone if paper_tone is not None else (241 if case == "white_on_white" else 232)
    print_image = np.full((height, width, 3), tone, dtype=np.uint8)
    picture = _photograph(height - 2 * margin, width - 2 * margin, rng, white=case == "white_motif")
    print_image[margin:height - margin, margin:width - margin] = picture
    corners = _projected_corners(width, height, center, angles)
    matrix = cv2.getPerspectiveTransform(_rectangle(0, 0, width - 1, height - 1), corners)
    rendered = cv2.warpPerspective(print_image, matrix, _PAGE_SIZE, borderMode=cv2.BORDER_REPLICATE)
    displacement = None
    if case == "curl":
        page_x = np.arange(_PAGE_SIZE[0], dtype=np.float32)
        displacement = 15 * np.sin(np.pi * np.clip((page_x - corners[0, 0]) / width, 0.0, 1.0))
        rendered = _curl(rendered, displacement, cv2.INTER_LINEAR)

    references = {}
    for mode, inset in (("full_print", 0), ("image_only", margin)):
        mask = np.zeros((height, width), dtype=np.uint8)
        mask[inset:height - inset, inset:width - inset] = 1
        source_mask = _warp(mask, matrix, cv2.INTER_NEAREST)
        source_corners = _rectangle(inset, inset, width - 1 - inset, height - 1 - inset)
        shape = (height - 2 * inset, width - 2 * inset)
        target_landmarks = _LANDMARKS * [shape[1] - 1, shape[0] - 1]
        source_landmarks = cv2.perspectiveTransform(
            (target_landmarks + inset).reshape(1, -1, 2), matrix,
        )[0]
        source_corners = cv2.perspectiveTransform(source_corners.reshape(1, 4, 2), matrix)[0]
        if displacement is not None:
            source_mask = _curl(source_mask, displacement, cv2.INTER_NEAREST)
            for points in (source_corners, source_landmarks):
                points[:, 1] += np.interp(points[:, 0], np.arange(len(displacement)), displacement)
        references[mode] = CropReference(
            source_mask, source_corners, shape, source_landmarks, target_landmarks, True,
            print_image[inset:height - inset, inset:width - inset].copy(),
        )
    return rendered, references


def _page(case: str, rng: np.random.Generator) -> PageFixture:
    width, height = _PAGE_SIZE
    paper_color = 249 if case == "white_on_white" else 146
    grain = cv2.resize(rng.normal(0, .75, (15, 20)).astype(np.float32), _PAGE_SIZE)
    page = np.clip(np.repeat((paper_color + grain)[..., None], 3, axis=-1), 0, 255).astype(np.uint8)
    owner = np.zeros((height, width), dtype=np.uint8)
    layers = []
    if case == "dense_gutters":
        for row in range(5):
            for column in range(6):
                centers = np.array([80 + 135 * column, 125 + 92 * row]) + rng.uniform(-1.5, 1.5, 2)
                angles = (2.0, -3.0, -1.3 + rng.uniform(-0.65, 0.65))
                layers.append(_layer(
                    case, rng, width=124, height=84, center=tuple(centers), angles=angles,
                    margin=0 if (row + column) % 3 == 0 else 5, paper_tone=227 + (column % 3) * 7,
                ))
    elif case != "negative":
        portrait = case == "portrait"
        width_px, height_px = (233, 330) if portrait else (332, 228)
        center = (420.2, 312.7)
        angles = (0.0, 0.0, 0.0)
        if case in {"overlap", "touching", "touching_same_color"}:
            center = (250.5, 312.5)
        elif case == "clipped":
            center = (129.5, 300.5)
        elif case == "camera_tilt":
            angles = tuple(np.array([-22, 32, 9]) + rng.uniform(-2.0, 2.0, 3))
        elif case in {"cast_shadow", "glare"}:
            angles = (5.0, -10.0, -7.0)
        center = tuple(np.asarray(center) + rng.uniform(-.75, .75, 2))
        margin = {"borderless": 0, "paper_corner_mount": 18}.get(case, 12)
        layers.append(_layer(case, rng, width=width_px, height=height_px, center=center, angles=angles, margin=margin))
        if case in {"touching", "touching_same_color", "overlap"}:
            separation = 268.0 if case == "overlap" else 332.0
            layers.append(_layer(
                case, rng, width=282, height=200, center=(center[0] + separation - 25, center[1] + 16),
                angles=(0.0, 0.0, 7.0 if case == "overlap" else 0.0), margin=9,
                paper_tone=204 if case == "touching" else None,
            ))
    for index, (rendered, references) in enumerate(layers, start=1):
        print_mask = references["full_print"].mask
        if case == "cast_shadow" or (case == "dense_gutters" and index % 3 == 2):
            offset_x, offset_y = (10, 12) if case == "cast_shadow" else (2, 3)
            shift = np.float32([[1, 0, offset_x], [0, 1, offset_y]])
            shadow = cv2.warpAffine(print_mask.astype(np.float32), shift, _PAGE_SIZE)
            shadow = cv2.GaussianBlur(shadow, (13, 13), 2.7 if case == "cast_shadow" else 1.2)
            page = np.clip(page * (1 - .44 * shadow[..., None]), 0, 255).astype(np.uint8)
        visible = print_mask != 0
        owner[visible] = index
        page[visible] = rendered[visible]
    if case == "glare":
        reflection = np.zeros((height, width), dtype=np.float32)
        cv2.ellipse(reflection, (480, 187), (70, 21), -11, 0, 360, 1, -1)
        reflection = cv2.GaussianBlur(reflection, (39, 39), 10)
        page = np.rint(page * (1 - reflection[..., None]) + 253 * reflection[..., None]).astype(np.uint8)
    if case in {"photo_corners", "paper_corner_mount"}:
        clip_length = 29 if case == "paper_corner_mount" else 38
        clips = np.zeros((height, width), dtype=np.uint8)
        for i, (x, y) in enumerate(layers[0][1]["full_print"].corners):
            sx, sy = (1 if i in {0, 3} else -1), (1 if i in {0, 1} else -1)
            triangle = np.rint([[x - 3*sx, y - 3*sy], [x + clip_length*sx, y - 3*sy], [x - 3*sx, y + clip_length*sy]])
            cv2.fillConvexPoly(clips, triangle.astype(np.int32), 1)
        owner[clips != 0] = 0
        page[clips != 0] = (45, 53, 64)

    prints = []
    for index, (_, references) in enumerate(layers, start=1):
        visible = (owner == index).astype(np.uint8)
        contours, _ = cv2.findContours(visible, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        polygon = max(contours, key=cv2.contourArea).reshape(-1, 2) / [width - 1, height - 1]
        modes = {}
        for mode, reference in references.items():
            obstruction = np.any(reference.mask & (1 - visible))
            in_frame = ((reference.corners >= [0, 0]) & (reference.corners <= [width - 1, height - 1])).all()
            modes[mode] = CropReference(
                reference.mask & visible,
                reference.corners,
                reference.shape,
                reference.source_landmarks,
                reference.landmarks,
                bool(not obstruction and in_frame and case not in {"curl", "touching_same_color"}),
                reference.image,
            )
        prints.append(PrintFixture([tuple(point) for point in polygon], modes))
    return PageFixture(case, _CASES[case], page, _CAMERA.copy(), prints)


def synthetic_crop_pages(split: str) -> list[PageFixture]:
    """Separate development texture/jitter from locked reference seeds."""
    seeds = {"development": 30291, "locked": 85997}
    rng = np.random.default_rng(seeds[split])
    # The dense page is available for tuning. Preserve the locked benchmark as a holdout.
    return [_page(case, rng) for case in _CASES if split == "development" or case != "dense_gutters"]


def geometry_seed(print_fixture: PrintFixture, image_shape: tuple[int, ...]) -> tuple[dict, dict]:
    """A slightly imprecise proposal for isolating refinement from detection."""
    height, width = image_shape[:2]
    full = print_fixture.reference_modes["full_print"]
    jitter = np.array([[1.3, -2.0], [-2.7, 1.1], [2.9, 1.7], [-1.1, -2.4]])
    points = np.clip((full.corners + jitter) / [width - 1, height - 1], 0, 1)
    vertices = [{"x": float(x), "y": float(y)} for x, y in points]
    visible = [{"x": float(x), "y": float(y)} for x, y in print_fixture.visible_polygon]
    x1, y1 = points.min(axis=0)
    x2, y2 = points.max(axis=0)
    box = {"x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2)}
    return box, {"polygon": vertices, "visible_polygon": visible, "source": "yolo_seg_mask"}
