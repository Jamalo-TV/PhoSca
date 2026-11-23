"""Estimate print aspect ratios from corroborated geometry on the same page."""

from dataclasses import dataclass

import cv2
import numpy as np

from app.pipeline.print_edges import refine_print_boundary


PRINT_ASPECT_REFERENCE_REVIEW = "PRINT_ASPECT_REFERENCE_REVIEW"
_MAX_RECTANGLE_RESIDUAL_PX = .85


@dataclass(frozen=True)
class MeasuredPrintReference:
    photo_id: str
    quad: np.ndarray
    aspect: float


@dataclass(frozen=True)
class CalibratedPagePlane:
    image_to_plane: np.ndarray
    supporting_prints: int

    def estimate_aspect(self, points: np.ndarray) -> tuple[float, dict] | None:
        plane = _project(points, self.image_to_plane)
        if plane is None:
            return None
        residual = _rectangle_residual(points, plane, self.image_to_plane)
        aspect = _ratio(plane)
        if not residual <= _MAX_RECTANGLE_RESIDUAL_PX or not .1 <= aspect <= 10:
            return None
        return aspect, {"supporting_prints": self.supporting_prints, "source_corner_residual_px": residual}


def _project(points: np.ndarray, matrix: np.ndarray) -> np.ndarray | None:
    rays = np.column_stack((points, np.ones(len(points)))) @ matrix.T
    depth = rays[:, 2]
    if not np.isfinite(rays).all() or not (np.all(depth > 1e-6) or np.all(depth < -1e-6)):
        return None
    return rays[:, :2] / depth[:, None]


def _plane_map(quad: np.ndarray, aspect: float) -> np.ndarray:
    metric = np.float32([[0, 0], [aspect, 0], [aspect, 1], [0, 1]])
    return cv2.getPerspectiveTransform(np.float32(quad), metric)


def _ratio(points: np.ndarray) -> float:
    lengths = np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1)
    return float(lengths[::2].sum() / lengths[1::2].sum())


def _rectangle_residual(points: np.ndarray, plane: np.ndarray, matrix: np.ndarray) -> float:
    # Fit a rectangle in metric coordinates, then express the error in source pixels.
    along = plane[1] + plane[2] - plane[0] - plane[3]
    down = plane[2] + plane[3] - plane[1] - plane[0]
    along = along / np.linalg.norm(along)
    down = down / np.linalg.norm(down)
    along += [down[1], -down[0]]
    along /= np.linalg.norm(along)
    basis = np.array([along, [-along[1], along[0]]]).T
    local = plane @ basis
    left, right = local[[0, 3], 0].mean(), local[[1, 2], 0].mean()
    top, bottom = local[[0, 1], 1].mean(), local[[2, 3], 1].mean()
    rectangle = np.array([[left, top], [right, top], [right, bottom], [left, bottom]]) @ basis.T
    try:
        inverse = np.linalg.inv(matrix)
    except np.linalg.LinAlgError:
        return float("inf")
    back = _project(rectangle, inverse)
    return float(np.linalg.norm(back - points, axis=1).max()) if back is not None else float("inf")


def fit_calibrated_page_plane(
    quads: list[np.ndarray], camera: np.ndarray, image_shape: tuple[int, ...],
) -> CalibratedPagePlane | None:
    """Pool normals from source-verified prints with the same camera intrinsics."""
    minimum_prints = 6
    if len(quads) < minimum_prints:
        return None
    inverse_camera = np.linalg.inv(camera)
    unit_quad = np.float32([[0, 0], [1, 0], [1, 1], [0, 1]])
    normals = []
    for quad in quads:
        homography = cv2.getPerspectiveTransform(unit_quad, quad)
        axes = inverse_camera @ homography[:, :2]
        normal = np.cross(axes[:, 0], axes[:, 1])
        normal /= np.linalg.norm(normal)
        normals.append(-normal if normal[2] < 0 else normal)

    def plane_map(normal: np.ndarray) -> np.ndarray | None:
        norm = np.linalg.norm(normal)
        if not norm > 1e-8:
            return None
        normal = normal / norm
        axis = np.eye(3)[np.argmin(np.abs(normal))]
        across = np.cross(axis, normal)
        across /= np.linalg.norm(across)
        down = np.cross(normal, across)
        return np.array([across, down, normal]) @ inverse_camera

    def supporting_quads(matrix: np.ndarray) -> np.ndarray:
        return np.array([
            (plane := _project(quad, matrix)) is not None
            and _rectangle_residual(quad, plane, matrix) <= _MAX_RECTANGLE_RESIDUAL_PX
            for quad in quads
        ])

    normals = np.asarray(normals)
    initial = plane_map(np.median(normals, axis=0))
    if initial is None:
        return None
    support = supporting_quads(initial)
    required = max(minimum_prints, (2 * len(quads) + 2) // 3)
    if support.sum() < required:
        return None
    matrix = plane_map(normals[support].mean(axis=0))
    if matrix is None:
        return None
    support &= supporting_quads(matrix)
    centers = np.asarray(quads)[support].mean(axis=1)
    if len(centers) < required or np.any(np.ptp(centers, axis=0) < .25 * np.array(image_shape[1::-1])):
        return None
    return CalibratedPagePlane(matrix, len(centers))


def _corner_uncertainty(points: np.ndarray, reference: MeasuredPrintReference, nominal: float) -> float:
    # Sum the local sensitivity across independent half-pixel corner errors.
    total = 0.0
    for perturb_reference in (False, True):
        original = reference.quad if perturb_reference else points
        for vertex, axis in np.ndindex(original.shape):
            errors = []
            for sign in (-1, 1):
                shifted = original.copy()
                shifted[vertex, axis] += sign * .5
                quad = shifted if perturb_reference else reference.quad
                target = points if perturb_reference else shifted
                projected = _project(np.vstack((quad, target)), _plane_map(quad, reference.aspect))
                if projected is None:
                    return float("inf")
                errors.append(abs(_ratio(projected[4:]) / nominal - 1))
            total += max(errors)
    return total


def measured_page_aspect(
    image: np.ndarray, points: np.ndarray, references: list[MeasuredPrintReference],
) -> tuple[float | None, list[dict]]:
    estimates = []
    points = np.float32(points)
    for reference in references:
        quad = reference.quad
        if (
            quad.shape != (4, 2) or not np.isfinite(quad).all() or not .1 <= reference.aspect <= 10
            or np.any(quad < 0) or np.any(quad > np.array(image.shape[1::-1]) - 1)
            or not cv2.isContourConvex(np.float32(quad)) or cv2.contourArea(np.float32(quad), oriented=True) <= 0
            or np.linalg.norm(np.roll(quad, -1, axis=0) - quad, axis=1).min() < 60
            or cv2.intersectConvexConvex(points, np.float32(quad))[0] > 1
        ):
            continue
        edges = np.linalg.norm(np.roll(quad, -1, axis=0) - quad, axis=1)
        if np.max(np.abs(np.log(edges[:2] / edges[2:]))) < .02:
            continue
        refined, evidence, reasons = refine_print_boundary(image, quad, search_radius=4)
        if (refined is None or reasons or np.linalg.norm(refined - quad, axis=1).max() > 1.25
                or any(edge["support"] < .86 or (edge["corner_support"] or 0) < .75 for edge in evidence.values())):
            continue
        matrix = _plane_map(quad, reference.aspect)
        projected = _project(np.vstack((quad, points)), matrix)
        if projected is None:
            continue
        plane = projected[4:]
        residual = _rectangle_residual(points, plane, matrix)
        ratio = _ratio(plane)
        if not .1 <= ratio <= 10 or not residual <= _MAX_RECTANGLE_RESIDUAL_PX:
            continue
        variation = _corner_uncertainty(points, reference, ratio)
        if not variation <= .02:
            continue
        estimates.append({"photo_id": reference.photo_id, "aspect_ratio": ratio,
                          "source_corner_residual_px": residual, "relative_half_pixel_variation": variation})

    estimates.sort(key=lambda item: item["photo_id"])
    ratios = [item["aspect_ratio"] for item in estimates]
    if not ratios or max(ratios) / min(ratios) > 1.01:
        return None, estimates
    return float(np.median(ratios)), estimates
