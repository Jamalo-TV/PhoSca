"""Estimate print aspect ratios from corroborated geometry on the same page."""

from dataclasses import dataclass

import cv2
import numpy as np

from app.pipeline.print_edges import refine_print_boundary


PRINT_ASPECT_REFERENCE_REVIEW = "PRINT_ASPECT_REFERENCE_REVIEW"
_MAX_RECTANGLE_RESIDUAL_PX = .85
_MIN_PAGE_PRINTS = 6
_UNIT_QUAD = np.float32([[0, 0], [1, 0], [1, 1], [0, 1]])


@dataclass(frozen=True)
class MeasuredPrintReference:
    photo_id: str
    quad: np.ndarray
    aspect: float


@dataclass(frozen=True)
class PrintPagePlane:
    image_to_plane: np.ndarray
    supporting_prints: int
    independent_fits: tuple[np.ndarray, ...] = ()

    def estimate_aspect(self, points: np.ndarray) -> tuple[float, dict] | None:
        aspects = []
        residuals = []
        for matrix in (self.image_to_plane, *self.independent_fits):
            plane = _project(points, matrix)
            if plane is None:
                return None
            residual = _rectangle_residual(points, plane, matrix)
            aspect = _ratio(plane)
            if not residual <= _MAX_RECTANGLE_RESIDUAL_PX or not .1 <= aspect <= 10:
                return None
            aspects.append(aspect)
            residuals.append(residual)
        evidence = {"supporting_prints": self.supporting_prints, "source_corner_residual_px": residuals[0]}
        if self.independent_fits:
            difference = max(aspects) / min(aspects) - 1
            if difference > .01:
                return None
            evidence["independent_aspect_difference"] = difference
        return aspects[0], evidence


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


def _supporting_quads(quads: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    return np.array([
        (plane := _project(quad, matrix)) is not None
        and _rectangle_residual(quad, plane, matrix) <= _MAX_RECTANGLE_RESIDUAL_PX
        for quad in quads
    ])


def _distributed_support(quads: np.ndarray, required: int, image_shape: tuple[int, ...]) -> bool:
    return len(quads) >= required and bool(np.all(
        np.ptp(quads.mean(axis=1), axis=0) >= .25 * np.array(image_shape[1::-1])
    ))


def fit_calibrated_page_plane(
    quads: list[np.ndarray], camera: np.ndarray, image_shape: tuple[int, ...],
) -> PrintPagePlane | None:
    """Pool normals from source-verified prints with the same camera intrinsics."""
    if len(quads) < _MIN_PAGE_PRINTS:
        return None
    inverse_camera = np.linalg.inv(camera)
    normals = []
    for quad in quads:
        homography = cv2.getPerspectiveTransform(_UNIT_QUAD, quad)
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

    normals = np.asarray(normals)
    candidates = np.asarray(quads)
    initial = plane_map(np.median(normals, axis=0))
    if initial is None:
        return None
    support = _supporting_quads(candidates, initial)
    required = max(_MIN_PAGE_PRINTS, (2 * len(quads) + 2) // 3)
    if support.sum() < required:
        return None
    matrix = plane_map(normals[support].mean(axis=0))
    if matrix is None:
        return None
    support &= _supporting_quads(candidates, matrix)
    if not _distributed_support(candidates[support], required, image_shape):
        return None
    return PrintPagePlane(matrix, int(support.sum()))


def _fit_uncalibrated_metric(quads: np.ndarray, image_shape: tuple[int, ...]) -> np.ndarray | None:
    width, height = image_shape[1::-1]
    scale = max(width, height)
    center = np.array([width - 1, height - 1]) / 2
    affine = np.array([[1 / scale, 0, -center[0] / scale], [0, 1 / scale, -center[1] / scale], [0, 0, 1.]])
    # The first two columns of each print homography lie on the page's vanishing line.
    axes = np.array([
        cv2.getPerspectiveTransform(_UNIT_QUAD, quad)[:, :2].T
        for quad in np.float32((quads - center) / scale)
    ])
    lengths = np.linalg.norm(axes[:, :, :2], axis=2)
    weights = (lengths.min(axis=1) / lengths.max()) ** 2
    directions = (axes / lengths[:, :, None] * weights[:, None, None]).reshape(-1, 3)
    horizon, _, rank, _ = np.linalg.lstsq(directions[:, :2], -directions[:, 2], rcond=None)
    if rank < 2:
        return None
    affine[2] = np.r_[horizon, 1.] @ affine
    projected = _project(quads.reshape(-1, 2), affine)
    if projected is None:
        return None
    plane = projected.reshape(-1, 4, 2)
    along = plane[:, 1] + plane[:, 2] - plane[:, 0] - plane[:, 3]
    down = plane[:, 2] + plane[:, 3] - plane[:, 0] - plane[:, 1]
    along /= np.linalg.norm(along, axis=1, keepdims=True)
    down /= np.linalg.norm(down, axis=1, keepdims=True)
    # Fit a plane metric of trace two from independent right-angle constraints.
    products = along * down
    orthogonal = np.column_stack((products[:, 0] - products[:, 1], (along * down[:, ::-1]).sum(axis=1)))
    (a, b), _, rank, singular = np.linalg.lstsq(orthogonal * weights[:, None], -products.sum(axis=1) * weights, rcond=None)
    if rank < 2 or singular[-1] < .25 * singular[0] or np.hypot(a, b) >= 1 - 1e-6:
        return None
    metric = np.linalg.cholesky([[1 + a, b], [b, 1 - a]]).T
    affine[:2] = metric @ affine[:2]
    return affine


def fit_uncalibrated_page_plane(quads: list[np.ndarray], image_shape: tuple[int, ...]) -> PrintPagePlane | None:
    if len(quads) < _MIN_PAGE_PRINTS:
        return None
    candidates = np.asarray(quads)
    required = max(_MIN_PAGE_PRINTS, (2 * len(quads) + 2) // 3)
    matrix = _fit_uncalibrated_metric(candidates, image_shape)
    if matrix is None:
        return None
    supported = candidates[_supporting_quads(candidates, matrix)]
    if not _distributed_support(supported, required, image_shape):
        return None
    if len(supported) < len(candidates):
        matrix = _fit_uncalibrated_metric(supported, image_shape)
        if matrix is None or not _supporting_quads(supported, matrix).all():
            return None
    directions = supported[:, 1] - supported[:, 0]
    angle_order = np.argsort(np.arctan2(directions[:, 1], directions[:, 0]) % (np.pi / 2), kind="stable")
    # Each disjoint fit must explain a diverse set of print orientations.
    first = _fit_uncalibrated_metric(supported[angle_order[::2]], image_shape)
    second = _fit_uncalibrated_metric(supported[angle_order[1::2]], image_shape)
    if first is None or second is None:
        return None
    return PrintPagePlane(matrix, len(supported), (first, second))


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
