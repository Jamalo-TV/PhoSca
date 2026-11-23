"""Estimate a print's aspect in the plane of a measured peer on the same page."""

from dataclasses import dataclass

import cv2
import numpy as np

from app.pipeline.print_edges import refine_print_boundary


PRINT_ASPECT_REFERENCE_REVIEW = "PRINT_ASPECT_REFERENCE_REVIEW"


@dataclass(frozen=True)
class MeasuredPrintReference:
    photo_id: str
    quad: np.ndarray
    aspect: float


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
        if not .1 <= ratio <= 10 or not residual <= .85:
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
