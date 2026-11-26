"""Offer an editable paper outline when two neighboring stocks are distinguishable."""

from dataclasses import dataclass

import cv2
import numpy as np

from app.pipeline.print_edges import _edge_band, _is_pale_stock, _replace_quad_side, _side_line, refine_print_boundary


ADJACENT_STOCK_REVIEW_REVISION = 2
ADJACENT_STOCK_REVIEW_SOURCE = "adjacent_paper_stock"
_SIDES = ("top", "right", "bottom", "left")
_OFFSETS = [-6, -4, 4, 6]


@dataclass(frozen=True)
class _StockFit:
    quad: np.ndarray
    stock: np.ndarray
    exterior: np.ndarray


def _strong_side(evidence: dict) -> bool:
    residual = evidence.get("maximum_residual_px")
    return (not evidence["ambiguous"] and evidence["support"] >= .90
            and (evidence["corner_support"] or 0) >= .80 and residual is not None and residual <= 1.1)


def _stock_fit(image: np.ndarray, seed: np.ndarray, quad: np.ndarray, evidence: dict) -> _StockFit | None:
    if any(side not in evidence for side in _SIDES):
        return None
    trusted = [index for index, name in enumerate(_SIDES) if _strong_side(evidence[name])]
    uncertain = [index for index, name in enumerate(_SIDES) if evidence[name]["ambiguous"]]
    if len(trusted) + len(uncertain) != 4 or not 1 <= len(uncertain) <= 2:
        return None
    bands = [_edge_band(image, quad[index], quad[(index + 1) % 4], _OFFSETS) for index in trusted]
    if any(band is None for band in bands):
        return None
    samples = np.concatenate(bands)
    stock = np.median(samples[:, 2:], axis=(0, 1))
    if not _is_pale_stock(stock):
        return None
    fitted = quad
    for index in uncertain:
        radius = evidence[_SIDES[index]]["search_radius_px"]
        line, observed = _side_line(image, seed[index], seed[(index + 1) % 4], radius, rim_color=stock)
        if line is None or not _strong_side(observed):
            return None
        fitted = _replace_quad_side(fitted, index, line, radius, image.shape)
        if fitted is None:
            return None
    # Loading a review suggestion should never narrow the previous outline
    # by a visible amount.
    if any(cv2.pointPolygonTest(fitted, tuple(point), True) < -.75 for point in quad):
        return None
    return _StockFit(fitted, stock, samples[:, :2])


def _corroborated_stock(
    image: np.ndarray, fit: _StockFit, neighbor: _StockFit, backing: np.ndarray,
) -> bool:
    def sample_support(band: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        outside, inside = band[:, :2], band[:, 2:]
        own = (np.linalg.norm(inside - fit.stock, axis=2) <= 4).all(axis=1)
        album = (np.linalg.norm(outside - backing, axis=2) <= 8).all(axis=1)
        adjacent = (np.linalg.norm(outside - neighbor.stock, axis=2) <= 8).all(axis=1)
        return own & (album | adjacent), own & adjacent

    adjacent_stock = False
    for start, stop in zip(fit.quad, np.roll(fit.quad, -1, axis=0), strict=True):
        band = _edge_band(image, start, stop, _OFFSETS)
        if band is None:
            return False
        supported, adjacent = sample_support(band)
        if supported.mean() < .90 or any(part.mean() < .72 for part in np.split(supported, 4)):
            return False
        fractions = np.float32([4, 6, 8, 10, 13, 16, 20]) / np.linalg.norm(stop - start)
        if fractions[-1] >= .5:
            return False
        for positions in (fractions, 1 - fractions):
            arm = _edge_band(image, start, stop, _OFFSETS, fractions=positions)
            if arm is None:
                return False
            corner, _ = sample_support(arm)
            if not corner[:2].all() or np.count_nonzero(corner) < 5:
                return False
        # A second palette has to occur where that print actually meets the
        # sampled rim, rather than in a detached pattern in the album.
        projected = start + np.linspace(.10, .90, len(band))[:, None] * (stop - start)
        if np.count_nonzero(adjacent) >= .20 * len(band):
            near = [cv2.pointPolygonTest(neighbor.quad, tuple(point), True) >= -9
                    for point in projected[adjacent]]
            adjacent_stock |= np.mean(near) >= .90
    return adjacent_stock


def suggest_adjacent_stock_outline(
    image: np.ndarray, seed: np.ndarray, quad: np.ndarray, evidence: dict, neighbor_seed: np.ndarray,
) -> np.ndarray | None:
    """Return a provisional correction. The stocks do not establish physical ownership."""
    own = _stock_fit(image, seed, quad, evidence)
    if own is None:
        return None
    neighbor_quad, neighbor_evidence, reasons = refine_print_boundary(image, neighbor_seed)
    if neighbor_quad is None or reasons != ["PRINT_BOUNDARY_AMBIGUOUS"]:
        return None
    neighbor = _stock_fit(image, neighbor_seed, neighbor_quad, neighbor_evidence)
    if neighbor is None or np.linalg.norm(own.stock - neighbor.stock) <= 16:
        return None
    if cv2.intersectConvexConvex(own.quad, neighbor.quad)[0] > .05:
        return None
    if cv2.contourArea(own.quad) < 1.005 * cv2.contourArea(quad):
        return None
    backing = np.median(np.concatenate((own.exterior, neighbor.exterior)), axis=(0, 1))
    if min(np.linalg.norm(backing - item.stock) for item in (own, neighbor)) <= 16:
        return None
    if _corroborated_stock(image, own, neighbor, backing) and _corroborated_stock(image, neighbor, own, backing):
        return own.quad
    return None
