from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class BorderEdgeAssessment:
    edge: str
    trim_pixels: int
    confidence: float
    is_physical_border: bool
    reason_codes: tuple[str, ...]
    measurements: dict[str, float]

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class BorderTrimResult:
    image: np.ndarray
    offsets: dict[str, int]
    confidence: float
    applied: bool
    reason_codes: tuple[str, ...]
    edges: dict[str, BorderEdgeAssessment]

    def metadata(self) -> dict:
        return {
            "method": "content_aware_independent_edges_v1",
            "offsets": dict(self.offsets),
            "confidence": self.confidence,
            "applied": self.applied,
            "reason_codes": list(self.reason_codes),
            "edges": {name: edge.as_dict() for name, edge in self.edges.items()},
        }


def _oriented(image: np.ndarray, edge: str) -> np.ndarray:
    if edge == "top":
        return image
    if edge == "bottom":
        return image[::-1]
    if edge == "left":
        return np.transpose(image, (1, 0, 2))
    if edge == "right":
        return np.transpose(image[:, ::-1], (1, 0, 2))
    raise ValueError(f"Unsupported edge: {edge}")


def _protected_edge_veto(
    edge: str, trim_pixels: int, width: int, height: int, protected_regions: list[dict] | None
) -> bool:
    if trim_pixels <= 0 or not protected_regions:
        return False
    ratio = trim_pixels / (width if edge in {"left", "right"} else height)
    for region in protected_regions:
        box = region.get("box") or region.get("bounding_box") or {}
        try:
            if edge == "left" and float(box["x1"]) < ratio:
                return True
            if edge == "right" and float(box["x2"]) > 1.0 - ratio:
                return True
            if edge == "top" and float(box["y1"]) < ratio:
                return True
            if edge == "bottom" and float(box["y2"]) > 1.0 - ratio:
                return True
        except (KeyError, TypeError, ValueError):
            continue
    return False


def _assess_edge(
    image: np.ndarray,
    edge: str,
    *,
    max_crop_ratio: float,
    padding: int,
    protected_regions: list[dict] | None,
) -> BorderEdgeAssessment:
    view = _oriented(image, edge)
    depth, span = view.shape[:2]
    max_depth = min(max(0, int(round(depth * max_crop_ratio))), max(0, depth // 3))
    if max_depth < 3 or span < 16:
        return BorderEdgeAssessment(edge, 0, 0.0, False, ("BORDER_IMAGE_TOO_SMALL",), {})

    probe = max(2, min(8, max_depth // 3))
    lab = cv2.cvtColor(view, cv2.COLOR_BGR2LAB).astype(np.float32)
    gray = cv2.cvtColor(view, cv2.COLOR_BGR2GRAY).astype(np.float32)
    outer = lab[:probe]
    outer_color = np.median(outer.reshape(-1, 3), axis=0)
    outer_distance = np.linalg.norm(outer - outer_color, axis=2)
    dispersion = float(np.median(outer_distance) * 1.4826)

    along_gradient = np.abs(np.diff(gray, axis=1))
    texture = np.mean(along_gradient, axis=1) if along_gradient.size else np.zeros(depth, dtype=np.float32)
    outer_texture = float(np.median(texture[:probe]))
    cross_gradient = np.mean(np.abs(np.diff(gray, axis=0)), axis=1)
    baseline_cross = float(np.median(cross_gradient[: max(probe, min(max_depth, 10))])) if cross_gradient.size else 0.0
    line_distance = np.median(np.linalg.norm(lab - outer_color, axis=2), axis=1)

    candidate_depth = 0
    candidate_confidence = 0.0
    candidate_measurements: dict[str, float] = {}
    for index in range(max(2, probe), max_depth + 1):
        if index >= depth - 2:
            break
        window_end = min(depth, index + 4)
        color_change = float(np.median(line_distance[index:window_end]))
        texture_change = max(0.0, float(np.median(texture[index:window_end])) - outer_texture)
        transition = float(cross_gradient[index - 1]) if index - 1 < len(cross_gradient) else 0.0
        sustained_content = color_change >= 13.0 or texture_change >= 7.0
        supported_transition = transition >= max(5.0, baseline_cross * 1.8 + 2.0)
        if not sustained_content or not supported_transition:
            continue
        uniformity = max(0.0, min(1.0, 1.0 - dispersion / 18.0))
        color_score = max(0.0, min(1.0, color_change / 28.0))
        transition_score = max(0.0, min(1.0, transition / 24.0))
        confidence = uniformity * 0.36 + color_score * 0.34 + transition_score * 0.30
        if confidence > candidate_confidence:
            candidate_depth = index
            candidate_confidence = confidence
            candidate_measurements = {
                "outer_dispersion": dispersion,
                "outer_texture": outer_texture,
                "baseline_cross_gradient": baseline_cross,
                "boundary_color_change": color_change,
                "boundary_texture_change": texture_change,
                "boundary_gradient": transition,
            }
        # Prefer the first strong physical transition. Later scene edges must
        # not turn a quiet sky or dark foreground into a border.
        if confidence >= 0.74:
            break

    if candidate_depth <= 0 or candidate_confidence < 0.68:
        return BorderEdgeAssessment(
            edge,
            0,
            round(candidate_confidence, 4),
            False,
            ("BORDER_TRIM_UNCERTAIN",) if candidate_depth else ("NO_SUPPORTED_PHYSICAL_BORDER",),
            candidate_measurements
            or {
                "outer_dispersion": dispersion,
                "outer_texture": outer_texture,
                "baseline_cross_gradient": baseline_cross,
            },
        )

    trim_pixels = max(0, candidate_depth - max(0, padding))
    if _protected_edge_veto(edge, trim_pixels, image.shape[1], image.shape[0], protected_regions):
        return BorderEdgeAssessment(
            edge,
            0,
            round(candidate_confidence, 4),
            False,
            ("BORDER_PROTECTED_CONTENT_VETO",),
            candidate_measurements,
        )
    return BorderEdgeAssessment(
        edge,
        trim_pixels,
        round(candidate_confidence, 4),
        trim_pixels > 0,
        ("PHYSICAL_BORDER_TRANSITION_SUPPORTED",),
        candidate_measurements,
    )


def trim_physical_border(
    image: np.ndarray,
    *,
    max_crop_ratio: float = 0.12,
    padding: int = 2,
    protected_regions: list[dict] | None = None,
) -> BorderTrimResult:
    """Conservatively classify and trim four physical print-border edges.

    A quiet or uniformly coloured photographic region is insufficient. Each
    edge needs both a uniform outer band and a sustained, local transition into
    different content. Ambiguous and protected edges are preserved.
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Border classification expects a three-channel BGR image.")
    height, width = image.shape[:2]
    edges = {
        name: _assess_edge(
            image,
            name,
            max_crop_ratio=max_crop_ratio,
            padding=padding,
            protected_regions=protected_regions,
        )
        for name in ("top", "bottom", "left", "right")
    }
    offsets = {name: edge.trim_pixels for name, edge in edges.items()}
    top, bottom = offsets["top"], height - offsets["bottom"]
    left, right = offsets["left"], width - offsets["right"]
    if bottom - top < height * 0.62 or right - left < width * 0.62:
        offsets = {name: 0 for name in offsets}
        return BorderTrimResult(
            image.copy(),
            offsets,
            0.0,
            False,
            ("BORDER_TRIM_CONTENT_AREA_VETO",),
            edges,
        )
    applied = any(offsets.values())
    used_confidence = [edge.confidence for edge in edges.values() if edge.trim_pixels]
    confidence = float(min(used_confidence)) if used_confidence else 0.0
    reasons = tuple(dict.fromkeys(reason for edge in edges.values() for reason in edge.reason_codes))
    trimmed = image[top:bottom, left:right].copy() if applied else image.copy()
    return BorderTrimResult(trimmed, offsets, round(confidence, 4), applied, reasons, edges)


def remove_uniform_border(
    image: np.ndarray,
    *,
    max_crop_ratio: float = 0.12,
    min_content_ratio: float = 0.012,
    color_distance_threshold: float = 18.0,
    padding: int = 2,
) -> np.ndarray:
    """Backward-compatible wrapper around the content-aware classifier.

    The legacy parameters remain accepted for callers, but trimming decisions
    no longer rely on a global pixel-count threshold.
    """
    del min_content_ratio, color_distance_threshold
    return trim_physical_border(image, max_crop_ratio=max_crop_ratio, padding=padding).image
