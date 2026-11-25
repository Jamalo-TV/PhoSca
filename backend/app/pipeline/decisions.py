"""Typed confidence components and stable human-review reason codes."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from enum import Enum
from math import exp, log


class ReviewReason(str, Enum):
    PHOTO_COUNT_UNCERTAIN = "PHOTO_COUNT_UNCERTAIN"
    OVERLAPPING_CANDIDATES = "OVERLAPPING_CANDIDATES"
    BOUNDARY_LOW_CONTRAST = "BOUNDARY_LOW_CONTRAST"
    POSSIBLE_CONTENT_CUTOFF = "POSSIBLE_CONTENT_CUTOFF"
    PAGE_BACKGROUND_INCLUDED = "PAGE_BACKGROUND_INCLUDED"
    QUADRILATERAL_UNSTABLE = "QUADRILATERAL_UNSTABLE"
    PARTIAL_PHOTO = "PARTIAL_PHOTO"
    ORIENTATION_UNCERTAIN = "ORIENTATION_UNCERTAIN"
    BORDER_TRIM_UNCERTAIN = "BORDER_TRIM_UNCERTAIN"
    ENHANCEMENT_ARTIFACT_RISK = "ENHANCEMENT_ARTIFACT_RISK"
    FACE_RESTORATION_RISK = "FACE_RESTORATION_RISK"
    TEXT_DISTORTION_RISK = "TEXT_DISTORTION_RISK"


def is_reviewed_non_print(rejection: dict | None, bounding_box: dict | None, mask: dict | None) -> bool:
    return bool(rejection and rejection == {"bounding_box": bounding_box, "segmentation_mask": mask})


def photo_is_reviewed_non_print(photo: object) -> bool:
    metadata = getattr(photo, "enhancement_applied", None) or {}
    return is_reviewed_non_print(
        metadata.get("review_rejected_detection"), getattr(photo, "bounding_box", None), getattr(photo, "segmentation_mask", None),
    )


@dataclass(frozen=True)
class ConfidenceComponents:
    detection_confidence: float | None = None
    boundary_confidence: float | None = None
    geometry_confidence: float | None = None
    perspective_confidence: float | None = None
    orientation_confidence: float | None = None
    border_trim_confidence: float | None = None
    enhancement_safety_confidence: float | None = None
    overall_release_confidence: float | None = None

    def as_dict(self) -> dict[str, float | None]:
        return asdict(self)


def _unit(value: object) -> float | None:
    if value is None:
        return None
    if not isinstance(value, (str, int, float)):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return max(0.0, min(1.0, number))


def aggregate_release_confidence(components: ConfidenceComponents) -> float | None:
    """Aggregate without allowing one high score to hide a weak component."""
    values = [
        _unit(getattr(components, field.name))
        for field in fields(components)
        if field.name != "overall_release_confidence"
    ]
    available = [value for value in values if value is not None]
    if not available:
        return None
    geometric_mean = exp(sum(log(max(value, 1e-6)) for value in available) / len(available))
    # A critical component remains dominant, while the small allowance avoids
    # pretending that every imperfect sub-score is a total failure.
    return round(min(geometric_mean, min(available) + 0.12), 4)


def confidence_components(existing: dict | None = None, **updates: object) -> ConfidenceComponents:
    source = dict(existing or {})
    source.update(updates)
    values = {
        field.name: _unit(source.get(field.name))
        for field in fields(ConfidenceComponents)
        if field.name != "overall_release_confidence"
    }
    preliminary = ConfidenceComponents(**values)
    return ConfidenceComponents(**values, overall_release_confidence=aggregate_release_confidence(preliminary))


def component_review_reasons(components: ConfidenceComponents) -> list[str]:
    reasons: list[str] = []
    thresholds = {
        "detection_confidence": (0.70, ReviewReason.PHOTO_COUNT_UNCERTAIN),
        "boundary_confidence": (0.66, ReviewReason.BOUNDARY_LOW_CONTRAST),
        "geometry_confidence": (0.72, ReviewReason.QUADRILATERAL_UNSTABLE),
        "perspective_confidence": (0.72, ReviewReason.QUADRILATERAL_UNSTABLE),
        "orientation_confidence": (0.60, ReviewReason.ORIENTATION_UNCERTAIN),
        "border_trim_confidence": (0.68, ReviewReason.BORDER_TRIM_UNCERTAIN),
        "enhancement_safety_confidence": (0.74, ReviewReason.ENHANCEMENT_ARTIFACT_RISK),
    }
    for name, (threshold, reason) in thresholds.items():
        value = getattr(components, name)
        if value is not None and value < threshold:
            reasons.append(reason.value)
    return list(dict.fromkeys(reasons))
