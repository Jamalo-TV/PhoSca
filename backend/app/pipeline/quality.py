from dataclasses import dataclass


@dataclass(frozen=True)
class SegmentationQuality:
    aspect_ratio: float
    geometry_valid: bool
    review_reasons: list[str]


def evaluate_segmentation_geometry(
    width: float,
    height: float,
    *,
    min_aspect_ratio: float = 0.5,
    max_aspect_ratio: float = 3.0,
) -> SegmentationQuality:
    if width <= 0 or height <= 0:
        return SegmentationQuality(
            aspect_ratio=0.0,
            geometry_valid=False,
            review_reasons=["invalid_detection_dimensions"],
        )

    aspect_ratio = float(width / height)
    review_reasons: list[str] = []
    if aspect_ratio < min_aspect_ratio:
        review_reasons.append("aspect_ratio_too_tall")
    if aspect_ratio > max_aspect_ratio:
        review_reasons.append("aspect_ratio_too_wide")

    return SegmentationQuality(
        aspect_ratio=aspect_ratio,
        geometry_valid=not review_reasons,
        review_reasons=review_reasons,
    )


def segmentation_review_reasons(
    *,
    confidence: float,
    confidence_threshold: float,
    quality: SegmentationQuality,
) -> list[str]:
    reasons = list(quality.review_reasons)
    if confidence < confidence_threshold:
        reasons.append("segmentation_confidence_below_threshold")
    return reasons
