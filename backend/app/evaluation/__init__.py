"""Reproducible, model-independent evaluation primitives for PhoSca."""

from app.evaluation.cropping import (
    CroppingDatasetMetrics,
    PageCroppingMetrics,
    evaluate_cropping_dataset,
    evaluate_page_crops,
)

__all__ = [
    "CroppingDatasetMetrics",
    "PageCroppingMetrics",
    "evaluate_cropping_dataset",
    "evaluate_page_crops",
]
