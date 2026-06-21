from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from app.pipeline.image_ops import calculate_blur_score


VARIANT_ORDER = ("original", "enhanced", "premium")


def _shape(image: np.ndarray) -> dict[str, int]:
    height, width = image.shape[:2]
    return {"width": int(width), "height": int(height)}


def _source_delta(source: np.ndarray, image: np.ndarray) -> dict[str, float]:
    comparable = image
    if comparable.shape[:2] != source.shape[:2]:
        comparable = cv2.resize(comparable, (source.shape[1], source.shape[0]), interpolation=cv2.INTER_AREA)
    source_f = source.astype(np.float32)
    image_f = comparable.astype(np.float32)
    mean_abs_delta = float(np.mean(np.abs(source_f - image_f)))
    source_mean = np.mean(source_f, axis=(0, 1))
    image_mean = np.mean(image_f, axis=(0, 1))
    color_delta = float(np.linalg.norm(source_mean - image_mean))
    return {
        "mean_abs_delta": mean_abs_delta,
        "source_similarity": max(0.0, min(1.0, 1.0 - (mean_abs_delta / 255.0))),
        "color_delta": color_delta,
    }


def _metrics(source: np.ndarray | None, image: np.ndarray) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "shape": _shape(image),
        "blur_score": calculate_blur_score(image),
    }
    if source is not None:
        metrics.update(_source_delta(source, image))
    return metrics


def _warnings(name: str, metadata: dict, metrics: dict) -> list[str]:
    warnings: list[str] = []
    if metadata.get("method") == "failed" or metadata.get("error"):
        warnings.append("enhancement_failed_fell_back_to_source")
    if metrics.get("source_similarity", 1.0) < 0.62:
        warnings.append("large_source_delta_review_recommended")
    if name == "premium" and metadata.get("diffusion_used"):
        warnings.append("creative_diffusion_variant_review_required")
    return warnings


def variant_record(
    name: str,
    path: Path,
    image: np.ndarray,
    metadata: dict | None = None,
    source_image: np.ndarray | None = None,
    selected: bool = False,
) -> dict[str, Any]:
    metadata = dict(metadata or {})
    metrics = _metrics(source_image, image)
    return {
        "name": name,
        "path": str(path),
        "selected": selected,
        "model_chain": metadata.get("models", []),
        "params": metadata.get("config", {}),
        "metrics": metrics,
        "warnings": _warnings(name, metadata, metrics),
        "metadata": metadata,
    }


def build_variant_bundle(
    original_path: Path,
    original: np.ndarray,
    enhanced_path: Path,
    enhanced: np.ndarray,
    enhanced_metadata: dict,
    premium_path: Path,
    premium: np.ndarray,
    premium_metadata: dict,
    selected_variant: str = "enhanced",
) -> dict[str, Any]:
    variants = {
        "original": variant_record(
            "original",
            original_path,
            original,
            metadata={"method": "source"},
            selected=selected_variant == "original",
        ),
        "enhanced": variant_record(
            "enhanced",
            enhanced_path,
            enhanced,
            metadata=enhanced_metadata,
            source_image=original,
            selected=selected_variant == "enhanced",
        ),
        "premium": variant_record(
            "premium",
            premium_path,
            premium,
            metadata=premium_metadata,
            source_image=original,
            selected=selected_variant == "premium",
        ),
    }
    return {
        "variant_schema_version": 1,
        "selected_variant": selected_variant,
        "variants": variants,
    }


def variants_from_metadata(metadata: dict | None) -> dict[str, dict]:
    if not metadata:
        return {}
    variants = metadata.get("variants")
    return variants if isinstance(variants, dict) else {}


def variant_path_from_photo(photo: Any, variant: str) -> str | None:
    if variant == "original":
        return photo.original_storage_path

    variants = variants_from_metadata(getattr(photo, "enhancement_applied", None))
    record = variants.get(variant)
    if isinstance(record, dict) and record.get("path"):
        return str(record["path"])

    if variant == "enhanced":
        return photo.storage_path
    if variant == "premium" and photo.storage_path:
        return str(photo.storage_path).replace("_enhanced.jpg", "_premium.jpg")
    return None


def variant_urls(photo: Any) -> dict[str, str | None]:
    names = list(VARIANT_ORDER)
    for name in variants_from_metadata(getattr(photo, "enhancement_applied", None)):
        if name not in names:
            names.append(name)
    return {
        name: f"/api/v1/photos/{photo.id}/image?variant={name}" if variant_path_from_photo(photo, name) else None
        for name in names
    }
