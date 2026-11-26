from __future__ import annotations

from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import cv2
import numpy as np

from app.config import get_settings
from app.core.metrics import RELEASE_DECISIONS
from app.pipeline.camera_calibration import capture_calibration
from app.pipeline.crop_validation import crop_validation_evidence
from app.pipeline.decisions import photo_is_reviewed_non_print
from app.pipeline.edge_prints import capture_edge_requires_review
from app.pipeline.image_ops import calculate_blur_score
from app.pipeline.manual_paper_boundary import (
    manual_paper_download_needs_review, manual_scalloped_outline, paper_claims_need_overlap_check,
)
from app.pipeline.protected_regions import album_mark_matches_quad
from app.pipeline.scalloped_prints import SCALLOPED_PRINT_REVISION
from app.pipeline.quality_critic import evaluate_candidate
from app.pipeline.release import select_released_variant

FULL_PRINT_REVIEW_PREVIEW = "full_print_preview"
MOTIF_REVIEW_PREVIEW = "image_only_preview"
CROP_REVIEW_PREVIEWS = {
    FULL_PRINT_REVIEW_PREVIEW: "full_print",
    MOTIF_REVIEW_PREVIEW: "image_only",
}

VARIANT_ORDER = (
    "full_print",
    "image_only",
    "source_corrected",
    "source_corrected_untrimmed",
    "source_corrected_trimmed",
    "conservative_enhanced",
    "restoration_candidate",
    "face_restoration_candidate",
    "original",
    "enhanced",
    "premium",
)


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
    if name in {"restoration_candidate", "face_restoration_candidate"} or metadata.get("generative"):
        warnings.append("model_restoration_candidate_review_required")
    return warnings


def select_safe_variant(
    variants: dict[str, dict[str, Any]], *, photo_status: Any = None, metadata: dict | None = None
) -> str:
    """Return the only variant that is safe to expose by default.

    Derived images are opt-in until a later quality critic can explicitly mark
    them as auto-releasable.  This makes the original crop the stable fallback
    for model errors, incomplete stages, and unreviewed experimental output.
    """
    return select_released_variant(variants, photo_status=photo_status, metadata=metadata)[0]


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
    gates = evaluate_candidate(source_image, image, metadata)
    warnings = _warnings(name, metadata, metrics)
    warnings.extend(f"quality_gate_failed:{gate['name']}" for gate in gates if gate.get("passed") is False)
    return {
        "name": name,
        "kind": metadata.get("variant_kind", name),
        "path": str(path),
        "selected": selected,
        "model_chain": metadata.get("models", []),
        "params": metadata.get("config", {}),
        "metrics": metrics,
        "warnings": list(dict.fromkeys(warnings)),
        "gates": gates,
        "metadata": metadata,
    }


def build_variant_bundle(
    original_path: Path,
    original: np.ndarray,
    enhanced_path: Path | None = None,
    enhanced: np.ndarray | None = None,
    enhanced_metadata: dict | None = None,
    premium_path: Path | None = None,
    premium: np.ndarray | None = None,
    premium_metadata: dict | None = None,
    selected_variant: str | None = None,
    photo_status: Any = None,
    release_metadata: dict | None = None,
    *,
    canonical_kinds: bool = False,
    conservative_path: Path | None = None,
    conservative: np.ndarray | None = None,
    conservative_metadata: dict | None = None,
    restoration_path: Path | None = None,
    restoration: np.ndarray | None = None,
    restoration_metadata: dict | None = None,
) -> dict[str, Any]:
    source_name = "source_corrected" if canonical_kinds else "original"
    variants: dict[str, dict[str, Any]] = {
        source_name: variant_record(
            source_name,
            original_path,
            original,
            metadata={"method": "source", "variant_kind": "source_corrected"},
            selected=False,
        ),
    }
    resolved_conservative_path = conservative_path or (enhanced_path if canonical_kinds else None)
    resolved_conservative = conservative if conservative is not None else (enhanced if canonical_kinds else None)
    resolved_conservative_metadata = (
        conservative_metadata if conservative_metadata is not None else (enhanced_metadata if canonical_kinds else None)
    )
    if resolved_conservative_path is not None and resolved_conservative is not None:
        variants["conservative_enhanced"] = variant_record(
            "conservative_enhanced",
            resolved_conservative_path,
            resolved_conservative,
            metadata=resolved_conservative_metadata,
            source_image=original,
            selected=False,
        )
    elif enhanced_path is not None and enhanced is not None:
        variants["enhanced"] = variant_record(
            "enhanced",
            enhanced_path,
            enhanced,
            metadata=enhanced_metadata,
            source_image=original,
            selected=False,
        )
    if restoration_path is not None and restoration is not None:
        restoration_name = str((restoration_metadata or {}).get("variant_kind") or "restoration_candidate")
        if restoration_name not in {"restoration_candidate", "face_restoration_candidate"}:
            restoration_name = "restoration_candidate"
        variants[restoration_name] = variant_record(
            restoration_name,
            restoration_path,
            restoration,
            metadata=restoration_metadata,
            source_image=original,
            selected=False,
        )
    if premium_path is not None and premium is not None:
        variants["premium"] = variant_record(
            "premium",
            premium_path,
            premium,
            metadata=premium_metadata,
            source_image=original,
            selected=False,
        )
    # A caller hint may choose the faithful source, but derived candidates must
    # always pass the central release selector.
    source_hint = selected_variant in {"original", "source_corrected"}
    selected_variant, release_reasons = (
        (source_name, [])
        if source_hint
        else select_released_variant(
            variants,
            photo_status=photo_status,
            metadata=release_metadata,
        )
    )
    for name, variant in variants.items():
        variant["selected"] = name == selected_variant
    RELEASE_DECISIONS.labels(variant=selected_variant, blocked=str(bool(release_reasons)).lower()).inc()
    return {
        "variant_schema_version": 1,
        "selected_variant": selected_variant,
        "release": {"selected_variant": selected_variant, "reasons": release_reasons},
        "variants": variants,
    }


def variants_from_metadata(metadata: dict | None) -> dict[str, dict]:
    if not metadata:
        return {}
    variants = metadata.get("variants")
    return variants if isinstance(variants, dict) else {}


def variant_path_from_photo(photo: Any, variant: str) -> str | None:
    # An explicit source-corrected view remains available to review the rejection.
    if photo_is_reviewed_non_print(photo) and variant not in {"source_corrected", "source_corrected_untrimmed"}:
        return None
    metadata = getattr(photo, "enhancement_applied", None) or {}
    correction = metadata.get("correction") or {}
    modes = correction.get("crop_modes") or {}
    variants = variants_from_metadata(metadata)
    precise_crop = bool(modes or "image_only" in variants)
    mode = "image_only" if precise_crop and variant == "source_corrected_trimmed" else variant
    automatic = getattr(photo, "segmentation_confidence", None) is not None
    mask = getattr(photo, "segmentation_mask", None) or {}
    if variant in {"full_print", "image_only", "source_corrected_trimmed"}:
        if mask.get("projective_outline_review") is True and (
            mask.get("projective_outline_confirmed") is not True or (modes.get("full_print") or {}).get("status") != "ready"
        ):
            return None
        if capture_edge_requires_review(mask):
            return None
        if paper_claims_need_overlap_check(mask, metadata, automatic=automatic):
            return None
        if not automatic:
            quad = mask.get("warp_quad") or mask.get("polygon")
            mark = (metadata.get("review_provenance") or {}).get("reviewed_album_mark")
            if mark and (not album_mark_matches_quad(mark, quad)
                         or (correction.get("protected_regions") or {}).get("reviewed_album_mark_id") != mark["id"]):
                return None
            if manual_paper_download_needs_review(metadata, quad):
                return None
            if manual_scalloped_outline(metadata, quad) is not None and (
                mode != "image_only"
                or (modes.get(mode) or {}).get("scalloped_motif_ownership_revision") != SCALLOPED_PRINT_REVISION
            ):
                return None
    if mode in {"full_print", "image_only"} or mode in CROP_REVIEW_PREVIEWS:
        if mode in CROP_REVIEW_PREVIEWS:
            path = correction.get(f"{mode}_path")
            crop_mode = modes.get(CROP_REVIEW_PREVIEWS[mode]) or {}
            if (crop_mode.get("status") == "review_needed" and crop_mode.get("outline_preview")
                    and isinstance(path, str) and Path(path).is_file()):
                return path
            return None
        if (modes.get(mode) or {}).get("status") != "ready":
            return None
        if (correction.get("aspect_ratio_estimation") or {}).get("method") == "calibrated_camera":
            if automatic:
                return None
            evidence = correction.get("camera_calibration") or {}
            calibration = capture_calibration(get_settings().camera_calibration_path, evidence.get("capture_sha256"))
            if calibration is None or calibration.profile_sha256 != evidence.get("profile_sha256"):
                return None
        if automatic:
            receipt = correction.get("crop_validation")
            if not isinstance(receipt, dict):
                return None
            if receipt != crop_validation_evidence(get_settings(), receipt.get("model_sha256")):
                return None
    if variant in {"original", "source_corrected"}:
        record = variants.get(variant) or variants.get("source_corrected") or variants.get("original")
        if isinstance(record, dict) and record.get("path"):
            return str(record["path"])
        # Legacy rows stored their only crop directly, before variant records were added.
        return photo.original_storage_path or (getattr(photo, "storage_path", None) if not variants else None)

    record = variants.get(variant)
    if isinstance(record, dict) and record.get("path"):
        return str(record["path"])

    return None


def released_variant_from_photo(photo: Any) -> str:
    """Re-evaluate legacy metadata at read time rather than trusting it."""
    metadata = getattr(photo, "enhancement_applied", None) or {}
    variants = variants_from_metadata(metadata)
    if not variants:
        return "original"
    return select_safe_variant(variants, photo_status=getattr(photo, "status", None), metadata=metadata)


def photo_image_url(photo: Any, variant: str | None = None) -> str:
    metadata = getattr(photo, "enhancement_applied", None) or {}
    version = metadata.get("released_asset_id") or (metadata.get("correction") or {}).get("asset_id")
    params = {"variant": variant} if variant else {}
    if version:
        params["v"] = str(version)
    url = f"/api/v1/photos/{photo.id}/image"
    return f"{url}?{urlencode(params)}" if params else url


def variant_urls(photo: Any) -> dict[str, str | None]:
    names = [*VARIANT_ORDER, *CROP_REVIEW_PREVIEWS]
    for name in variants_from_metadata(getattr(photo, "enhancement_applied", None)):
        if name not in names:
            names.append(name)
    return {
        name: photo_image_url(photo, name) if variant_path_from_photo(photo, name) else None
        for name in names
    }
