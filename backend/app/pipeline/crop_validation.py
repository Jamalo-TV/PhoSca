"""Bind automatic precise crop downloads to the evaluated configuration."""
from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path

import cv2

from app.config import Settings
from app.pipeline.crop_sampling import OWNED_BILINEAR
from app.pipeline.confidence import is_sha256, load_confidence_calibration
from app.pipeline.decisions import ReviewReason
from app.pipeline.model_registry import sha256
from app.pipeline.multipanel_prints import MULTIPANEL_PRINT_REVISION
from app.pipeline.segmentation_filter import SOURCE_CONTOUR_REVISION

CROP_VALIDATION_SCHEMA = 31
CROP_VALIDATION_UNAVAILABLE = "CROP_VALIDATION_UNAVAILABLE"


@lru_cache(maxsize=8)
def _hash_file(path: Path, modified_ns: int, size: int) -> str:
    return sha256(path)


def _optional_hash(path: Path) -> str | None:
    try:
        stat = path.stat()
        return _hash_file(path, stat.st_mtime_ns, stat.st_size)
    except OSError:
        return None


def crop_validation_profile(settings: Settings) -> dict:
    visible_model = settings.visible_mask_model_path if settings.enable_precise_photo_crops else None
    return {
        "pipeline_version": settings.pipeline_version,
        "precise_photo_crops": settings.enable_precise_photo_crops,
        "crop_sampling": OWNED_BILINEAR if settings.enable_precise_photo_crops else None,
        "page_plane_preview": 2 if settings.enable_precise_photo_crops else None,
        "print_boundary": 6 if settings.enable_precise_photo_crops else None,
        "registered_model_required": settings.require_registered_models,
        "model_registry_sha256": _optional_hash(settings.model_manifest_path) if settings.require_registered_models else None,
        "source_contour_refinement": SOURCE_CONTOUR_REVISION if settings.enable_precise_photo_crops and visible_model is None else None,
        "visible_mask_refinement": None if visible_model is None else {
            "model_sha256": _optional_hash(visible_model),
            "manifest_sha256": _optional_hash(settings.model_manifest_path),
            "sleeve_reflection_filter": 1,
            "ownership_polygon_check": 1,
        },
        "classical_fallback": settings.enable_classical_segmentation_fallback,
        "classical_line_pairs": 1 if settings.enable_classical_segmentation_fallback else None,
        "separated_contrast_prints": (
            1 if settings.enable_precise_photo_crops and settings.enable_classical_segmentation_fallback else None
        ),
        "empty_drawn_frame_rejection": (
            1 if settings.enable_precise_photo_crops and settings.enable_classical_segmentation_fallback else None
        ),
        "album_window_detection": (
            1 if settings.enable_precise_photo_crops and settings.enable_classical_segmentation_fallback else None
        ),
        "multipanel_print_detection": (
            MULTIPANEL_PRINT_REVISION
            if settings.enable_precise_photo_crops and settings.enable_classical_segmentation_fallback else None
        ),
        "dense_tile_inference": (
            2 if settings.enable_precise_photo_crops and settings.enable_classical_segmentation_fallback else None
        ),
        "deskew_inference": 1 if settings.enable_precise_photo_crops and settings.enable_classical_segmentation_fallback else None,
        "border_analysis": settings.enable_border_trim,
        "curved_motif_check": 7 if settings.enable_precise_photo_crops and settings.enable_border_trim else None,
        "protected_region_guard": settings.enable_protected_region_guard,
        "orientation_enabled": settings.enable_photo_orientation_correction,
        "orientation_model_sha256": (
            _optional_hash(settings.photo_orientation_model_path) if settings.enable_photo_orientation_correction else None
        ),
        "require_confidence_calibration": settings.require_confidence_calibration,
        "calibration_sha256": _optional_hash(settings.segmentation_calibration_path),
        "proposal_threshold": settings.yolo_proposal_threshold,
        "acceptance_threshold": settings.yolo_confidence_threshold,
        "min_detection_aspect_ratio": settings.segmentation_min_aspect_ratio,
        "max_detection_aspect_ratio": settings.segmentation_max_aspect_ratio,
        "opencv_version": cv2.__version__,
    }


def segmentation_matches_crop_profile(settings: Settings, segmentation: dict) -> bool:
    expected = crop_validation_profile(settings)
    if _inference_failed(segmentation) or segmentation.get("validation_profile") != expected:
        return False
    return _matches_refinement(expected, segmentation)


def _inference_failed(segmentation: dict) -> bool:
    return bool(segmentation.get("onnx_error") or any(
        (segmentation.get(stage) or {}).get("error") for stage in ("dense_tile_inference", "deskew_inference")
    ))


def _matches_refinement(profile: dict, segmentation: dict) -> bool:
    revision = profile["source_contour_refinement"]
    if revision is not None:
        used = segmentation.get("source_contour_refinement")
        return isinstance(used, dict) and used.get("revision") == revision and not used.get("error")
    refiner = profile["visible_mask_refinement"]
    if refiner is None:
        return True
    used = segmentation.get("visible_mask_refinement") or {}
    return bool(
        refiner["model_sha256"] and used.get("model_sha256") == refiner["model_sha256"]
        and not used.get("error") and not used.get("ownership_unresolved")
    )


def detector_calibration_profile(settings: Settings, *, segmentation: dict | None = None) -> dict | None:
    profile = crop_validation_profile(settings)
    model = _optional_hash(settings.yolo_model_path)
    if not is_sha256(model) or not profile["model_registry_sha256"]:
        return None
    refiner = profile["visible_mask_refinement"]
    if refiner is not None and not (refiner["model_sha256"] and refiner["manifest_sha256"]):
        return None
    if segmentation is not None and (
        _inference_failed(segmentation)
        or (segmentation.get("model_artifact") or {}).get("sha256") != model
        or not _matches_refinement(profile, segmentation)
    ):
        return None
    return {"model_sha256": model, **{key: profile[key] for key in (
        "pipeline_version", "precise_photo_crops", "model_registry_sha256", "visible_mask_refinement", "source_contour_refinement",
        "print_boundary",
        "classical_fallback", "classical_line_pairs", "separated_contrast_prints", "empty_drawn_frame_rejection",
        "album_window_detection", "multipanel_print_detection", "dense_tile_inference",
        "deskew_inference", "proposal_threshold",
        "acceptance_threshold", "min_detection_aspect_ratio",
        "max_detection_aspect_ratio", "opencv_version",
    )}}


def route_ready_crop_modes_to_review(modes: dict, reasons: list[str]) -> None:
    for mode in modes.values():
        if mode.get("status") == "ready":
            mode.update(status="review_needed", geometry_status="ready", reason_codes=reasons.copy())


def automatic_crop_review_reasons(
    settings: Settings, confidence: float, geometry_valid: bool, calibrated: bool,
    detection_reasons: Sequence[str], orientation_reason: str | None,
) -> list[str]:
    reasons = list(detection_reasons)
    if settings.require_confidence_calibration and calibrated is not True:
        reasons.append("SEGMENTATION_CONFIDENCE_UNCALIBRATED")
    if not geometry_valid:
        reasons.append(ReviewReason.QUADRILATERAL_UNSTABLE.value)
    if not confidence >= settings.yolo_confidence_threshold:
        reasons.append(ReviewReason.PHOTO_COUNT_UNCERTAIN.value)
    if orientation_reason:
        reasons.append(orientation_reason)
    return reasons


def geometrically_ready_crop_modes(
    modes: dict, *, image_only_present: bool, image_only_is_full_print: bool,
) -> set[str]:
    full_ready = (modes.get("full_print") or {}).get("status") == "ready"
    ready = {"full_print"} if full_ready else set()
    motif = modes.get("image_only") or {}
    independent = motif.get("independent_of_full_print") is True and not image_only_is_full_print
    if image_only_present and motif.get("status") == "ready" and (full_ready or independent):
        ready.add("image_only")
    return ready


@lru_cache(maxsize=4)
def _read_report(path: Path, modified_ns: int, size: int) -> tuple[dict, str]:
    payload = path.read_bytes()
    return json.loads(payload), hashlib.sha256(payload).hexdigest()


def crop_validation_evidence(settings: Settings, model_sha256: str | None) -> dict | None:
    """Return evidence only for an intact, matching real-album validation report."""
    if not isinstance(model_sha256, str) or len(model_sha256) != 64:
        return None
    try:
        path = settings.crop_validation_report_path
        stat = path.stat()
        report, digest = _read_report(path, stat.st_mtime_ns, stat.st_size)
        if (
            not isinstance(report, dict) or report.get("schema_version") != CROP_VALIDATION_SCHEMA
            or report.get("model_sha256") != model_sha256
            or _optional_hash(settings.yolo_model_path) != model_sha256
        ):
            return None
        current = crop_validation_profile(settings)
        if report.get("validation_profile") != current or not current["require_confidence_calibration"]:
            return None
        load_confidence_calibration(settings.segmentation_calibration_path, detector_calibration_profile(settings))
        refiner = current["visible_mask_refinement"]
        if refiner is not None and not (refiner["model_sha256"] and refiner["manifest_sha256"]):
            return None
        split = report.get("split") or {}
        gates = report.get("validation_gates") or {}
        modes = gates.get("rendered_crop_modes") or {}
        manifest_sha256 = report.get("dataset_manifest_sha256")
        if (
            report.get("legacy_crop") is not False or report.get("geometry_only") is not False
            or split.get("synthetic") or split.get("album_independent") is not True
            or not is_sha256(manifest_sha256)
            or gates.get("locked_real_album_manifest") is not True
            or gates.get("locked_reference_artifacts") is not True
            or gates.get("calibration_holdout_disjoint") is not True
            or gates.get("shared_recoverable_album_groups") is not True
            or gates.get("standard_detection_thresholds") is not True
            or gates.get("detection") is not True or gates.get("final_instance_segmentation") is not True
            or modes.get("full_print") is not True or modes.get("image_only") is not True
        ):
            return None
        return {"model_sha256": model_sha256, "report_sha256": digest}
    except (OSError, ValueError, TypeError, AttributeError):
        return None
