from __future__ import annotations

import hashlib
from time import perf_counter

import cv2
import numpy as np

from app.pipeline.color_correction import apply_color_corrections
from app.pipeline.degradation import DegradationReport, analyze_degradation
from app.pipeline.enhancement_types import (
    EnhancementOperation,
    EnhancementPlan,
    normalize_bgr_uint8,
)
from app.pipeline.presets import EnhancementConfig, get_enhancement_config
from app.pipeline.quality_assessment import QualityAssessor, QualityDelta


def prepare_for_cutout(image: np.ndarray) -> tuple[np.ndarray, dict]:
    """Apply conservative, geometry-preserving cleanup before photo detection."""
    normalized, contract = normalize_bgr_uint8(image)
    lab = cv2.cvtColor(normalized, cv2.COLOR_BGR2LAB)
    lightness, channel_a, channel_b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.4, tileGridSize=(8, 8))
    local_contrast = clahe.apply(lightness)
    blended_lightness = cv2.addWeighted(lightness, 0.72, local_contrast, 0.28, 0)
    prepared = cv2.cvtColor(cv2.merge((blended_lightness, channel_a, channel_b)), cv2.COLOR_LAB2BGR)
    prepared = cv2.bilateralFilter(prepared, d=5, sigmaColor=12, sigmaSpace=5)
    return prepared, {
        "method": "conservative_cutout_prep",
        "order": "before_segmentation_and_crop",
        "geometry_preserved": prepared.shape[:2] == normalized.shape[:2],
        "steps": ["luminance_clahe_blend", "edge_preserving_bilateral"],
        "clahe_clip_limit": 1.4,
        "clahe_blend_weight": 0.28,
        "input_contract": contract.as_dict(),
    }


def _array_sha256(image: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(str(image.dtype).encode("ascii"))
    digest.update(np.asarray(image.shape, dtype=np.int64).tobytes())
    digest.update(np.ascontiguousarray(image).tobytes())
    return digest.hexdigest()


def plan_conservative_enhancement(report: DegradationReport, config: EnhancementConfig) -> EnhancementPlan:
    if report.monochrome_score >= 0.78:
        return EnhancementPlan(
            route="noop",
            reason_codes=("MONOCHROME_PRESERVE_SOURCE",),
            confidence=report.analysis_confidence,
        )

    warm_tone_ambiguous = report.sepia_ambiguity >= 0.58
    strong_luminance_damage = report.contrast_score >= 0.45 or report.underexposure_score >= 0.28
    if warm_tone_ambiguous and not strong_luminance_damage:
        return EnhancementPlan(
            route="noop",
            reason_codes=("INTENTIONAL_WARM_TONE_AMBIGUOUS",),
            confidence=report.analysis_confidence,
        )

    operations: list[EnhancementOperation] = []
    if not warm_tone_ambiguous and config.denoise_enabled and report.should_denoise:
        operations.append(
            EnhancementOperation(
                name="edge_texture_aware_denoise",
                params={
                    "strength": min(0.62, max(config.denoise_strength, report.denoise_strength_recommended * 0.92))
                },
                reason_codes=("NOISE_OR_COMPRESSION_DETECTED",),
                expected_effect="reduce flat-region luma/chroma noise without smoothing supported edges",
            )
        )
    if not warm_tone_ambiguous and config.color_correction_enabled and report.should_correct_color:
        operations.append(
            EnhancementOperation(
                name="neutral_white_balance",
                params={"max_lab_delta": 8.0, "cast_confidence": report.cast_confidence},
                reason_codes=("RELIABLE_NEUTRAL_COLOR_CAST",),
                expected_effect="reduce a measured neutral-region chroma cast",
            )
        )
    if report.should_adjust_tone:
        operations.append(
            EnhancementOperation(
                name="bounded_luminance_tone",
                params={
                    "max_gain": 1.16,
                    "blend": min(0.42, 0.18 + report.contrast_score * 0.24),
                    "midtone_lift": min(12.0, report.underexposure_score * 24.0),
                },
                reason_codes=tuple(
                    code
                    for code, enabled in (
                        ("LOW_CONTRAST_DETECTED", report.contrast_score >= 0.28),
                        ("UNDEREXPOSURE_DETECTED", report.underexposure_score >= 0.28),
                        ("WARM_TONE_CHROMA_PRESERVED", warm_tone_ambiguous),
                    )
                    if enabled
                ),
                expected_effect="expand luminance contrast without altering chroma or clipping endpoints",
            )
        )
    if not warm_tone_ambiguous and report.should_sharpen and config.sharpen_amount > 0:
        operations.append(
            EnhancementOperation(
                name="thresholded_scale_aware_sharpen",
                params={"amount": min(0.20, config.sharpen_amount)},
                reason_codes=("LOW_EDGE_ACUTANCE",),
                expected_effect="restore mild edge acutance with local overshoot bounds",
            )
        )

    if not operations:
        return EnhancementPlan(
            route="noop",
            reason_codes=("NO_ACTIONABLE_DEGRADATION",),
            confidence=report.analysis_confidence,
        )
    return EnhancementPlan(
        route="conservative_deterministic",
        operations=tuple(operations),
        reason_codes=tuple(dict.fromkeys(code for operation in operations for code in operation.reason_codes)),
        confidence=report.analysis_confidence,
    )


def _edge_texture_aware_denoise(
    image: np.ndarray, report: DegradationReport, strength: float
) -> tuple[np.ndarray, dict]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_f = gray.astype(np.float32)
    gradient = cv2.magnitude(
        cv2.Sobel(gray_f, cv2.CV_32F, 1, 0, ksize=3),
        cv2.Sobel(gray_f, cv2.CV_32F, 0, 1, ksize=3),
    )
    local_mean = cv2.GaussianBlur(gray_f, (0, 0), 2.0)
    local_sq = cv2.GaussianBlur(gray_f * gray_f, (0, 0), 2.0)
    local_std = np.sqrt(np.maximum(local_sq - local_mean * local_mean, 0.0))
    edge_threshold = max(5.0, float(np.percentile(gradient, 68)))
    texture_threshold = max(7.0, float(np.percentile(local_std, 62)))
    flat_weight = np.clip(1.0 - gradient / (edge_threshold * 2.4), 0.0, 1.0) * np.clip(
        1.0 - local_std / (texture_threshold * 2.2), 0.0, 1.0
    )
    flat_weight = cv2.GaussianBlur(flat_weight, (0, 0), 1.1)[:, :, None]

    h_luma = max(2, min(8, int(round(3.0 + strength * 8.0))))
    h_chroma = max(2, min(8, int(round(3.0 + min(1.0, report.chroma_noise / 10.0) * 6.0))))
    denoised = cv2.fastNlMeansDenoisingColored(image, None, h_luma, h_chroma, 7, 21)
    blend = np.clip(flat_weight * min(0.74, 0.42 + strength * 0.52), 0.0, 0.74)
    candidate = np.rint(image.astype(np.float32) * (1.0 - blend) + denoised.astype(np.float32) * blend).astype(np.uint8)
    return candidate, {
        "method": "flat_region_nlm_blend",
        "h_luma": h_luma,
        "h_chroma": h_chroma,
        "max_blend": float(blend.max()),
        "mean_blend": float(blend.mean()),
    }


def _bounded_luminance_tone(
    image: np.ndarray, max_gain: float, blend: float, midtone_lift: float = 0.0
) -> tuple[np.ndarray, dict]:
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lightness = lab[:, :, 0].astype(np.float32)
    low, median, high = (float(value) for value in np.percentile(lightness, [5, 50, 95]))
    current_range = max(1.0, high - low)
    desired_gain = min(max_gain, max(1.0, 176.0 / current_range))
    mapped = median + (lightness - median) * desired_gain
    mapped = np.clip(mapped, max(2.0, low - 6.0), min(253.0, high + 6.0))
    if midtone_lift > 0.0:
        # Smoothly lift recoverable midtones while pinning mathematical black
        # and white, avoiding a flat additive exposure offset.
        mapped += midtone_lift * np.sin(np.pi * np.clip(lightness / 255.0, 0.0, 1.0))
    toned = lightness * (1.0 - blend) + mapped * blend
    output_lab = lab.copy()
    output_lab[:, :, 0] = np.rint(np.clip(toned, 0.0, 255.0)).astype(np.uint8)
    return cv2.cvtColor(output_lab, cv2.COLOR_LAB2BGR), {
        "method": "bounded_lab_luminance_curve",
        "input_percentiles": [low, median, high],
        "gain": desired_gain,
        "blend": blend,
        "midtone_lift": midtone_lift,
    }


def _thresholded_sharpen(image: np.ndarray, report: DegradationReport, amount: float) -> tuple[np.ndarray, dict]:
    sigma = float(np.clip(min(image.shape[:2]) / 650.0, 0.65, 1.35))
    source = image.astype(np.float32)
    blurred = cv2.GaussianBlur(source, (0, 0), sigma)
    detail = source - blurred
    magnitude = np.max(np.abs(detail), axis=2)
    threshold = max(2.5, report.luma_noise * 1.15)
    support = np.clip((magnitude - threshold) / max(threshold * 2.0, 1.0), 0.0, 1.0)[:, :, None]
    candidate = source + detail * support * amount
    local_min = cv2.erode(image, np.ones((3, 3), np.uint8)).astype(np.float32) - 2.0
    local_max = cv2.dilate(image, np.ones((3, 3), np.uint8)).astype(np.float32) + 2.0
    candidate = np.minimum(np.maximum(candidate, local_min), local_max)
    return np.rint(np.clip(candidate, 0.0, 255.0)).astype(np.uint8), {
        "method": "thresholded_local_bound_unsharp",
        "sigma": sigma,
        "amount": amount,
        "detail_threshold": threshold,
        "support_fraction": float(np.mean(support > 0.05)),
        "overshoot_bound": 2.0,
    }


def _operation_accepted(name: str, delta: QualityDelta, before: np.ndarray, after: np.ndarray) -> tuple[bool, str]:
    if np.array_equal(before, after):
        return False, "no_effect"
    if (
        delta.clipping_delta > 0.012
        or max(delta.channel_shadow_delta + delta.channel_highlight_delta, default=0.0) > 0.02
    ):
        return False, "new_clipping"
    if delta.local_structure_min < 0.82 or delta.halo_delta > 0.018:
        return False, "structural_or_halo_risk"
    if name == "edge_texture_aware_denoise":
        accepted = delta.noise_delta <= -0.10 and delta.texture_ratio >= 0.76 and abs(delta.saturation_delta) <= 0.025
        return accepted, "target_noise_not_improved" if not accepted else "target_improved"
    if name == "neutral_white_balance":
        accepted = delta.color_delta <= 9.0 and abs(delta.saturation_delta) <= 0.035
        return accepted, "color_motion_exceeds_bound" if not accepted else "bounded_cast_correction"
    if name == "bounded_luminance_tone":
        luminance_improved = delta.dynamic_range_delta >= 0.006 or 1.0 <= delta.luminance_delta <= 12.0
        accepted = luminance_improved and delta.color_delta <= 1.5
        return accepted, "target_contrast_not_improved" if not accepted else "target_improved"
    if name == "thresholded_scale_aware_sharpen":
        accepted = delta.sharpness_delta > 0.0 and delta.noise_delta <= 0.8 and delta.texture_ratio <= 1.35
        return accepted, "target_acutance_not_improved" if not accepted else "target_improved"
    return not QualityAssessor().should_revert(delta, min_quality_delta=-0.015), "generic_gate"


def conservative_enhance_photo(
    image: np.ndarray,
    preset: str | None = "auto",
    degradation_report: DegradationReport | None = None,
) -> tuple[np.ndarray, dict]:
    """Produce a faithful, deterministic candidate without Torch or generative models."""
    started = perf_counter()
    source, contract = normalize_bgr_uint8(image)
    report = degradation_report or analyze_degradation(source)
    config = get_enhancement_config(preset, recommended_preset=report.recommended_preset)
    plan = plan_conservative_enhancement(report, config)
    assessor = QualityAssessor()
    working = source.copy()
    step_metadata: list[dict] = []

    for operation in plan.operations:
        if operation.name == "edge_texture_aware_denoise":
            candidate, details = _edge_texture_aware_denoise(working, report, float(operation.params["strength"]))
        elif operation.name == "neutral_white_balance":
            candidate, details = apply_color_corrections(working, report, config)
        elif operation.name == "bounded_luminance_tone":
            candidate, details = _bounded_luminance_tone(
                working,
                max_gain=float(operation.params["max_gain"]),
                blend=float(operation.params["blend"]),
                midtone_lift=float(operation.params.get("midtone_lift", 0.0)),
            )
        elif operation.name == "thresholded_scale_aware_sharpen":
            candidate, details = _thresholded_sharpen(working, report, float(operation.params["amount"]))
        else:  # Defensive fail-closed behavior for future plan/schema changes.
            step_metadata.append(
                {"name": operation.name, "applied": False, "quality_gate": "reverted", "reason": "unknown_operation"}
            )
            continue

        delta = assessor.compare(working, candidate)
        accepted, reason = _operation_accepted(operation.name, delta, working, candidate)
        step_metadata.append(
            {
                "name": operation.name,
                "applied": accepted,
                "quality_gate": "accepted" if accepted else "reverted",
                "reason": reason,
                "params": operation.params,
                "evidence": details,
                "quality_delta": delta.as_dict(),
            }
        )
        if accepted:
            working = candidate

    final_delta = assessor.compare(source, working)
    final_reverted = assessor.should_revert(final_delta, min_quality_delta=-0.02)
    if final_reverted:
        working = source.copy()
    allowed_steps = [step["name"] for step in step_metadata if step.get("applied") is True]
    no_op = bool(np.array_equal(working, source))
    metadata = {
        "method": "no_op" if no_op else "conservative_deterministic",
        "variant_kind": "conservative_enhanced",
        "preset": config.preset,
        "config": config.as_dict(),
        "models": [],
        "input_contract": contract.as_dict(),
        "degradation_report": report.as_dict(),
        "plan": plan.as_dict(),
        "steps": step_metadata,
        "planner": {
            "route": plan.route,
            "no_op": no_op,
            "allowed_steps": allowed_steps,
            "reason_codes": list(plan.reason_codes),
        },
        "final_quality_delta": final_delta.as_dict(),
        "final_quality_gate": "reverted" if final_reverted else "accepted",
        "runtime_ms": round((perf_counter() - started) * 1000.0, 3),
        "input_pixel_sha256": _array_sha256(source),
        "output_pixel_sha256": _array_sha256(working),
        "auto_release": False,
    }
    return working, metadata


def recover_flash_highlights(image: np.ndarray) -> tuple[np.ndarray, dict]:
    """Do not invent saturated content in a faithful enhancement route."""
    normalized, _ = normalize_bgr_uint8(image)
    blown = int(cv2.countNonZero(cv2.inRange(normalized, np.array([250] * 3, np.uint8), np.array([255] * 3, np.uint8))))
    return normalized.copy(), {
        "flash_recovery": False,
        "blown_pixels": blown,
        "reason": "saturated_pixels_have_no_recoverable_source_evidence",
    }


def reduce_fold_shadows(image: np.ndarray) -> tuple[np.ndarray, dict]:
    """Preserve pixels until a validated fold/shadow detector is available."""
    normalized, _ = normalize_bgr_uint8(image)
    return normalized.copy(), {
        "fold_removal": False,
        "fold_pixels": 0,
        "reason": "validated_fold_mask_unavailable",
    }


def enhance_photo(image: np.ndarray) -> tuple[np.ndarray, dict]:
    """Compatibility wrapper for the deterministic conservative producer."""
    return conservative_enhance_photo(image)
