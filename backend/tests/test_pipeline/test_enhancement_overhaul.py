from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np


def _color_scene(height: int = 180, width: int = 240) -> np.ndarray:
    gradient = np.tile(np.linspace(45, 210, width, dtype=np.float32), (height, 1))
    image = np.dstack((gradient * 0.86, gradient, gradient * 1.06))
    cv2.rectangle(image, (24, 28), (104, height - 24), (70, 128, 188), -1)
    cv2.circle(image, (width - 66, height // 2), 42, (185, 112, 72), -1)
    return np.clip(image, 0, 255).astype(np.uint8)


def test_direct_array_contract_normalizes_gray_alpha_uint16_and_float() -> None:
    from app.pipeline.enhancement_types import normalize_bgr_uint8

    gray, gray_contract = normalize_bgr_uint8(np.full((12, 18), 77, dtype=np.uint8))
    assert gray.shape == (12, 18, 3)
    assert gray_contract.grayscale_expanded is True

    bgra = np.zeros((8, 10, 4), dtype=np.uint8)
    bgra[:, :, 3] = 128
    alpha, alpha_contract = normalize_bgr_uint8(bgra)
    assert alpha_contract.alpha_composited is True
    assert 126 <= int(alpha[0, 0, 0]) <= 128

    uint16, uint16_contract = normalize_bgr_uint8(np.full((6, 7, 3), 32768, dtype=np.uint16))
    assert uint16.dtype == np.uint8
    assert 127 <= int(uint16[0, 0, 0]) <= 128
    assert uint16_contract.range_conversion == "uint16_full_range_to_uint8"

    unit_float, float_contract = normalize_bgr_uint8(np.full((6, 7, 3), 0.5, dtype=np.float32))
    assert 127 <= int(unit_float[0, 0, 0]) <= 128
    assert float_contract.range_conversion == "float_unit_to_uint8"


def test_degradation_profile_has_evidence_and_preserves_high_key_monochrome() -> None:
    from app.pipeline.degradation import analyze_degradation
    from app.pipeline.enhancement import conservative_enhance_photo

    ramp = np.tile(np.linspace(136, 245, 220, dtype=np.uint8), (140, 1))
    image = np.dstack((ramp, ramp, ramp))
    report = analyze_degradation(image)
    output, metadata = conservative_enhance_photo(image, degradation_report=report)

    assert report.monochrome_score >= 0.75
    assert report.signals["contrast"].evidence["p50"] > 0
    assert metadata["planner"]["no_op"] is True
    assert "MONOCHROME_PRESERVE_SOURCE" in metadata["planner"]["reason_codes"]
    assert np.array_equal(output, image)


def test_ambiguous_sepia_is_noop_in_faithful_route() -> None:
    from app.pipeline.degradation import analyze_degradation
    from app.pipeline.enhancement import conservative_enhance_photo

    base = np.tile(np.linspace(35, 180, 220, dtype=np.uint8), (140, 1)).astype(np.int16)
    image = np.dstack((base, base + 18, base + 48)).clip(0, 255).astype(np.uint8)
    report = analyze_degradation(image)
    output, metadata = conservative_enhance_photo(image, degradation_report=report)

    assert report.sepia_ambiguity >= 0.58
    assert metadata["planner"]["no_op"] is True
    assert np.array_equal(output, image)


def test_strong_luminance_damage_improves_without_chroma_correction() -> None:
    from app.evaluation.enhancement import measure_enhancement
    from app.pipeline.enhancement import conservative_enhance_photo
    from scripts.evaluate_enhancement import _synthetic_cases

    fixtures = {case["id"]: case for case in _synthetic_cases()}
    expected_minimum_psnr_gain = {"faded_yellow_cast": 0.25, "underexposed": 0.15}
    for case_id, minimum_gain in expected_minimum_psnr_gain.items():
        case = fixtures[case_id]
        source = case["source"]
        output, metadata = conservative_enhance_photo(source)
        source_metrics = measure_enhancement(source, case["reference"])
        output_metrics = measure_enhancement(output, case["reference"])

        assert source_metrics.psnr is not None
        assert output_metrics.psnr is not None
        assert output_metrics.psnr >= source_metrics.psnr + minimum_gain
        assert metadata["planner"]["allowed_steps"] == ["bounded_luminance_tone"]
        assert "WARM_TONE_CHROMA_PRESERVED" in metadata["planner"]["reason_codes"]
        quality_delta = metadata["steps"][0]["quality_delta"]
        assert quality_delta["color_delta"] < 0.10
        assert quality_delta["clipping_delta"] == 0.0
        assert quality_delta["local_structure_min"] > 0.98


def test_luminance_tuning_keeps_good_warm_sepia_and_monochrome_fixtures_exact() -> None:
    from app.pipeline.enhancement import conservative_enhance_photo
    from scripts.evaluate_enhancement import _synthetic_cases

    fixtures = {case["id"]: case for case in _synthetic_cases()}
    for case_id in ("already_good", "intentional_sepia", "intentional_monochrome"):
        source = fixtures[case_id]["source"]
        output, metadata = conservative_enhance_photo(source)
        assert metadata["planner"]["no_op"] is True
        assert np.array_equal(output, source)


def test_detected_noise_is_reduced_without_reporting_reverted_steps() -> None:
    from app.pipeline.degradation import analyze_degradation
    from app.pipeline.enhancement import conservative_enhance_photo
    from app.pipeline.quality_assessment import QualityAssessor

    clean = _color_scene()
    noise = np.random.default_rng(3).normal(0, 10, clean.shape).astype(np.int16)
    noisy = np.clip(clean.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    report = analyze_degradation(noisy)
    output, metadata = conservative_enhance_photo(noisy, degradation_report=report)

    assert report.should_denoise is True
    assert "edge_texture_aware_denoise" in metadata["planner"]["allowed_steps"]
    assert QualityAssessor().assess(output).noise_level < QualityAssessor().assess(noisy).noise_level
    assert metadata["planner"]["allowed_steps"] == [
        step["name"] for step in metadata["steps"] if step["quality_gate"] == "accepted"
    ]
    assert metadata["input_pixel_sha256"] != metadata["output_pixel_sha256"]
    assert metadata["runtime_ms"] >= 0


def test_white_balance_requires_reliable_neutral_evidence_and_is_bounded() -> None:
    from app.pipeline.color_correction import apply_color_corrections
    from app.pipeline.degradation import analyze_degradation
    from app.pipeline.presets import get_enhancement_config

    image = _color_scene()
    report = replace(
        analyze_degradation(image),
        should_correct_color=True,
        cast_score=0.45,
        cast_confidence=0.92,
        cast_lab=(8.0, -6.0),
        neutral_fraction=0.40,
        monochrome_score=0.0,
        sepia_ambiguity=0.0,
    )
    output, metadata = apply_color_corrections(image, report, get_enhancement_config("balanced"))
    assert metadata["color_correction"] is True
    assert max(abs(value) for value in metadata["evidence"]["applied_lab_delta"]) <= 8.0
    assert output.shape == image.shape

    ambiguous = replace(report, sepia_ambiguity=0.8)
    preserved, skipped = apply_color_corrections(image, ambiguous, get_enhancement_config("balanced"))
    assert skipped["color_correction"] is False
    assert np.array_equal(preserved, image)


def test_quality_critic_rejects_local_structural_mutation() -> None:
    from app.pipeline.quality_critic import evaluate_candidate

    source = _color_scene(240, 320)
    candidate = source.copy()
    candidate[82:158, 126:198] = cv2.GaussianBlur(candidate[82:158, 126:198], (0, 0), 12)
    cv2.rectangle(candidate, (145, 106), (178, 132), (250, 250, 250), -1)
    gates = evaluate_candidate(source, candidate, {"method": "deep_learning", "variant_kind": "restoration_candidate"})
    assert next(gate for gate in gates if gate["name"] == "local_structure")["passed"] is False


def test_quality_critic_rejects_anisotropic_dimensions_and_new_clipping() -> None:
    from app.pipeline.quality_critic import evaluate_candidate

    source = _color_scene()
    candidate = cv2.resize(source, (source.shape[1] * 2, source.shape[0]), interpolation=cv2.INTER_CUBIC)
    candidate = cv2.convertScaleAbs(candidate, alpha=2.8, beta=80)
    gates = evaluate_candidate(source, candidate, {"method": "deterministic"})
    assert next(gate for gate in gates if gate["name"] == "dimension_contract")["passed"] is False
    assert next(gate for gate in gates if gate["name"] == "clipping_regression")["passed"] is False


def test_deep_restoration_requires_explicit_opt_in_and_never_auto_releases() -> None:
    from app.pipeline.dl_enhancement import dl_enhance_photo

    image = cv2.resize(_color_scene(), (80, 60), interpolation=cv2.INTER_AREA)

    class MockEnhancer:
        def enhance(self, value, outscale=2):
            return cv2.resize(
                value, (value.shape[1] * outscale, value.shape[0] * outscale), interpolation=cv2.INTER_CUBIC
            ), None

    with patch("app.pipeline.dl_enhancement.get_dl_models", return_value=(MockEnhancer(), None)) as loader:
        source, disabled = dl_enhance_photo(image)
        loader.assert_not_called()
        restored, enabled = dl_enhance_photo(
            image,
            allow_model_restoration=True,
            model_paths={"realesrgan": Path("mock.pth")},
            model_artifacts={"realesrgan": {"sha256": "a" * 64}},
        )

    assert np.array_equal(source, image)
    assert disabled["planner"]["unsafe_models_enabled"] is False
    assert restored.shape[:2] == (image.shape[0] * 2, image.shape[1] * 2)
    assert enabled["variant_kind"] == "restoration_candidate"
    assert enabled["model_artifacts"]["realesrgan"]["sha256"] == "a" * 64
    assert enabled["auto_release"] is False


def test_canonical_variant_bundle_never_selects_model_restoration(tmp_path) -> None:
    from app.pipeline.variants import build_variant_bundle

    source = _color_scene(80, 100)
    restored = cv2.resize(source, (200, 160), interpolation=cv2.INTER_CUBIC)
    bundle = build_variant_bundle(
        original_path=tmp_path / "source.jpg",
        original=source,
        canonical_kinds=True,
        restoration_path=tmp_path / "restored.jpg",
        restoration=restored,
        restoration_metadata={
            "method": "deep_learning_restoration",
            "variant_kind": "restoration_candidate",
            "generative": True,
            "auto_release": True,
        },
    )

    assert set(bundle["variants"]) == {"source_corrected", "restoration_candidate"}
    assert bundle["selected_variant"] == "source_corrected"
    assert any("model_restoration" in reason for reason in bundle["release"]["reasons"])


def test_flash_and_fold_compatibility_helpers_preserve_source_pixels() -> None:
    from app.pipeline.enhancement import recover_flash_highlights, reduce_fold_shadows

    image = np.full((80, 100, 3), 80, dtype=np.uint8)
    cv2.rectangle(image, (20, 20), (72, 58), (255, 255, 255), -1)
    flash, flash_metadata = recover_flash_highlights(image)
    folds, fold_metadata = reduce_fold_shadows(image)
    assert flash_metadata["flash_recovery"] is False
    assert fold_metadata["fold_removal"] is False
    assert np.array_equal(flash, image)
    assert np.array_equal(folds, image)
