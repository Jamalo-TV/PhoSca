from __future__ import annotations

import numpy as np
import pytest


def _variants() -> dict:
    return {
        "original": {"path": "original.jpg", "metadata": {"method": "source"}, "warnings": []},
        "enhanced": {"path": "enhanced.jpg", "metadata": {"auto_release": True}, "warnings": []},
    }


def test_release_selector_vetoes_review_and_stage_errors() -> None:
    from app.pipeline.release import select_released_variant

    assert select_released_variant(_variants(), photo_status="review_needed")[0] == "original"
    failed = _variants()
    failed["enhanced"]["metadata"] = {"auto_release": True, "method": "failed"}
    assert select_released_variant(failed)[0] == "original"


def test_release_selector_only_releases_explicitly_gated_variant() -> None:
    from app.pipeline.release import select_released_variant

    variants = _variants()
    variants["enhanced"]["metadata"] = {}
    selected, reasons = select_released_variant(variants)
    assert selected == "original"
    assert "no_derived_variant_passed_release_gate" in reasons


def test_release_policy_shadow_and_kill_switch_veto_candidates() -> None:
    from app.pipeline.release import select_released_variant

    selected, reasons = select_released_variant(
        _variants(),
        metadata={"release_policy": {"shadow_mode": True, "kill_switch": True, "canary_eligible": False}},
    )
    assert selected == "original"
    assert reasons == ["policy:kill_switch", "policy:shadow_mode", "policy:not_in_canary"]


def test_protected_region_crossing_crop_is_a_hard_failure() -> None:
    from app.pipeline.perspective import crop_and_correct_photo_with_metadata

    image = np.full((100, 100, 3), 180, dtype=np.uint8)
    with pytest.raises(ValueError, match="protected_region_cut"):
        crop_and_correct_photo_with_metadata(
            image,
            {"x1": 0.2, "y1": 0.2, "x2": 0.8, "y2": 0.8},
            auto_orient=False,
            protected_regions=[{"type": "face", "box": {"x1": 0.1, "y1": 0.4, "x2": 0.3, "y2": 0.6}}],
        )


def test_perspective_metadata_contains_stability_and_protection_gates() -> None:
    from app.pipeline.perspective import crop_and_correct_photo_with_metadata

    image = np.full((100, 100, 3), 180, dtype=np.uint8)
    _, metadata = crop_and_correct_photo_with_metadata(
        image,
        {"x1": 0.2, "y1": 0.2, "x2": 0.8, "y2": 0.8},
        auto_orient=False,
        protected_regions=[],
    )
    assert metadata["quad_validation"]["reprojection_error"] <= 0.05
    assert metadata["protected_regions"]["passed"] is True


def test_canonical_load_image_applies_exif_orientation(tmp_path) -> None:
    from PIL import Image
    from app.pipeline.image_ops import load_image

    source = Image.fromarray(np.full((10, 20, 3), 120, dtype=np.uint8), "RGB")
    exif = Image.Exif()
    exif[274] = 6
    path = tmp_path / "oriented.jpg"
    source.save(path, exif=exif)
    image = load_image(path)
    assert image.shape[:2] == (20, 10)
