from __future__ import annotations


def test_low_geometry_confidence_is_not_hidden_by_detection_score() -> None:
    from app.pipeline.decisions import component_review_reasons, confidence_components

    components = confidence_components(
        detection_confidence=0.99,
        boundary_confidence=0.94,
        geometry_confidence=0.31,
        perspective_confidence=0.42,
        orientation_confidence=0.91,
        enhancement_safety_confidence=0.96,
    )

    assert components.overall_release_confidence is not None
    assert components.overall_release_confidence <= 0.43
    assert "QUADRILATERAL_UNSTABLE" in component_review_reasons(components)


def test_missing_optional_components_do_not_become_fake_zeroes() -> None:
    from app.pipeline.decisions import confidence_components

    components = confidence_components(detection_confidence=0.88, geometry_confidence=0.84)

    assert components.orientation_confidence is None
    assert components.border_trim_confidence is None
    assert components.overall_release_confidence is not None
    assert 0.84 <= components.overall_release_confidence <= 0.88
