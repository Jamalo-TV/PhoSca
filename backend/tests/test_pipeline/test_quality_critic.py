from __future__ import annotations

import numpy as np


def test_quality_critic_rejects_failed_stage_and_invalid_pixels() -> None:
    from app.pipeline.quality_critic import evaluate_candidate

    image = np.full((20, 20, 3), 120, dtype=np.uint8)
    gates = evaluate_candidate(image, image, {"method": "failed", "error": "model unavailable"})
    assert next(gate for gate in gates if gate["name"] == "stage_integrity")["passed"] is False


def test_quality_critic_records_large_regression() -> None:
    from app.pipeline.quality_critic import evaluate_candidate

    source = np.tile(np.arange(80, dtype=np.uint8), (80, 1))
    source = np.dstack((source, source, source))
    degraded = np.full_like(source, 255)
    gates = evaluate_candidate(source, degraded)
    assert next(gate for gate in gates if gate["name"] == "source_delta")["passed"] is False
