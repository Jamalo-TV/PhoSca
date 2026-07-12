"""Independent deterministic quality critic for derived photo candidates."""
from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from app.pipeline.quality_assessment import QualityAssessor


def evaluate_candidate(source: np.ndarray | None, candidate: np.ndarray, metadata: dict | None = None) -> list[dict[str, Any]]:
    """Return auditable veto gates without selecting or generating an image.

    This is intentionally independent from restoration stages.  It is a
    conservative baseline for later calibrated semantic/identity critics, not
    a claim of perceptual ground truth.
    """
    metadata = metadata or {}
    gates: list[dict[str, Any]] = []
    gates.append({
        "name": "decoded_pixels",
        "passed": bool(candidate.size and candidate.ndim == 3 and candidate.shape[2] in {3, 4} and np.isfinite(candidate).all()),
        "details": {"shape": [int(value) for value in candidate.shape]},
    })
    stage_failed = metadata.get("method") == "failed" or bool(metadata.get("error"))
    gates.append({"name": "stage_integrity", "passed": not stage_failed, "details": {"method": metadata.get("method")}})
    if source is None or source.size == 0 or candidate.size == 0:
        return gates

    comparable = candidate
    if comparable.shape[:2] != source.shape[:2]:
        comparable = cv2.resize(comparable, (source.shape[1], source.shape[0]), interpolation=cv2.INTER_AREA)
    delta = QualityAssessor().compare(source, comparable)
    gates.extend(
        [
            {
                "name": "quality_regression",
                "passed": not QualityAssessor().should_revert(delta, min_quality_delta=-0.03),
                "details": delta.as_dict(),
            },
            {
                "name": "source_delta",
                "passed": float(np.mean(np.abs(source.astype(np.float32) - comparable.astype(np.float32)))) <= 96.0,
                "details": {"max_mean_absolute_delta": 96.0},
            },
        ]
    )
    return gates
