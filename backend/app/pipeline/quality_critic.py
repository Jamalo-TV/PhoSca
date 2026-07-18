"""Independent deterministic quality critic for derived photo candidates."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from app.pipeline.enhancement_types import normalize_bgr_uint8
from app.pipeline.quality_assessment import QualityAssessor


def _gate(name: str, passed: bool, details: dict[str, Any]) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "details": details}


def evaluate_candidate(
    source: np.ndarray | None, candidate: np.ndarray, metadata: dict | None = None
) -> list[dict[str, Any]]:
    """Return auditable veto gates without selecting or generating an image."""
    metadata = metadata or {}
    decoded = bool(
        isinstance(candidate, np.ndarray)
        and candidate.size
        and candidate.ndim in {2, 3}
        and np.issubdtype(candidate.dtype, np.number)
        and np.isfinite(candidate).all()
    )
    gates: list[dict[str, Any]] = [
        _gate(
            "decoded_pixels",
            decoded,
            {
                "shape": [int(value) for value in getattr(candidate, "shape", ())],
                "dtype": str(getattr(candidate, "dtype", "unknown")),
            },
        )
    ]
    nested = metadata.get("ai_metadata") or {}
    stage_failed = (
        metadata.get("method") == "failed"
        or bool(metadata.get("error"))
        or nested.get("method") == "failed"
        or bool(nested.get("error"))
    )
    gates.append(
        _gate(
            "stage_integrity",
            not stage_failed,
            {"method": metadata.get("method"), "nested_method": nested.get("method")},
        )
    )
    kind = str(metadata.get("variant_kind") or "")
    generative = kind in {"restoration_candidate", "face_restoration_candidate"} or bool(metadata.get("generative"))
    gates.append(
        _gate(
            "generative_release_policy",
            not (generative and metadata.get("auto_release") is True),
            {"variant_kind": kind, "auto_release": metadata.get("auto_release", False)},
        )
    )
    if not decoded or source is None or not isinstance(source, np.ndarray) or source.size == 0:
        return gates

    try:
        source_bgr, _ = normalize_bgr_uint8(source)
        candidate_bgr, _ = normalize_bgr_uint8(candidate)
    except ValueError as exc:
        gates.append(_gate("working_image_contract", False, {"error": str(exc)}))
        return gates
    gates.append(
        _gate("working_image_contract", True, {"color_space": "sRGB", "channel_order": "BGR", "dtype": "uint8"})
    )

    source_h, source_w = source_bgr.shape[:2]
    candidate_h, candidate_w = candidate_bgr.shape[:2]
    scale_x = candidate_w / max(source_w, 1)
    scale_y = candidate_h / max(source_h, 1)
    aspect_error = abs((candidate_w / max(candidate_h, 1)) / max(source_w / max(source_h, 1), 1e-6) - 1.0)
    scale_anisotropy = abs(scale_x / max(scale_y, 1e-6) - 1.0)
    gates.append(
        _gate(
            "dimension_contract",
            0.5 <= scale_x <= 4.0 and 0.5 <= scale_y <= 4.0 and aspect_error <= 0.005 and scale_anisotropy <= 0.01,
            {
                "source": [source_w, source_h],
                "candidate": [candidate_w, candidate_h],
                "scale": [scale_x, scale_y],
                "aspect_error": aspect_error,
            },
        )
    )

    comparable = candidate_bgr
    if comparable.shape[:2] != source_bgr.shape[:2]:
        comparable = cv2.resize(comparable, (source_w, source_h), interpolation=cv2.INTER_AREA)
    assessor = QualityAssessor()
    delta = assessor.compare(source_bgr, candidate_bgr)
    gates.extend(
        [
            _gate("quality_regression", not assessor.should_revert(delta, min_quality_delta=-0.025), delta.as_dict()),
            _gate(
                "clipping_regression",
                delta.clipping_delta <= 0.02
                and max(delta.channel_shadow_delta + delta.channel_highlight_delta, default=0.0) <= 0.03,
                {
                    "overall_delta": delta.clipping_delta,
                    "shadow_delta": delta.channel_shadow_delta,
                    "highlight_delta": delta.channel_highlight_delta,
                },
            ),
            _gate("color_fidelity", delta.color_delta <= 10.0, {"mean_lab_delta": delta.color_delta}),
            _gate(
                "saturation_fidelity",
                abs(delta.saturation_delta) <= 0.08,
                {"mean_saturation_delta": delta.saturation_delta},
            ),
            _gate(
                "local_structure",
                delta.local_structure_min >= 0.76 and delta.local_structure_mean >= 0.90,
                {"minimum": delta.local_structure_min, "mean": delta.local_structure_mean},
            ),
            _gate("texture_preservation", 0.68 <= delta.texture_ratio <= 1.65, {"ratio": delta.texture_ratio}),
            _gate("halo_overshoot", delta.halo_delta <= 0.025, {"normalized_overshoot": delta.halo_delta}),
        ]
    )
    mean_abs_delta = float(np.mean(np.abs(source_bgr.astype(np.float32) - comparable.astype(np.float32))))
    tile_h = max(16, source_h // 4)
    tile_w = max(16, source_w // 4)
    local_deltas = [
        float(
            np.mean(
                np.abs(
                    source_bgr[y : min(source_h, y + tile_h), x : min(source_w, x + tile_w)].astype(np.float32)
                    - comparable[y : min(source_h, y + tile_h), x : min(source_w, x + tile_w)].astype(np.float32)
                )
            )
        )
        for y in range(0, source_h, tile_h)
        for x in range(0, source_w, tile_w)
    ]
    gates.append(
        _gate(
            "source_delta",
            mean_abs_delta <= 48.0 and max(local_deltas, default=0.0) <= 72.0,
            {
                "mean_absolute_delta": mean_abs_delta,
                "max_local_mean_delta": max(local_deltas, default=0.0),
                "limits": [48.0, 72.0],
            },
        )
    )
    return gates
