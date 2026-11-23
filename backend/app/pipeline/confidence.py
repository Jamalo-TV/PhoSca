"""Versioned, fail-closed segmentation confidence calibration."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np


CALIBRATION_SCHEMA = 2
CALIBRATION_INPUT_SCHEMA = 1


class CalibrationError(ValueError):
    pass


def is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= set("0123456789abcdef")


@lru_cache(maxsize=4)
def _load(path: Path, modified_ns: int, size: int) -> dict[str, Any]:
    artifact = json.loads(path.read_text(encoding="utf-8"))
    source_groups = artifact.get("source_groups")
    source_images = artifact.get("source_images")
    detector = artifact.get("detector_profile")
    if (
        artifact.get("schema_version") != CALIBRATION_SCHEMA or not artifact.get("independent_holdout_id")
        or not is_sha256(artifact.get("input_sha256")) or not is_sha256(artifact.get("dataset_manifest_sha256"))
        or not isinstance(source_groups, list) or not source_groups
        or not all(isinstance(group, str) and group.strip() for group in source_groups)
        or not isinstance(source_images, dict) or not source_images
        or not all(is_sha256(digest) and isinstance(group, str) and group.strip() for digest, group in source_images.items())
        or {group.strip().casefold() for group in source_images.values()}
        != {group.strip().casefold() for group in source_groups}
        or not isinstance(detector, dict) or not is_sha256(detector.get("model_sha256"))
        or type(artifact.get("sample_count")) is not int or artifact["sample_count"] < 20
    ):
        raise CalibrationError("Calibration artifact needs versioned independent calibration evidence.")
    strata = artifact.get("strata")
    if not isinstance(strata, dict) or not strata:
        raise CalibrationError("Calibration artifact needs at least one fitted detector source.")
    for name, record in strata.items():
        x, y = record.get("raw", []), record.get("calibrated", [])
        if (
            not isinstance(x, list) or not isinstance(y, list) or len(x) < 2 or len(x) != len(y)
            or not all(isinstance(value, (int, float)) and 0.0 <= value <= 1.0 for value in x + y)
            or any(left >= right for left, right in zip(x, x[1:]))
            or any(left > right for left, right in zip(y, y[1:]))
        ):
            raise CalibrationError(f"Invalid monotonic calibration points for {name}.")
    return artifact


def load_confidence_calibration(path: Path, detector_profile: dict | None) -> dict[str, Any]:
    stat = path.stat()
    artifact = _load(path, stat.st_mtime_ns, stat.st_size)
    if detector_profile is None or artifact.get("detector_profile") != detector_profile:
        raise CalibrationError("Calibration artifact belongs to a different detector configuration.")
    return artifact


def calibrate_confidence(
    raw: float, source: str, path: Path, *, detector_profile: dict | None = None,
) -> tuple[float, dict[str, Any]]:
    if not path.is_file():
        return float(raw), {"calibrated": False, "reason": "artifact_missing", "raw": float(raw)}
    try:
        artifact = load_confidence_calibration(path, detector_profile)
        record = artifact["strata"].get(source)
        if record is None:
            return float(raw), {"calibrated": False, "reason": "detector_source_unfitted", "raw": float(raw)}
        value = float(np.interp(float(raw), record["raw"], record["calibrated"]))
        return value, {
            "calibrated": True,
            "raw": float(raw),
            "value": value,
            "stratum": source,
            "independent_holdout_id": artifact["independent_holdout_id"],
            "model_sha256": artifact["detector_profile"]["model_sha256"],
        }
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        return float(raw), {"calibrated": False, "reason": f"artifact_invalid:{type(exc).__name__}", "raw": float(raw)}
