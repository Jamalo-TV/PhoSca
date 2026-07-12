"""Versioned, fail-closed segmentation confidence calibration."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np


class CalibrationError(ValueError):
    pass


@lru_cache(maxsize=4)
def _load(path: str, modified_ns: int) -> dict[str, Any]:
    artifact = json.loads(Path(path).read_text(encoding="utf-8"))
    if artifact.get("schema_version") != 1 or not artifact.get("independent_holdout_id"):
        raise CalibrationError("Calibration artifact needs schema_version=1 and independent_holdout_id.")
    strata = artifact.get("strata")
    if not isinstance(strata, dict) or "default" not in strata:
        raise CalibrationError("Calibration artifact needs a default stratum.")
    for name, record in strata.items():
        x, y = record.get("raw", []), record.get("calibrated", [])
        if len(x) < 2 or len(x) != len(y) or list(x) != sorted(x) or list(y) != sorted(y):
            raise CalibrationError(f"Invalid monotonic calibration points for {name}.")
    return artifact


def calibrate_confidence(raw: float, source: str, path: Path) -> tuple[float, dict[str, Any]]:
    if not path.is_file():
        return float(raw), {"calibrated": False, "reason": "artifact_missing", "raw": float(raw)}
    try:
        artifact = _load(str(path), path.stat().st_mtime_ns)
        record = artifact["strata"].get(source, artifact["strata"]["default"])
        value = float(np.interp(float(raw), record["raw"], record["calibrated"]))
        return value, {
            "calibrated": True,
            "raw": float(raw),
            "value": value,
            "stratum": source if source in artifact["strata"] else "default",
            "independent_holdout_id": artifact["independent_holdout_id"],
        }
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        return float(raw), {"calibrated": False, "reason": f"artifact_invalid:{type(exc).__name__}", "raw": float(raw)}
