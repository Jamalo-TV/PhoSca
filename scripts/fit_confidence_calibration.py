"""Fit monotonic confidence mappings from independently labelled predictions."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from hashlib import sha256
from itertools import groupby
from pathlib import Path


def isotonic_points(rows: list[dict]) -> tuple[list[float], list[float]]:
    ordered = sorted(((float(row["score"]), row["correct"]) for row in rows), key=lambda item: item[0])
    if len(ordered) < 20:
        raise ValueError("At least 20 labelled predictions are required per fitted stratum.")
    if not all(0.0 <= score <= 1.0 and label in (True, False) for score, label in ordered):
        raise ValueError("Calibration rows require a score in [0, 1] and a binary correctness label.")
    blocks = []
    for score, group in groupby(ordered, key=lambda item: item[0]):
        labels = [int(label) for _, label in group]
        blocks.append({"first": score, "last": score, "sum": sum(labels), "count": len(labels)})
    index = 0
    while index < len(blocks) - 1:
        if blocks[index]["sum"] / blocks[index]["count"] <= blocks[index + 1]["sum"] / blocks[index + 1]["count"]:
            index += 1
            continue
        blocks[index]["last"] = blocks[index + 1]["last"]
        blocks[index]["sum"] += blocks[index + 1]["sum"]
        blocks[index]["count"] += blocks[index + 1]["count"]
        blocks.pop(index + 1)
        index = max(0, index - 1)
    raw, calibrated = [], []
    for block in blocks:
        value = block["sum"] / block["count"]
        raw.append(block["first"])
        calibrated.append(value)
        if block["last"] > block["first"]:
            raw.append(block["last"])
            calibrated.append(value)
    if raw[0] > 0:
        raw.insert(0, 0.0)
        calibrated.insert(0, calibrated[0])
    if raw[-1] < 1:
        raw.append(1.0)
        calibrated.append(calibrated[-1])
    return raw, calibrated


def main() -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
    from app.pipeline.confidence import CALIBRATION_INPUT_SCHEMA, CALIBRATION_SCHEMA, is_sha256

    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path, help="Versioned scores from evaluate_onnx_segmentation.py --calibration-output.")
    parser.add_argument("--holdout-id", required=True)
    parser.add_argument("--output", type=Path, default=Path("models/segmentation-calibration.json"))
    args = parser.parse_args()
    if args.input.resolve() == args.output.resolve() or not args.holdout_id.strip():
        parser.error("Use a separate artifact path and a nonempty holdout ID.")
    contents = args.input.read_bytes()
    data = json.loads(contents)
    if not isinstance(data, dict) or data.get("schema_version") != CALIBRATION_INPUT_SCHEMA or data.get("role") != "calibration":
        parser.error("Use a calibration export from an independently reviewed album split.")
    detector_profile = data.get("detector_profile")
    source_images = data.get("source_images")
    manifest_hash = data.get("dataset_manifest_sha256")
    rows = data.get("rows")
    if (
        not isinstance(detector_profile, dict) or not is_sha256(detector_profile.get("model_sha256"))
        or not is_sha256(manifest_hash) or not isinstance(source_images, dict) or not source_images
        or not all(is_sha256(digest) and isinstance(group, str) and group.strip() for digest, group in source_images.items())
        or not isinstance(rows, list)
    ):
        parser.error("Calibration export lacks the detector profile or album provenance.")
    indexes = set()
    for row in rows:
        if (
            not isinstance(row, dict) or not is_sha256(row.get("image_sha256"))
            or type(row.get("prediction_index")) is not int or row["prediction_index"] < 0
            or not isinstance(row.get("source_group"), str)
            or row["source_group"] != source_images.get(row["image_sha256"])
            or not isinstance(row.get("source"), str) or not row["source"].strip()
            or type(row.get("correct")) is not bool or type(row.get("score")) not in (int, float)
            or not 0 <= row["score"] <= 1
        ):
            parser.error("Calibration rows require unique prediction IDs, album provenance, scores and binary labels.")
        indexes.add((row["image_sha256"], row["prediction_index"]))
    if len(indexes) != len(rows):
        parser.error("Calibration export contains duplicate predictions.")
    by_source = {}
    for row in rows:
        by_source.setdefault(row["source"], []).append(row)
    strata = {source: subset for source, subset in by_source.items()
              if len(subset) >= 20 and {row["correct"] for row in subset} == {True, False}}
    if not strata:
        parser.error("At least one detector source needs 20 predictions with both correct and incorrect examples.")
    artifact = {
        "schema_version": CALIBRATION_SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "independent_holdout_id": args.holdout_id,
        "source_groups": sorted(set(source_images.values())),
        "source_images": source_images,
        "dataset_manifest_sha256": manifest_hash,
        "detector_profile": detector_profile,
        "input_sha256": sha256(contents).hexdigest(),
        "sample_count": len(rows),
        "strata": {},
    }
    for name, subset in strata.items():
        raw, calibrated = isotonic_points(subset)
        artifact["strata"][name] = {"raw": raw, "calibrated": calibrated, "sample_count": len(subset)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
