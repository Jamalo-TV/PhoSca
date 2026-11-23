"""Fit monotonic confidence mappings from independently labelled predictions."""
from __future__ import annotations

import argparse
import json
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
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path, help="JSON list with score, correct, source_group, and optional source fields.")
    parser.add_argument("--holdout-id", required=True)
    parser.add_argument("--output", type=Path, default=Path("models/segmentation-calibration.json"))
    args = parser.parse_args()
    contents = args.input.read_bytes()
    rows = json.loads(contents)
    source_groups = {str(row.get("source_group") or "").strip().casefold() for row in rows}
    strata = {"default": rows}
    for source in sorted({str(row.get("source", "default")) for row in rows}):
        subset = [row for row in rows if str(row.get("source", "default")) == source]
        if source != "default" and len(subset) >= 20:
            strata[source] = subset
    artifact = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "independent_holdout_id": args.holdout_id,
        "source_groups": sorted(source_groups) if "" not in source_groups else [],
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
