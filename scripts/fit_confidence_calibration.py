"""Fit monotonic confidence mappings from independently labelled predictions."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def isotonic_points(rows: list[dict]) -> tuple[list[float], list[float]]:
    ordered = sorted((float(row["score"]), int(bool(row["correct"]))) for row in rows)
    if len(ordered) < 20:
        raise ValueError("At least 20 labelled predictions are required per fitted stratum.")
    blocks = [{"scores": [score], "sum": label, "count": 1} for score, label in ordered]
    index = 0
    while index < len(blocks) - 1:
        if blocks[index]["sum"] / blocks[index]["count"] <= blocks[index + 1]["sum"] / blocks[index + 1]["count"]:
            index += 1
            continue
        blocks[index]["scores"].extend(blocks[index + 1]["scores"])
        blocks[index]["sum"] += blocks[index + 1]["sum"]
        blocks[index]["count"] += blocks[index + 1]["count"]
        blocks.pop(index + 1)
        index = max(0, index - 1)
    raw, calibrated = [0.0], [blocks[0]["sum"] / blocks[0]["count"]]
    for block in blocks:
        raw.append(sum(block["scores"]) / len(block["scores"]))
        calibrated.append(block["sum"] / block["count"])
    raw.append(1.0)
    calibrated.append(calibrated[-1])
    return raw, calibrated


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path, help="JSON list with score, correct, and optional source fields.")
    parser.add_argument("--holdout-id", required=True)
    parser.add_argument("--output", type=Path, default=Path("models/segmentation-calibration.json"))
    args = parser.parse_args()
    rows = json.loads(args.input.read_text(encoding="utf-8"))
    strata = {"default": rows}
    for source in sorted({str(row.get("source", "default")) for row in rows}):
        subset = [row for row in rows if str(row.get("source", "default")) == source]
        if source != "default" and len(subset) >= 20:
            strata[source] = subset
    artifact = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "independent_holdout_id": args.holdout_id,
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
