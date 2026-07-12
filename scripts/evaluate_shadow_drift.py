"""Compare stable and shadow prediction exports and fail on configured drift."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def summarize(rows: list[dict]) -> dict:
    confidences = [float(value) for row in rows for value in row.get("confidences", [])]
    counts = [int(row.get("detection_count", len(row.get("confidences", [])))) for row in rows]
    return {
        "pages": len(rows),
        "mean_count": float(np.mean(counts)) if counts else 0.0,
        "empty_page_rate": float(np.mean([count == 0 for count in counts])) if counts else 0.0,
        "mean_confidence": float(np.mean(confidences)) if confidences else 0.0,
        "review_rate": float(np.mean([bool(row.get("review_needed")) for row in rows])) if rows else 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("stable", type=Path)
    parser.add_argument("shadow", type=Path)
    parser.add_argument("--max-empty-rate-increase", type=float, default=0.01)
    parser.add_argument("--max-review-rate-increase", type=float, default=0.03)
    parser.add_argument("--max-mean-count-delta", type=float, default=0.25)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    stable_rows = json.loads(args.stable.read_text(encoding="utf-8"))
    shadow_rows = json.loads(args.shadow.read_text(encoding="utf-8"))
    stable_ids = {str(row["page_id"]) for row in stable_rows}
    shadow_ids = {str(row["page_id"]) for row in shadow_rows}
    if stable_ids != shadow_ids:
        raise ValueError("Stable and shadow exports must contain identical page IDs.")
    stable, shadow = summarize(stable_rows), summarize(shadow_rows)
    deltas = {key: shadow[key] - stable[key] for key in ("mean_count", "empty_page_rate", "mean_confidence", "review_rate")}
    gates = {
        "empty_page_rate": deltas["empty_page_rate"] <= args.max_empty_rate_increase,
        "review_rate": deltas["review_rate"] <= args.max_review_rate_increase,
        "mean_count": abs(deltas["mean_count"]) <= args.max_mean_count_delta,
    }
    report = {"schema_version": 1, "passed": all(gates.values()), "stable": stable, "shadow": shadow, "deltas": deltas, "gates": gates}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
