"""Compare two evaluate_onnx_segmentation reports on the same references."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import mean


DETECTION_METRICS = (
    "detection_precision", "detection_recall", "polygon_iou", "mask_iou", "boundary_f_score",
)
MODE_METRICS = (
    "footprint_iou", "corner_error_max_px", "aspect_ratio_relative_error",
    "landmark_error_p95_px", "content_cutoff", "foreign_fraction", "foreign_source_samples",
    "cutoff_beyond_1px", "foreign_beyond_1px", "outside_source_samples", "sampling_shortfall_px",
    "warp_metadata_error_max_px", "orientation_matches_reference",
)
MODE_STATE = ("status", "passes", "automatic_release_passes", "reason_codes")


def reference_rows(report: dict) -> dict:
    rows = {}
    for fixture in report["fixtures"]:
        for row in fixture["crop_modes"]:
            if row["truth_index"] is None:
                continue
            key = (fixture["source_group"], fixture["fixture"], row["truth_index"], row["mode"])
            if key in rows:
                raise ValueError(f"Duplicate reference {key!r}")
            rows[key] = row
    return rows


def reference_identity(report: dict) -> tuple:
    identity = tuple(report[key] for key in ("split", "dataset_manifest_sha256", "iou_threshold"))
    fixtures = [(page["source_group"], page["fixture"], page["expected"], page["annotation_provenance"])
                for page in report["fixtures"]]
    return identity, fixtures


def compare(baseline: dict, candidate: dict) -> dict:
    if reference_identity(baseline) != reference_identity(candidate):
        raise ValueError("Reports must use the same fixture inventory, split and scoring IoU")
    previous, current = reference_rows(baseline), reference_rows(candidate)
    reference_fields = ("verified", "recoverable", "difficulty")
    if (previous.keys() != current.keys() or any(
        tuple(row.get(field) for field in reference_fields)
        != tuple(current[key].get(field) for field in reference_fields)
        for key, row in previous.items()
    )):
        raise ValueError("Per-mode reference rows differ")

    modes = {}
    for mode in baseline["crop_modes"]:
        indexed_pairs = [(key, row, current[key]) for key, row in previous.items() if key[-1] == mode]
        pairs = [(before, after) for _, before, after in indexed_pairs]
        common_geometry = {}
        for metric in MODE_METRICS:
            values = [(a[metric], b[metric]) for a, b in pairs
                      if a.get(metric) is not None and b.get(metric) is not None]
            if values:
                a, b = zip(*values, strict=True)
                common_geometry[metric] = {"samples": len(values), "baseline_mean": mean(a), "candidate_mean": mean(b)}
        changed_cases, geometry_changes, newly_held = [], [], []
        for (group, fixture, index, _), before, after in indexed_pairs:
            reference = {"source_group": group, "fixture": fixture, "truth_index": index}
            if any(before[field] != after[field] for field in MODE_STATE):
                changed_cases.append({
                    **reference,
                    "baseline": {field: before.get(field) for field in (*MODE_STATE, *MODE_METRICS)},
                    "candidate": {field: after.get(field) for field in (*MODE_STATE, *MODE_METRICS)},
                })
            metrics = {metric: {"baseline": before.get(metric), "candidate": after.get(metric)}
                       for metric in MODE_METRICS if before.get(metric) != after.get(metric)}
            if metrics:
                geometry_changes.append({**reference, "metrics": metrics})
            if before["passes"] and after["status"] == "review_needed" and after.get("geometry_passes") is True:
                newly_held.append({**reference, "verified": after["verified"], "reason_codes": after["reason_codes"]})
        modes[mode] = {
            "baseline": baseline["crop_modes"][mode],
            "candidate": candidate["crop_modes"][mode],
            "reference_status_pairs": [
                {"baseline": before, "candidate": after, "count": count}
                for (before, after), count in sorted(Counter((a["status"], b["status"]) for a, b in pairs).items())
            ],
            "common_geometry": common_geometry,
            "changed_cases": changed_cases,
            "geometry_changes": geometry_changes,
            "newly_held_passing_geometry": {"count": len(newly_held), "cases": newly_held},
        }

    def summary(report):
        return {
            "model_sha256": report["model_sha256"],
            "detection": report["metrics"]["totals"] | {name: report["metrics"].get(name) for name in DETECTION_METRICS},
            "validation_profile": report.get("validation_profile"),
            "crop_mode_availability": report["crop_mode_availability"],
            "runtime": report["runtime"],
        }

    return {"baseline": summary(baseline), "candidate": summary(candidate), "crop_modes": modes}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = compare(json.loads(args.baseline.read_text()), json.loads(args.candidate.read_text()))
    except (KeyError, ValueError) as error:
        parser.error(str(error))
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
