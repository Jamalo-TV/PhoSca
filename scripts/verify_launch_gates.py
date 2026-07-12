"""Fail closed unless every Consumer launch gate has approved, intact evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


REQUIRED_GATES = {
    "representative_data",
    "double_review",
    "independent_safety_holdout",
    "crop_safety",
    "confidence_calibration",
    "enhancement_safety",
    "human_ab",
    "security",
    "privacy_dpia",
    "model_and_data_licenses",
    "retention_and_backup_restore",
    "slo_load_chaos",
    "canary_rollback",
}


def verify(manifest_path: Path) -> dict:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    base = manifest_path.parent
    records = {str(item.get("id")): item for item in manifest.get("gates", [])}
    results = []
    for gate_id in sorted(REQUIRED_GATES):
        record = records.get(gate_id, {})
        evidence = base / str(record.get("evidence", ""))
        expected = str(record.get("sha256", "")).lower()
        exists = evidence.is_file()
        actual = hashlib.sha256(evidence.read_bytes()).hexdigest() if exists else None
        approved = record.get("status") == "approved" and bool(record.get("approved_by")) and bool(record.get("approved_at"))
        results.append({
            "id": gate_id,
            "passed": approved and exists and len(expected) == 64 and actual == expected,
            "status": record.get("status", "missing"),
            "evidence": str(evidence),
            "evidence_exists": exists,
            "hash_matches": bool(actual and actual == expected),
        })
    unknown = sorted(set(records) - REQUIRED_GATES)
    return {"passed": all(item["passed"] for item in results), "results": results, "unknown_gates": unknown}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path, nargs="?", default=Path("evidence/launch-gates.json"))
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    result = verify(args.manifest)
    payload = json.dumps(result, indent=2, sort_keys=True)
    print(payload)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(payload + "\n", encoding="utf-8")
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
