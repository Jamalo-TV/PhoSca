import json

from app.pipeline.confidence import calibrate_confidence


def test_missing_calibration_fails_closed(tmp_path):
    value, evidence = calibrate_confidence(0.91, "yolo_onnx", tmp_path / "missing.json")
    assert value == 0.91
    assert evidence == {"calibrated": False, "reason": "artifact_missing", "raw": 0.91}


def test_versioned_calibration_interpolates_and_records_holdout(tmp_path):
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps({
        "schema_version": 1,
        "independent_holdout_id": "safety-2026-q3",
        "strata": {"default": {"raw": [0.0, 0.5, 1.0], "calibrated": [0.0, 0.4, 0.9]}},
    }), encoding="utf-8")
    value, evidence = calibrate_confidence(0.75, "unknown", path)
    assert value == 0.65
    assert evidence["calibrated"] is True
    assert evidence["independent_holdout_id"] == "safety-2026-q3"


def test_confidence_calibration_pools_tied_scores_and_rejects_duplicate_knots(tmp_path):
    from scripts.fit_confidence_calibration import isotonic_points

    scores = [{"score": 0.9, "correct": i < 10} for i in range(20)]
    raw, calibrated = isotonic_points(scores)
    assert raw == [0.0, 0.9, 1.0]
    assert calibrated == [0.5, 0.5, 0.5]
    path = tmp_path / "calibration.json"
    artifact = {"schema_version": 1, "independent_holdout_id": "tie-test", "strata": {"default": {
        "raw": raw, "calibrated": calibrated,
    }}}
    path.write_text(json.dumps(artifact), encoding="utf-8")
    value, evidence = calibrate_confidence(0.9, "default", path)
    assert evidence["calibrated"] is True and value == 0.5

    artifact["strata"]["default"] = {"raw": [0, 0.9, 0.9, 1], "calibrated": [0, 0, 1, 1]}
    duplicate_path = tmp_path / "duplicate.json"
    duplicate_path.write_text(json.dumps(artifact), encoding="utf-8")
    _, evidence = calibrate_confidence(0.9, "default", duplicate_path)
    assert evidence["calibrated"] is False
