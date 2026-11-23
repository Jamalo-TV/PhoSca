import json

from app.pipeline.confidence import CALIBRATION_SCHEMA, calibrate_confidence


DETECTOR = {"model_sha256": "a" * 64, "proposal_threshold": .25}


def _artifact(raw, calibrated):
    return {
        "schema_version": CALIBRATION_SCHEMA,
        "independent_holdout_id": "safety-2026-q3",
        "source_groups": ["album-calibration"], "sample_count": 20,
        "input_sha256": "b" * 64, "dataset_manifest_sha256": "c" * 64,
        "source_images": {"f" * 64: "album-calibration"},
        "detector_profile": DETECTOR,
        "strata": {"default": {"raw": raw, "calibrated": calibrated}},
    }


def test_missing_calibration_fails_closed(tmp_path):
    value, evidence = calibrate_confidence(0.91, "yolo_onnx", tmp_path / "missing.json")
    assert value == 0.91
    assert evidence == {"calibrated": False, "reason": "artifact_missing", "raw": 0.91}


def test_versioned_calibration_interpolates_and_records_holdout(tmp_path):
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps(_artifact([0.0, 0.5, 1.0], [0.0, 0.4, 0.9])), encoding="utf-8")
    value, evidence = calibrate_confidence(0.75, "default", path, detector_profile=DETECTOR)
    assert value == 0.65
    assert evidence["calibrated"] is True
    assert evidence["independent_holdout_id"] == "safety-2026-q3"
    assert evidence["model_sha256"] == DETECTOR["model_sha256"]
    score, evidence = calibrate_confidence(0.98, "edge_contour", path, detector_profile=DETECTOR)
    assert score == 0.98 and evidence["reason"] == "detector_source_unfitted"
    for other in (None, DETECTOR | {"model_sha256": "d" * 64}, DETECTOR | {"proposal_threshold": .15}):
        value, evidence = calibrate_confidence(0.75, "default", path, detector_profile=other)
        assert value == 0.75 and evidence["calibrated"] is False


def test_confidence_calibration_pools_tied_scores_and_rejects_duplicate_knots(tmp_path):
    from scripts.fit_confidence_calibration import isotonic_points

    scores = [{"score": 0.9, "correct": i < 10} for i in range(20)]
    raw, calibrated = isotonic_points(scores)
    assert raw == [0.0, 0.9, 1.0]
    assert calibrated == [0.5, 0.5, 0.5]
    path = tmp_path / "calibration.json"
    artifact = _artifact(raw, calibrated)
    path.write_text(json.dumps(artifact), encoding="utf-8")
    value, evidence = calibrate_confidence(0.9, "default", path, detector_profile=DETECTOR)
    assert evidence["calibrated"] is True and value == 0.5

    artifact["strata"]["default"] = {"raw": [0, 0.9, 0.9, 1], "calibrated": [0, 0, 1, 1]}
    duplicate_path = tmp_path / "duplicate.json"
    duplicate_path.write_text(json.dumps(artifact), encoding="utf-8")
    _, evidence = calibrate_confidence(0.9, "default", duplicate_path, detector_profile=DETECTOR)
    assert evidence["calibrated"] is False
