from __future__ import annotations

import hashlib
import json

import pytest


def test_model_registry_accepts_registered_hash(tmp_path) -> None:
    from app.pipeline.model_registry import verify_model_artifact

    model = tmp_path / "model.onnx"
    model.write_bytes(b"known-model")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"models": {"model.onnx": {"sha256": hashlib.sha256(b"known-model").hexdigest(), "license": "Apache-2.0"}}}))

    record = verify_model_artifact(model, manifest)
    assert record["id"] == "model.onnx"


def test_model_registry_rejects_tampered_or_unlicensed_artifact(tmp_path) -> None:
    from app.pipeline.model_registry import ModelArtifactError, verify_model_artifact

    model = tmp_path / "model.onnx"
    model.write_bytes(b"tampered")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"models": {"model.onnx": {"sha256": "0" * 64, "license": ""}}}))
    with pytest.raises(ModelArtifactError):
        verify_model_artifact(model, manifest)
