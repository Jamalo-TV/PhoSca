"""Local, hash-verified model artifact contract.

Runtime downloads make a visual pipeline non-reproducible.  A deployment can
opt into a model only when its local file is described by this manifest.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


class ModelArtifactError(ValueError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_model_manifest(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        raise ModelArtifactError(f"model manifest missing: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ModelArtifactError(f"invalid model manifest: {path}") from exc
    models = payload.get("models", payload)
    if not isinstance(models, dict):
        raise ModelArtifactError("model manifest must contain a models object")
    return models


def verify_model_artifact(model_path: Path, manifest_path: Path, *, model_id: str | None = None) -> dict[str, Any]:
    if not model_path.is_file():
        raise ModelArtifactError(f"model artifact missing: {model_path}")
    models = load_model_manifest(manifest_path)
    identifier = model_id or model_path.name
    record = models.get(identifier)
    if not isinstance(record, dict):
        raise ModelArtifactError(f"model artifact is not registered: {identifier}")
    expected_hash = record.get("sha256")
    if not isinstance(expected_hash, str) or len(expected_hash) != 64:
        raise ModelArtifactError(f"model artifact lacks a SHA-256: {identifier}")
    actual_hash = sha256(model_path)
    if actual_hash.lower() != expected_hash.lower():
        raise ModelArtifactError(f"model artifact hash mismatch: {identifier}")
    if not record.get("license"):
        raise ModelArtifactError(f"model artifact lacks license metadata: {identifier}")
    return {"id": identifier, "sha256": actual_hash, "license": record["license"], "input": record.get("input")}
