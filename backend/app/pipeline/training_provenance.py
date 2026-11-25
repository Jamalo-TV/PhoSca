"""Reviewed source inventory for registered detector checkpoints."""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from hashlib import sha256
from pathlib import Path

from app.pipeline.confidence import is_sha256
from app.pipeline.model_registry import load_model_manifest

TRAINING_PROVENANCE_SCHEMA = 1
FINAL_SPLIT_ROLES = ("test", "locked")


class TrainingProvenanceError(ValueError):
    pass


@dataclass(frozen=True)
class TrainingProvenance:
    manifest_sha256: dict[str, str]
    source_images: dict[str, str]


def _text(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def source_index(images: object) -> dict[str, str]:
    if not isinstance(images, dict) or not images or not all(
        is_sha256(digest) and _text(group) for digest, group in images.items()
    ):
        raise TrainingProvenanceError("Source inventory needs page SHA-256 values and physical album IDs.")
    return {digest: group.strip().casefold() for digest, group in images.items()}


def disjoint_sources(first: dict[str, str], second: dict[str, str]) -> bool:
    return first.keys().isdisjoint(second) and set(first.values()).isdisjoint(second.values())


@lru_cache(maxsize=8)
def _read_inventory(path: Path, modified_ns: int, size: int) -> tuple[object, str]:
    payload = path.read_bytes()
    return json.loads(payload), sha256(payload).hexdigest()


def _reviewed_inventory(payload: object, model_digest: str) -> dict[str, str]:
    if (not isinstance(payload, dict) or type(payload.get("schema_version")) is not int
            or payload["schema_version"] != TRAINING_PROVENANCE_SCHEMA):
        raise TrainingProvenanceError("Unknown training provenance schema.")
    review = payload.get("review")
    if (payload.get("model_sha256") != model_digest or not _text(payload.get("model_origin"))
            or not isinstance(review, dict) or not _text(review.get("reviewer")) or not _text(review.get("evidence"))):
        raise TrainingProvenanceError("Training provenance needs the exact checkpoint, origin and source review.")
    sources = payload.get("sources")
    if not isinstance(sources, list) or not all(
        isinstance(row, dict) and is_sha256(row.get("image_sha256"))
        and _text(row.get("source_group")) and _text(row.get("source"))
        and row.get("split") in ("train", "validation", "selection", "development")
        for row in sources
    ) or not any(row["split"] == "train" for row in sources):
        raise TrainingProvenanceError("Training provenance needs reviewed sources with hashes, album IDs and splits.")
    images = source_index({row["image_sha256"]: row["source_group"] for row in sources})
    if len(images) != len(sources):
        raise TrainingProvenanceError("Training provenance lists a source image more than once.")
    return images


def load_training_provenance(registry_path: Path, model_hashes: dict[str, str]) -> TrainingProvenance:
    """Load the pinned reviews and combine exposures across the active detector models."""
    registry = load_model_manifest(registry_path)
    inventory_digests = {}
    images = {}
    for name, model_digest in model_hashes.items():
        record = registry.get(name)
        descriptor = record.get("training_provenance") if isinstance(record, dict) else None
        if (not is_sha256(model_digest) or not isinstance(descriptor, dict)
                or str(record.get("sha256")).lower() != model_digest or not is_sha256(descriptor.get("sha256"))
                or not _text(descriptor.get("path"))):
            raise TrainingProvenanceError(f"Reviewed training provenance is missing for {name}.")
        inventory_path = Path(descriptor["path"])
        if inventory_path.is_absolute() or ".." in inventory_path.parts:
            raise TrainingProvenanceError(f"Training provenance for {name} must be relative to the model registry.")
        inventory_path = registry_path.parent / inventory_path
        stat = inventory_path.stat()
        payload, digest = _read_inventory(inventory_path, stat.st_mtime_ns, stat.st_size)
        if digest != descriptor["sha256"]:
            raise TrainingProvenanceError(f"Training provenance hash mismatch for {name}.")
        for image_digest, group in _reviewed_inventory(payload, model_digest).items():
            if images.setdefault(image_digest, group) != group:
                raise TrainingProvenanceError("Detector training inventories disagree about a source's album ID.")
        inventory_digests[name] = digest
    return TrainingProvenance(inventory_digests, source_index(images))
