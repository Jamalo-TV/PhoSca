from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest


def _quad(source: str = "manual_quad") -> dict:
    return {
        "source": source,
        "polygon": [
            {"x": 0.1, "y": 0.1},
            {"x": 0.8, "y": 0.1},
            {"x": 0.8, "y": 0.8},
            {"x": 0.1, "y": 0.8},
        ],
    }


def test_manifest_keeps_prediction_correction_events_and_provenance(tmp_path: Path) -> None:
    from scripts.export_reviewed_dataset import build_manifest_records

    original = {"bounding_box": {"x1": 0.08, "y1": 0.09, "x2": 0.81, "y2": 0.82}, "segmentation_mask": _quad("model")}
    pages = [
        {
            "page_id": "page-1",
            "album_id": "album-1",
            "original_filename": "same-name.jpg",
            "storage_path": str(tmp_path / "missing-source.jpg"),
            "file_hash_sha256": "a" * 64,
            "page_status": "completed",
            "processing_metadata": {"dataset": {"source_group": "scanner-a"}},
        }
    ]
    photos = [
        {
            "photo_id": "photo-1",
            "page_id": "page-1",
            "bounding_box": {"x1": 0.1, "y1": 0.1, "x2": 0.8, "y2": 0.8},
            "segmentation_mask": _quad(),
            "segmentation_confidence": None,
            "aspect_ratio": 1.0,
            "geometry_valid": True,
            "photo_status": "completed",
            "enhancement_applied": {
                "review_provenance": {"original_prediction": original, "prediction_provenance": {"run_id": "run-1"}},
                "correction": {"orientation": {"rotation_degrees": 90}},
                "variants": {"source_corrected": {"metrics": {"blur_score": 42.0}}},
            },
        }
    ]
    events = [
        {
            "event_id": "event-1",
            "task_type": "geometry",
            "task_id": "photo-1",
            "answer": "correct",
            "reason_code": None,
            "reviewer": "reviewer-a",
            "created_at": "2026-07-18T10:00:00Z",
            "event_metadata": {"schema_version": 2, "snapshot": {"photos": [{"original_prediction": original}]}},
        }
    ]
    runs = [
        {
            "run_id": "run-1",
            "page_id": "page-1",
            "input_sha256": "a" * 64,
            "pipeline_version": "v2",
            "config_fingerprint": "b" * 64,
            "run_status": "completed",
            "run_manifest": {"model": "seg-v2"},
            "created_at": "2026-07-18T09:00:00Z",
        }
    ]

    records = build_manifest_records(pages, photos, events, runs)
    assert len(records) == 1
    photo = records[0]["photos"][0]
    assert photo["state"] == "accepted"
    assert photo["original_prediction"] == original
    assert photo["current_geometry"]["segmentation_mask"]["source"] == "manual_quad"
    assert photo["prediction_provenance"]["run_id"] == "run-1"
    assert photo["review_events"][0]["metadata"]["snapshot"]
    assert records[0]["source_group"] == "scanner-a"


def test_group_split_is_atomic_and_refuses_cross_group_duplicate_hashes() -> None:
    from scripts.split_reviewed_dataset import split_records

    records: list[dict[str, Any]] = [
        {
            "page_id": f"{album}-{index}",
            "album_id": album,
            "source_group": source,
            "image": {"sha256": f"{album}-{index}"},
        }
        for album, source in (("a", "scanner-a"), ("b", "scanner-b"), ("c", "scanner-c"))
        for index in range(2)
    ]
    splits, _ = split_records(records, seed=7)
    locations = {record["album_id"]: name for name, values in splits.items() for record in values}
    assert len(locations) == 3
    for album in ("a", "b", "c"):
        assert all(
            record["album_id"] != album
            for name, values in splits.items()
            if name != locations[album]
            for record in values
        )

    records[-1]["image"]["sha256"] = records[0]["image"]["sha256"]
    with pytest.raises(ValueError, match="different album/source groups"):
        split_records(records, seed=7)


def test_legacy_yolo_requires_approval_and_rejects_invalid_coordinates(tmp_path: Path) -> None:
    from scripts.export_reviewed_yolo_labels import export_rows

    unreviewed = [
        {
            "page_id": "page-1",
            "original_filename": "page.jpg",
            "photo_id": "photo-1",
            "segmentation_mask": _quad(),
            "bounding_box": None,
        }
    ]
    report = export_rows(unreviewed, album_id="album", output_dir=tmp_path / "unreviewed")
    assert report.label_files_written == 0
    assert report.skipped_photos == ["photo-1: geometry is not reviewer-approved"]

    invalid = [
        {
            **unreviewed[0],
            "geometry_review_answer": "correct",
            "segmentation_mask": {**_quad(), "polygon": [{"x": -0.1, "y": 0.1}, *_quad()["polygon"][1:]]},
        }
    ]
    with pytest.raises(ValueError, match="normalized to 0..1"):
        export_rows(invalid, album_id="album", output_dir=tmp_path / "invalid")

    collision = [
        {**unreviewed[0], "page_id": "page-1", "geometry_review_answer": "correct"},
        {**unreviewed[0], "page_id": "page-2", "photo_id": "photo-2", "geometry_review_answer": "correct"},
    ]
    with pytest.raises(ValueError, match="duplicate original filenames"):
        export_rows(collision, album_id="album", output_dir=tmp_path / "collision")
