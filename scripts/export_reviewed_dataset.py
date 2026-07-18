"""Export review evidence as a provenance-rich, page-keyed JSONL manifest.

The manifest is the canonical learning interchange.  Lossy formats such as
YOLO labels should be derived from it, never used as the only review record.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path
from uuid import UUID

from PIL import Image
from sqlalchemy import create_engine, text


def _json(value: object, default):
    if value is None:
        return default
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, str):
        loaded = json.loads(value)
        return loaded
    return default


def _serial(value: object) -> object:
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, UUID):
        return str(value)
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _image_record(page: dict, *, output_dir: Path | None = None, copy_images: bool = False) -> dict:
    source = Path(str(page["storage_path"]))
    stored_sha = str(page["file_hash_sha256"])
    width = height = None
    actual_sha = None
    if source.is_file():
        actual_sha = _sha256(source)
        if actual_sha != stored_sha:
            raise ValueError(f"page {page['page_id']} image hash does not match the database")
        try:
            with Image.open(source) as image:
                width, height = image.size
        except OSError as exc:
            raise ValueError(f"page {page['page_id']} image cannot be decoded") from exc
    elif copy_images:
        raise ValueError(f"page {page['page_id']} image is missing: {source}")

    exported_path = None
    if copy_images and output_dir is not None:
        image_dir = output_dir / "images"
        image_dir.mkdir(parents=True, exist_ok=True)
        suffix = source.suffix.lower() or ".img"
        target = image_dir / f"{page['page_id']}{suffix}"
        shutil.copy2(source, target)
        exported_path = target.relative_to(output_dir).as_posix()
    return {
        "original_filename": str(page["original_filename"]),
        "storage_path": str(source),
        "exported_path": exported_path,
        "sha256": stored_sha,
        "verified_sha256": actual_sha,
        "width": width,
        "height": height,
    }


def _event_payload(row: dict) -> dict:
    metadata = _json(row.get("event_metadata"), {})
    return {
        "event_id": str(row["event_id"]),
        "task_type": str(row["task_type"]),
        "task_id": str(row["task_id"]),
        "answer": str(row["answer"]),
        "reason_code": row.get("reason_code"),
        "reviewer": str(row.get("reviewer") or "unknown"),
        "created_at": _serial(row.get("created_at")),
        "metadata": metadata,
    }


def _latest(events: list[dict], task_type: str) -> dict | None:
    matches = [event for event in events if event["task_type"] == task_type]
    return matches[-1] if matches else None


def _tags(events: list[dict], page_metadata: dict) -> list[str]:
    result = list(page_metadata.get("difficulty_tags") or [])
    for event in events:
        metadata = event.get("metadata") or {}
        client = metadata.get("client") or {}
        result.extend(client.get("difficulty_tags") or metadata.get("difficulty_tags") or [])
    return sorted({str(tag) for tag in result if str(tag).strip()})


def _photo_record(photo: dict, events: list[dict], run: dict | None) -> dict:
    mask = _json(photo.get("segmentation_mask"), None)
    box = _json(photo.get("bounding_box"), {})
    enhancement = _json(photo.get("enhancement_applied"), {})
    review_provenance = enhancement.get("review_provenance") or {}
    original_prediction = review_provenance.get("original_prediction")
    if original_prediction is None and not str((mask or {}).get("source", "")).startswith("manual"):
        original_prediction = {
            "bounding_box": box,
            "segmentation_mask": mask,
            "segmentation_confidence": photo.get("segmentation_confidence"),
            "aspect_ratio": photo.get("aspect_ratio"),
            "geometry_valid": bool(photo.get("geometry_valid")),
            "status": str(photo.get("photo_status")),
            "snapshot_source": "current_geometry_no_prior_revision",
        }
    prediction_provenance = review_provenance.get("prediction_provenance") or run or {}
    geometry_event = _latest(events, "geometry")
    answer = geometry_event.get("answer") if geometry_event else None
    manual = str((mask or {}).get("source", "")).startswith("manual")
    if answer == "correct":
        state = "accepted"
    elif answer == "wrong_object":
        state = "rejected"
    elif answer:
        state = "needs_correction"
    elif manual:
        state = "corrected_unreviewed"
    else:
        state = "unreviewed"
    added = manual and original_prediction is None
    correction = enhancement.get("correction") or {}
    enhancement_event = _latest(events, "enhancement")
    selection = None
    if enhancement_event:
        event_metadata = enhancement_event.get("metadata") or {}
        client = event_metadata.get("client") or event_metadata
        selection = {
            "answer": enhancement_event.get("answer"),
            "blind_choice": client.get("blind_choice"),
            "candidate_name": client.get("candidate_name"),
        }
    return {
        "photo_id": str(photo["photo_id"]),
        "state": state,
        "added_detection": added,
        "rejected_detection": state == "rejected",
        "original_prediction": original_prediction,
        "prediction_provenance": prediction_provenance,
        "current_geometry": {
            "bounding_box": box,
            "segmentation_mask": mask,
            "segmentation_confidence": photo.get("segmentation_confidence"),
            "aspect_ratio": photo.get("aspect_ratio"),
            "geometry_valid": bool(photo.get("geometry_valid")),
            "status": str(photo.get("photo_status")),
        },
        "correction": correction,
        "orientation": correction.get("orientation") or enhancement.get("orientation") or {},
        "review_reasons": list(enhancement.get("review_reasons") or []),
        "selected_variant": enhancement.get("selected_variant"),
        "enhancement_selection": selection,
        "variants": enhancement.get("variants") or {},
        "review_events": events,
        "reviewers": sorted({event["reviewer"] for event in events}),
    }


def build_manifest_records(
    pages: list[dict],
    photos: list[dict],
    event_rows: list[dict],
    runs: list[dict],
    *,
    output_dir: Path | None = None,
    copy_images: bool = False,
    include_unreviewed: bool = False,
) -> list[dict]:
    page_ids = {str(page["page_id"]) for page in pages}
    photos_by_page: dict[str, list[dict]] = defaultdict(list)
    photo_to_page: dict[str, str] = {}
    for photo in photos:
        page_id = str(photo["page_id"])
        photos_by_page[page_id].append(photo)
        photo_to_page[str(photo["photo_id"])] = page_id

    events_by_page: dict[str, list[dict]] = defaultdict(list)
    events_by_photo: dict[str, list[dict]] = defaultdict(list)
    for row in event_rows:
        event = _event_payload(row)
        task_id = event["task_id"]
        if event["task_type"] == "segmentation" and task_id in page_ids:
            events_by_page[task_id].append(event)
        elif task_id in photo_to_page:
            events_by_photo[task_id].append(event)
            events_by_page[photo_to_page[task_id]].append(event)
    for values in (*events_by_page.values(), *events_by_photo.values()):
        values.sort(key=lambda event: (str(event.get("created_at") or ""), event["event_id"]))

    latest_run: dict[str, dict] = {}
    for row in sorted(runs, key=lambda item: str(item.get("created_at") or "")):
        latest_run[str(row["page_id"])] = {
            "run_id": str(row["run_id"]),
            "input_sha256": row.get("input_sha256"),
            "pipeline_version": row.get("pipeline_version"),
            "config_fingerprint": row.get("config_fingerprint"),
            "status": row.get("run_status"),
            "manifest": _json(row.get("run_manifest"), {}),
        }

    records: list[dict] = []
    for page in pages:
        page_id = str(page["page_id"])
        page_events = events_by_page.get(page_id, [])
        page_photos = [
            _photo_record(photo, events_by_photo.get(str(photo["photo_id"]), []), latest_run.get(page_id))
            for photo in sorted(photos_by_page.get(page_id, []), key=lambda item: str(item["photo_id"]))
        ]
        has_learning_evidence = bool(page_events) or any(
            photo["state"] in {"accepted", "rejected", "needs_correction", "corrected_unreviewed"}
            for photo in page_photos
        )
        if not include_unreviewed and not has_learning_evidence:
            continue
        page_metadata = _json(page.get("processing_metadata"), {})
        dataset_metadata = page_metadata.get("dataset") or {}
        segmentation_event = _latest(page_events, "segmentation")
        reasons = {str(event["reason_code"]) for event in page_events if event.get("reason_code")}
        for photo in page_photos:
            reasons.update(str(reason) for reason in photo["review_reasons"])
        record = {
            "schema_version": 1,
            "page_id": page_id,
            "album_id": str(page["album_id"]),
            "source_group": str(dataset_metadata.get("source_group") or page["album_id"]),
            "page_status": str(page.get("page_status")),
            "image": _image_record(page, output_dir=output_dir, copy_images=copy_images),
            "run": latest_run.get(page_id),
            "segmentation_review": {
                "state": segmentation_event.get("answer") if segmentation_event else None,
                "expected_photo_count": len([photo for photo in page_photos if photo["state"] != "rejected"]),
            },
            "photos": page_photos,
            "review_events": page_events,
            "reviewers": sorted({event["reviewer"] for event in page_events}),
            "reason_codes": sorted(reasons),
            "difficulty_tags": _tags(page_events, dataset_metadata),
        }
        records.append(record)
    return sorted(records, key=lambda record: record["page_id"])


def _database_rows(database_url: str, album_id: str | None) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    sync_url = database_url.replace("+asyncpg", "").replace("+aiosqlite", "")
    engine = create_engine(sync_url)
    album_filter = "WHERE p.album_id = :album_id" if album_id else ""
    params = {"album_id": str(UUID(album_id))} if album_id else {}
    with engine.connect() as connection:
        pages = [
            dict(row)
            for row in connection.execute(
                text(
                    f"""
                    SELECT p.id AS page_id, p.album_id, p.original_filename, p.storage_path,
                           p.file_hash_sha256, p.status AS page_status, p.processing_metadata
                    FROM pages p {album_filter}
                    ORDER BY p.created_at ASC
                    """
                ),
                params,
            ).mappings()
        ]
        page_ids = {str(row["page_id"]) for row in pages}
        photos = [
            dict(row)
            for row in connection.execute(
                text(
                    f"""
                    SELECT ep.id AS photo_id, ep.page_id, ep.bounding_box, ep.segmentation_mask,
                           ep.segmentation_confidence, ep.aspect_ratio, ep.geometry_valid,
                           ep.status AS photo_status, ep.enhancement_applied
                    FROM extracted_photos ep
                    JOIN pages p ON p.id = ep.page_id
                    {album_filter}
                    ORDER BY ep.created_at ASC
                    """
                ),
                params,
            ).mappings()
        ]
        task_ids = page_ids | {str(row["photo_id"]) for row in photos}
        events = [
            dict(row)
            for row in connection.execute(
                text(
                    """
                    SELECT id AS event_id, task_type, task_id, answer, reason_code,
                           reviewer, metadata AS event_metadata, created_at
                    FROM review_events
                    ORDER BY created_at ASC
                    """
                )
            ).mappings()
            if str(row["task_id"]) in task_ids
        ]
        runs = [
            dict(row)
            for row in connection.execute(
                text(
                    f"""
                    SELECT pr.id AS run_id, pr.page_id, pr.input_sha256, pr.pipeline_version,
                           pr.config_fingerprint, pr.status AS run_status,
                           pr.manifest AS run_manifest, pr.created_at
                    FROM processing_runs pr
                    JOIN pages p ON p.id = pr.page_id
                    {album_filter}
                    ORDER BY pr.created_at ASC
                    """
                ),
                params,
            ).mappings()
        ]
    return pages, photos, events, runs


def export_reviewed_dataset(
    *,
    database_url: str,
    output_dir: Path,
    album_id: str | None = None,
    copy_images: bool = False,
    include_unreviewed: bool = False,
) -> list[dict]:
    pages, photos, events, runs = _database_rows(database_url, album_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    records = build_manifest_records(
        pages,
        photos,
        events,
        runs,
        output_dir=output_dir,
        copy_images=copy_images,
        include_unreviewed=include_unreviewed,
    )
    temporary = output_dir / ".manifest.jsonl.tmp"
    temporary.write_text(
        "".join(json.dumps(record, default=_serial, sort_keys=True) + "\n" for record in records), encoding="utf-8"
    )
    temporary.replace(output_dir / "manifest.jsonl")
    summary = {
        "schema_version": 1,
        "pages": len(records),
        "photos": sum(len(record["photos"]) for record in records),
        "source_groups": sorted({record["source_group"] for record in records}),
        "copied_images": copy_images,
    }
    (output_dir / "manifest-summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a provenance-rich reviewed PhoSca dataset manifest.")
    parser.add_argument("--database-url", default=os.environ.get("DATABASE_URL"))
    parser.add_argument("--album-id", default=None)
    parser.add_argument("--output", type=Path, default=Path("data/reviewed_dataset"))
    parser.add_argument(
        "--copy-images", action="store_true", help="Copy originals using page UUID names into output/images."
    )
    parser.add_argument("--include-unreviewed", action="store_true")
    args = parser.parse_args()
    if not args.database_url:
        raise SystemExit("DATABASE_URL or --database-url is required")
    records = export_reviewed_dataset(
        database_url=args.database_url,
        output_dir=args.output,
        album_id=args.album_id,
        copy_images=args.copy_images,
        include_unreviewed=args.include_unreviewed,
    )
    print(json.dumps({"pages": len(records), "output": str(args.output / "manifest.jsonl")}, indent=2))


if __name__ == "__main__":
    main()
