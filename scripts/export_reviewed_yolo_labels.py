from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from uuid import UUID

from sqlalchemy import create_engine, text


@dataclass(frozen=True)
class ExportedLabelReport:
    album_id: str
    pages_seen: int
    label_files_written: int
    polygons_written: int
    missing_pages: list[str]
    skipped_photos: list[str]


def load_json_field(value: object) -> dict | None:
    if value is None:
        return None
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        loaded = json.loads(value)
        return loaded if isinstance(loaded, dict) else None
    return None


def strict_unit(value: object) -> float:
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError("polygon coordinates must be finite and normalized to 0..1")
    return number


def polygon_area(points: list[tuple[float, float]]) -> float:
    area = 0.0
    for index, (x1, y1) in enumerate(points):
        x2, y2 = points[(index + 1) % len(points)]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0


def _orientation(a: tuple[float, float], b: tuple[float, float], c: tuple[float, float]) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _proper_intersection(
    a: tuple[float, float],
    b: tuple[float, float],
    c: tuple[float, float],
    d: tuple[float, float],
) -> bool:
    ab_c = _orientation(a, b, c)
    ab_d = _orientation(a, b, d)
    cd_a = _orientation(c, d, a)
    cd_b = _orientation(c, d, b)
    return ab_c * ab_d < 0.0 and cd_a * cd_b < 0.0


def validate_polygon(points: list[tuple[float, float]]) -> None:
    if len(points) < 4:
        raise ValueError("polygon must contain at least four points")
    if len(set(points)) != len(points):
        raise ValueError("polygon contains duplicate points")
    if polygon_area(points) < 1e-5:
        raise ValueError("polygon area is too small")
    size = len(points)
    for first in range(size):
        a = points[first]
        b = points[(first + 1) % size]
        for second in range(first + 1, size):
            if second in {first, (first + 1) % size} or (second + 1) % size in {first, (first + 1) % size}:
                continue
            c = points[second]
            d = points[(second + 1) % size]
            if _proper_intersection(a, b, c, d):
                raise ValueError("polygon self-intersects")


def mask_is_manual(mask: dict | None) -> bool:
    return bool(mask) and str(mask.get("source", "")).startswith("manual")


def polygon_from_mask_or_box(
    mask: dict | None,
    box: dict | None,
    *,
    include_box_fallback: bool,
) -> list[tuple[float, float]] | None:
    if mask and isinstance(mask.get("polygon"), list) and len(mask["polygon"]) >= 4:
        points = [(strict_unit(point["x"]), strict_unit(point["y"])) for point in mask["polygon"]]
        validate_polygon(points)
        return points
    if include_box_fallback and box:
        points = [
            (strict_unit(box["x1"]), strict_unit(box["y1"])),
            (strict_unit(box["x2"]), strict_unit(box["y1"])),
            (strict_unit(box["x2"]), strict_unit(box["y2"])),
            (strict_unit(box["x1"]), strict_unit(box["y2"])),
        ]
        validate_polygon(points)
        return points
    return None


def yolo_line(points: list[tuple[float, float]], *, class_id: int) -> str:
    validate_polygon(points)
    coords = " ".join(f"{coord:.6f}" for point in points for coord in point)
    return f"{class_id} {coords}"


def export_rows(
    rows: list[dict],
    *,
    album_id: str,
    output_dir: Path,
    class_id: int = 0,
    manual_only: bool = False,
    include_box_fallback: bool = False,
    require_review: bool = True,
    strict: bool = True,
) -> ExportedLabelReport:
    output_dir.mkdir(parents=True, exist_ok=True)
    page_records: dict[str, dict] = {}
    filenames_to_pages: dict[str, set[str]] = {}
    for row in rows:
        page_key = str(row.get("page_id") or row["original_filename"])
        filename = str(row["original_filename"])
        page_records.setdefault(page_key, {"filename": filename, "lines": [], "resolved": False})
        filenames_to_pages.setdefault(filename, set()).add(page_key)
    collisions = {name: pages for name, pages in filenames_to_pages.items() if len(pages) > 1}
    if collisions:
        names = ", ".join(sorted(collisions))
        raise ValueError(f"duplicate original filenames would merge labels ({names}); use the manifest-first exporter")

    skipped: list[str] = []
    for row in rows:
        page_key = str(row.get("page_id") or row["original_filename"])
        record = page_records[page_key]
        page_answer = row.get("segmentation_review_answer")
        if row.get("photo_id") is None:
            record["resolved"] = page_answer == "correct"
            continue
        photo_id = str(row["photo_id"])
        answer = row.get("geometry_review_answer") or row.get("review_answer")
        if answer == "wrong_object":
            record["resolved"] = True
            continue
        if require_review and answer != "correct":
            skipped.append(f"{photo_id}: geometry is not reviewer-approved")
            continue
        mask = load_json_field(row.get("segmentation_mask"))
        box = load_json_field(row.get("bounding_box"))
        if manual_only and not mask_is_manual(mask):
            skipped.append(f"{photo_id}: approved mask is not manual")
            continue
        try:
            points = polygon_from_mask_or_box(mask, box, include_box_fallback=include_box_fallback)
            if points is None:
                raise ValueError("missing polygon")
            record["lines"].append(yolo_line(points, class_id=class_id))
            record["resolved"] = True
        except (KeyError, TypeError, ValueError) as exc:
            if strict:
                raise ValueError(f"{photo_id}: {exc}") from exc
            skipped.append(f"{photo_id}: {exc}")

    label_files_written = 0
    polygons_written = 0
    missing_pages: list[str] = []
    for record in sorted(page_records.values(), key=lambda item: item["filename"]):
        filename = record["filename"]
        lines = record["lines"]
        if not record["resolved"]:
            missing_pages.append(filename)
            continue
        output_path = output_dir / f"{Path(filename).stem}.txt"
        # An empty file is a deliberate, reviewed negative page rather than a
        # missing annotation.
        output_path.write_text(("\n".join(lines) + "\n") if lines else "", encoding="utf-8")
        label_files_written += 1
        polygons_written += len(lines)

    return ExportedLabelReport(
        album_id=album_id,
        pages_seen=len(page_records),
        label_files_written=label_files_written,
        polygons_written=polygons_written,
        missing_pages=missing_pages,
        skipped_photos=skipped,
    )


def export_album_labels(
    *,
    album_id: str,
    database_url: str,
    output_dir: Path,
    class_id: int = 0,
    manual_only: bool = False,
    include_box_fallback: bool = False,
    require_review: bool = True,
    strict: bool = True,
) -> ExportedLabelReport:
    UUID(album_id)
    sync_url = database_url.replace("+asyncpg", "").replace("+aiosqlite", "")
    engine = create_engine(sync_url)
    with engine.connect() as connection:
        rows = (
            connection.execute(
                text(
                    """
                SELECT
                    p.id AS page_id,
                    p.original_filename,
                    ep.id AS photo_id,
                    ep.bounding_box,
                    ep.segmentation_mask,
                    (
                        SELECT re.answer
                        FROM review_events re
                        WHERE re.task_type = 'geometry' AND re.task_id = ep.id
                        ORDER BY re.created_at DESC
                        LIMIT 1
                    ) AS geometry_review_answer,
                    (
                        SELECT re.answer
                        FROM review_events re
                        WHERE re.task_type = 'segmentation' AND re.task_id = p.id
                        ORDER BY re.created_at DESC
                        LIMIT 1
                    ) AS segmentation_review_answer
                FROM pages p
                LEFT JOIN extracted_photos ep ON ep.page_id = p.id
                WHERE p.album_id = :album_id
                ORDER BY p.created_at ASC, ep.created_at ASC
                """
                ),
                {"album_id": album_id},
            )
            .mappings()
            .all()
        )

    return export_rows(
        [dict(row) for row in rows],
        album_id=album_id,
        output_dir=output_dir,
        class_id=class_id,
        manual_only=manual_only,
        include_box_fallback=include_box_fallback,
        require_review=require_review,
        strict=strict,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Export strictly reviewed PhoSca geometry to legacy YOLO labels.")
    parser.add_argument("--album-id", required=True)
    parser.add_argument("--database-url", default=os.environ.get("DATABASE_URL"))
    parser.add_argument("--output", type=Path, default=Path("data/label_exports/yolo"))
    parser.add_argument("--class-id", type=int, default=0)
    parser.add_argument(
        "--manual-only", action="store_true", help="Require both reviewer approval and a manually saved mask."
    )
    parser.add_argument(
        "--include-box-fallback", action="store_true", help="Use rectangular boxes when an approved polygon is missing."
    )
    parser.add_argument(
        "--allow-unreviewed", action="store_true", help="Unsafe bootstrap mode: permit current, unreviewed masks."
    )
    parser.add_argument(
        "--skip-invalid", action="store_true", help="Report invalid geometry instead of failing the export."
    )
    parser.add_argument("--require-complete", action="store_true", help="Exit nonzero for unresolved pages or photos.")
    args = parser.parse_args()

    if not args.database_url:
        raise SystemExit("DATABASE_URL or --database-url is required")
    report = export_album_labels(
        album_id=args.album_id,
        database_url=args.database_url,
        output_dir=args.output,
        class_id=args.class_id,
        manual_only=args.manual_only,
        include_box_fallback=args.include_box_fallback,
        require_review=not args.allow_unreviewed,
        strict=not args.skip_invalid,
    )
    print(json.dumps(asdict(report), indent=2))
    if args.require_complete and (report.missing_pages or report.skipped_photos):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
