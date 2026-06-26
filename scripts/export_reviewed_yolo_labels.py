from __future__ import annotations

import argparse
import json
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
        return json.loads(value)
    return None


def clamp_unit(value: object) -> float:
    number = float(value)
    return max(0.0, min(1.0, number))


def polygon_area(points: list[tuple[float, float]]) -> float:
    area = 0.0
    for index, (x1, y1) in enumerate(points):
        x2, y2 = points[(index + 1) % len(points)]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0


def mask_is_manual(mask: dict | None) -> bool:
    if not mask:
        return False
    source = str(mask.get("source", ""))
    return source.startswith("manual")


def polygon_from_mask_or_box(mask: dict | None, box: dict | None, *, include_box_fallback: bool) -> list[tuple[float, float]] | None:
    if mask and isinstance(mask.get("polygon"), list) and len(mask["polygon"]) >= 4:
        return [(clamp_unit(point["x"]), clamp_unit(point["y"])) for point in mask["polygon"]]
    if include_box_fallback and box:
        return [
            (clamp_unit(box["x1"]), clamp_unit(box["y1"])),
            (clamp_unit(box["x2"]), clamp_unit(box["y1"])),
            (clamp_unit(box["x2"]), clamp_unit(box["y2"])),
            (clamp_unit(box["x1"]), clamp_unit(box["y2"])),
        ]
    return None


def yolo_line(points: list[tuple[float, float]], *, class_id: int) -> str:
    if len(points) < 4:
        raise ValueError("polygon must contain at least four points")
    if polygon_area(points) < 1e-5:
        raise ValueError("polygon area is too small")
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
) -> ExportedLabelReport:
    output_dir.mkdir(parents=True, exist_ok=True)
    pages = sorted({str(row["original_filename"]) for row in rows})
    lines_by_page: dict[str, list[str]] = {page: [] for page in pages}
    skipped: list[str] = []

    for row in rows:
        original_filename = str(row["original_filename"])
        photo_id = str(row.get("photo_id") or "no-photo")
        if row.get("photo_id") is None:
            continue
        mask = load_json_field(row.get("segmentation_mask"))
        box = load_json_field(row.get("bounding_box"))
        if manual_only and not mask_is_manual(mask):
            skipped.append(f"{photo_id}: mask is not manual")
            continue
        points = polygon_from_mask_or_box(mask, box, include_box_fallback=include_box_fallback)
        if points is None:
            skipped.append(f"{photo_id}: missing polygon")
            continue
        try:
            lines_by_page[original_filename].append(yolo_line(points, class_id=class_id))
        except ValueError as exc:
            skipped.append(f"{photo_id}: {exc}")

    label_files_written = 0
    polygons_written = 0
    missing_pages: list[str] = []
    for original_filename, lines in lines_by_page.items():
        if not lines:
            missing_pages.append(original_filename)
            continue
        output_path = output_dir / f"{Path(original_filename).stem}.txt"
        output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        label_files_written += 1
        polygons_written += len(lines)

    return ExportedLabelReport(
        album_id=album_id,
        pages_seen=len(pages),
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
) -> ExportedLabelReport:
    UUID(album_id)
    sync_url = database_url.replace("+asyncpg", "").replace("+aiosqlite", "")
    engine = create_engine(sync_url)
    with engine.connect() as connection:
        rows = connection.execute(
            text(
                """
                SELECT
                    p.original_filename,
                    ep.id AS photo_id,
                    ep.bounding_box,
                    ep.segmentation_mask
                FROM pages p
                LEFT JOIN extracted_photos ep ON ep.page_id = p.id
                WHERE p.album_id = :album_id
                ORDER BY p.created_at ASC, ep.created_at ASC
                """
            ),
            {"album_id": album_id},
        ).mappings().all()

    return export_rows(
        [dict(row) for row in rows],
        album_id=album_id,
        output_dir=output_dir,
        class_id=class_id,
        manual_only=manual_only,
        include_box_fallback=include_box_fallback,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Export in-app reviewed PhoSca segmentation masks to YOLO segmentation labels.")
    parser.add_argument("--album-id", required=True)
    parser.add_argument("--database-url", default=os.environ.get("DATABASE_URL"))
    parser.add_argument("--output", type=Path, default=Path("data/label_exports/yolo"))
    parser.add_argument("--class-id", type=int, default=0)
    parser.add_argument("--manual-only", action="store_true", help="Export only masks saved from manual review tools.")
    parser.add_argument("--include-box-fallback", action="store_true", help="Use rectangular boxes when a polygon mask is missing.")
    parser.add_argument("--require-complete", action="store_true", help="Exit nonzero if any album page has no exported polygons.")
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
    )
    print(json.dumps(asdict(report), indent=2))
    if args.require_complete and report.missing_pages:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
