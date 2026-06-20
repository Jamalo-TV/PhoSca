from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

from sqlalchemy import create_engine, text


@dataclass(frozen=True)
class Box:
    x1: float
    y1: float
    x2: float
    y2: float


def polygon_to_box(points: list[tuple[float, float]]) -> Box:
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return Box(min(xs), min(ys), max(xs), max(ys))


def box_iou(a: Box, b: Box) -> float:
    x_left = max(a.x1, b.x1)
    y_top = max(a.y1, b.y1)
    x_right = min(a.x2, b.x2)
    y_bottom = min(a.y2, b.y2)
    if x_right <= x_left or y_bottom <= y_top:
        return 0.0
    intersection = (x_right - x_left) * (y_bottom - y_top)
    area_a = (a.x2 - a.x1) * (a.y2 - a.y1)
    area_b = (b.x2 - b.x1) * (b.y2 - b.y1)
    return intersection / (area_a + area_b - intersection)


def read_yolo_segmentation(label_path: Path) -> list[Box]:
    boxes: list[Box] = []
    if not label_path.exists():
        return boxes
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 7:
            continue
        coords = [float(value) for value in parts[1:]]
        points = list(zip(coords[0::2], coords[1::2]))
        boxes.append(polygon_to_box(points))
    return boxes


def best_match_iou(expected: Box, detected: list[Box]) -> float:
    return max((box_iou(expected, candidate) for candidate in detected), default=0.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--album-id", required=True)
    parser.add_argument("--labels", type=Path, default=Path("data/golden_fixtures/labels"))
    parser.add_argument("--database-url", default=os.environ.get("DATABASE_URL"))
    parser.add_argument("--output", type=Path, default=Path("data/golden_segmentation_report.json"))
    args = parser.parse_args()

    if not args.database_url:
        raise SystemExit("DATABASE_URL or --database-url is required")

    engine = create_engine(args.database_url.replace("+asyncpg", "").replace("+aiosqlite", ""))
    with engine.connect() as connection:
        rows = connection.execute(
            text(
                """
                SELECT p.original_filename, ep.bounding_box
                FROM pages p
                JOIN extracted_photos ep ON ep.page_id = p.id
                WHERE p.album_id = :album_id
                """
            ),
            {"album_id": str(UUID(args.album_id))},
        ).mappings().all()

    detected_by_file: dict[str, list[Box]] = {}
    for row in rows:
        box = row["bounding_box"]
        if isinstance(box, str):
            box = json.loads(box)
        detected_by_file.setdefault(row["original_filename"], []).append(
            Box(float(box["x1"]), float(box["y1"]), float(box["x2"]), float(box["y2"]))
        )

    fixture_reports = []
    all_ious: list[float] = []
    for label_path in sorted(args.labels.glob("*.txt")):
        expected = read_yolo_segmentation(label_path)
        detected = detected_by_file.get(label_path.with_suffix(".jpg").name, [])
        ious = [best_match_iou(box, detected) for box in expected]
        all_ious.extend(ious)
        fixture_reports.append({"fixture": label_path.stem, "ious": ious, "mean_iou": sum(ious) / len(ious) if ious else math.nan})

    mean_iou = sum(all_ious) / len(all_ious) if all_ious else 0.0
    payload = {"mean_iou": mean_iou, "fixtures": fixture_reports}
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if mean_iou < 0.85:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
