from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

import cv2
import numpy as np
from sqlalchemy import create_engine, text


@dataclass(frozen=True)
class Box:
    x1: float
    y1: float
    x2: float
    y2: float


Polygon = list[tuple[float, float]]


def box_to_polygon(box: Box) -> Polygon:
    return [(box.x1, box.y1), (box.x2, box.y1), (box.x2, box.y2), (box.x1, box.y2)]


def polygon_iou(a: Polygon, b: Polygon, *, canvas_size: int = 1024) -> float:
    if len(a) < 3 or len(b) < 3:
        return 0.0
    a_mask = np.zeros((canvas_size, canvas_size), dtype=np.uint8)
    b_mask = np.zeros_like(a_mask)
    a_points = np.array(
        [[int(round(max(0.0, min(1.0, x)) * (canvas_size - 1))), int(round(max(0.0, min(1.0, y)) * (canvas_size - 1)))] for x, y in a],
        dtype=np.int32,
    )
    b_points = np.array(
        [[int(round(max(0.0, min(1.0, x)) * (canvas_size - 1))), int(round(max(0.0, min(1.0, y)) * (canvas_size - 1)))] for x, y in b],
        dtype=np.int32,
    )
    cv2.fillPoly(a_mask, [a_points], 1)
    cv2.fillPoly(b_mask, [b_points], 1)
    intersection = int(np.logical_and(a_mask, b_mask).sum())
    union = int(np.logical_or(a_mask, b_mask).sum())
    return intersection / union if union else 0.0


def read_yolo_segmentation(label_path: Path) -> list[Polygon]:
    polygons: list[Polygon] = []
    if not label_path.exists():
        return polygons
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 7:
            continue
        coords = [float(value) for value in parts[1:]]
        points = list(zip(coords[0::2], coords[1::2]))
        polygons.append(points)
    return polygons


def best_match_iou(expected: Polygon, detected: list[Polygon]) -> float:
    return max((polygon_iou(expected, candidate) for candidate in detected), default=0.0)


def _hungarian_assignment(cost: list[list[float]]) -> list[int]:
    """Return a minimum-cost one-to-one assignment for a square cost matrix."""
    size = len(cost)
    u = [0.0] * (size + 1)
    v = [0.0] * (size + 1)
    parent = [0] * (size + 1)
    way = [0] * (size + 1)
    for row in range(1, size + 1):
        parent[0] = row
        column = 0
        min_values = [float("inf")] * (size + 1)
        used = [False] * (size + 1)
        while True:
            used[column] = True
            current_row = parent[column]
            delta = float("inf")
            next_column = 0
            for candidate_column in range(1, size + 1):
                if used[candidate_column]:
                    continue
                current = cost[current_row - 1][candidate_column - 1] - u[current_row] - v[candidate_column]
                if current < min_values[candidate_column]:
                    min_values[candidate_column] = current
                    way[candidate_column] = column
                if min_values[candidate_column] < delta:
                    delta = min_values[candidate_column]
                    next_column = candidate_column
            for index in range(size + 1):
                if used[index]:
                    u[parent[index]] += delta
                    v[index] -= delta
                else:
                    min_values[index] -= delta
            column = next_column
            if parent[column] == 0:
                break
        while True:
            previous_column = way[column]
            parent[column] = parent[previous_column]
            column = previous_column
            if column == 0:
                break
    assignment = [-1] * size
    for column in range(1, size + 1):
        if parent[column]:
            assignment[parent[column] - 1] = column - 1
    return assignment


def match_detections(expected: list[Polygon], detected: list[Polygon], *, iou_threshold: float = 0.85) -> dict[str, int]:
    """Hungarian one-to-one matching with explicit false-positive accounting."""
    size = max(len(expected), len(detected))
    if not size:
        return {"true_positive": 0, "false_negative": 0, "false_positive": 0, "duplicates": 0}
    ious = [[polygon_iou(truth, prediction) for prediction in detected] for truth in expected]
    # Dummy and below-threshold pairs have equal cost.  The optimizer first
    # maximizes the number of valid matches and then maximizes their IoU.
    cost = [[1.0] * size for _ in range(size)]
    for truth_index, row in enumerate(ious):
        for detection_index, iou in enumerate(row):
            if iou >= iou_threshold:
                cost[truth_index][detection_index] = 1.0 - iou
    assignment = _hungarian_assignment(cost)
    matched_truth: set[int] = set()
    matched_detections: set[int] = set()
    for truth_index, detection_index in enumerate(assignment[:len(expected)]):
        if detection_index < len(detected) and ious[truth_index][detection_index] >= iou_threshold:
            matched_truth.add(truth_index)
            matched_detections.add(detection_index)
    eligible_detections = {
        detection_index
        for detection_index in range(len(detected))
        if any(row[detection_index] >= iou_threshold for row in ious)
    }
    true_positive = len(matched_truth)
    return {
        "true_positive": true_positive,
        "false_negative": len(expected) - true_positive,
        "false_positive": len(detected) - true_positive,
        "duplicates": len(eligible_detections - matched_detections),
    }


def detected_polygon(bounding_box: dict, segmentation_mask: dict | None) -> Polygon:
    if segmentation_mask and segmentation_mask.get("polygon"):
        points = segmentation_mask["polygon"]
        if len(points) >= 3:
            return [(float(point["x"]), float(point["y"])) for point in points]
    return box_to_polygon(
        Box(
            float(bounding_box["x1"]),
            float(bounding_box["y1"]),
            float(bounding_box["x2"]),
            float(bounding_box["y2"]),
        )
    )


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
                SELECT p.original_filename, ep.bounding_box, ep.segmentation_mask
                FROM pages p
                JOIN extracted_photos ep ON ep.page_id = p.id
                WHERE p.album_id = :album_id
                """
            ),
            {"album_id": str(UUID(args.album_id))},
        ).mappings().all()

    detected_by_file: dict[str, list[Polygon]] = {}
    for row in rows:
        box = row["bounding_box"]
        if isinstance(box, str):
            box = json.loads(box)
        mask = row["segmentation_mask"]
        if isinstance(mask, str):
            mask = json.loads(mask)
        detected_by_file.setdefault(row["original_filename"], []).append(detected_polygon(box, mask))

    fixture_reports = []
    all_ious: list[float] = []
    totals = {"true_positive": 0, "false_positive": 0, "false_negative": 0, "duplicates": 0}
    for label_path in sorted(args.labels.glob("*.txt")):
        expected = read_yolo_segmentation(label_path)
        detected = detected_by_file.get(label_path.with_suffix(".jpg").name, [])
        ious = [best_match_iou(box, detected) for box in expected]
        counts = match_detections(expected, detected)
        for name in totals:
            totals[name] += counts[name]
        all_ious.extend(ious)
        fixture_reports.append({"fixture": label_path.stem, "ious": ious, "mean_iou": sum(ious) / len(ious) if ious else math.nan, "page_perfect": counts["false_positive"] == 0 and counts["false_negative"] == 0, **counts})

    mean_iou = sum(all_ious) / len(all_ious) if all_ious else 0.0
    precision = totals["true_positive"] / (totals["true_positive"] + totals["false_positive"]) if totals["true_positive"] + totals["false_positive"] else 0.0
    recall = totals["true_positive"] / (totals["true_positive"] + totals["false_negative"]) if totals["true_positive"] + totals["false_negative"] else 0.0
    page_perfect = sum(1 for report in fixture_reports if report["page_perfect"])
    payload = {"mean_iou": mean_iou, "precision": precision, "recall": recall, "page_perfect": page_perfect, "page_count": len(fixture_reports), **totals, "fixtures": fixture_reports}
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if mean_iou < 0.85:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
