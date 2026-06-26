from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse


@dataclass(frozen=True)
class ConversionReport:
    tasks: int
    labels_written: int
    polygons_written: int
    missing_annotations: list[str]
    skipped_results: list[str]


def image_stem_from_task(task: dict) -> str:
    image_value = str(task.get("data", {}).get("image", ""))
    parsed = urlparse(image_value)
    query_path = parse_qs(parsed.query).get("d", [""])[0]
    path_value = unquote(query_path or parsed.path or image_value).replace("\\", "/")
    stem = Path(path_value).stem
    if not stem:
        raise ValueError("task is missing data.image")
    return stem


def normalized_points(points: object) -> list[tuple[float, float]]:
    if not isinstance(points, list) or len(points) < 4:
        raise ValueError("polygon must contain at least four points")
    normalized: list[tuple[float, float]] = []
    for point in points:
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            raise ValueError("polygon points must be [x, y] pairs")
        x = float(point[0]) / 100.0
        y = float(point[1]) / 100.0
        if x < 0.0 or x > 1.0 or y < 0.0 or y > 1.0:
            raise ValueError("polygon points must be Label Studio percentages in 0..100")
        normalized.append((x, y))
    return normalized


def polygon_area(points: list[tuple[float, float]]) -> float:
    area = 0.0
    for index, (x1, y1) in enumerate(points):
        x2, y2 = points[(index + 1) % len(points)]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0


def yolo_line_from_result(result: dict, *, class_ids: dict[str, int]) -> str:
    if result.get("type") != "polygonlabels":
        raise ValueError("result is not a polygonlabels annotation")
    value = result.get("value", {})
    labels = value.get("polygonlabels", [])
    if not labels:
        raise ValueError("polygon annotation has no label")
    label = str(labels[0])
    if label not in class_ids:
        raise ValueError(f"unknown label {label!r}")
    points = normalized_points(value.get("points"))
    if polygon_area(points) < 1e-5:
        raise ValueError("polygon area is too small")
    coords = " ".join(f"{coord:.6f}" for point in points for coord in point)
    return f"{class_ids[label]} {coords}"


def reviewed_results(task: dict, *, use_predictions: bool) -> list[dict]:
    results: list[dict] = []
    for annotation in task.get("annotations", []) or []:
        if annotation.get("was_cancelled"):
            continue
        results.extend(annotation.get("result", []) or [])
    if results or not use_predictions:
        return results
    for prediction in task.get("predictions", []) or []:
        results.extend(prediction.get("result", []) or [])
    return results


def convert_tasks(
    tasks: list[dict],
    *,
    output_dir: Path,
    class_ids: dict[str, int] | None = None,
    use_predictions: bool = False,
    overwrite: bool = True,
) -> ConversionReport:
    class_ids = class_ids or {"photo": 0}
    output_dir.mkdir(parents=True, exist_ok=True)
    labels_written = 0
    polygons_written = 0
    missing_annotations: list[str] = []
    skipped_results: list[str] = []

    for task in tasks:
        stem = image_stem_from_task(task)
        lines: list[str] = []
        for index, result in enumerate(reviewed_results(task, use_predictions=use_predictions), 1):
            try:
                lines.append(yolo_line_from_result(result, class_ids=class_ids))
            except (TypeError, ValueError) as exc:
                skipped_results.append(f"{stem}#{index}: {exc}")
        if not lines:
            missing_annotations.append(stem)
            continue
        output_path = output_dir / f"{stem}.txt"
        if output_path.exists() and not overwrite:
            raise SystemExit(f"Refusing to overwrite existing label: {output_path}")
        output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        labels_written += 1
        polygons_written += len(lines)

    return ConversionReport(
        tasks=len(tasks),
        labels_written=labels_written,
        polygons_written=polygons_written,
        missing_annotations=sorted(missing_annotations),
        skipped_results=skipped_results,
    )


def parse_class_ids(values: list[str]) -> dict[str, int]:
    class_ids = {"photo": 0}
    for value in values:
        if "=" not in value:
            raise SystemExit(f"Class mapping must be LABEL=ID, got {value!r}")
        label, raw_id = value.split("=", 1)
        class_ids[label] = int(raw_id)
    return class_ids


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert reviewed Label Studio polygon JSON exports to YOLO segmentation labels.")
    parser.add_argument("--input", type=Path, required=True, help="Label Studio JSON export file.")
    parser.add_argument("--output", type=Path, default=Path("data/label_exports/yolo"), help="Destination directory for YOLO .txt labels.")
    parser.add_argument("--class", dest="classes", action="append", default=[], help="Optional LABEL=ID mapping; defaults to photo=0.")
    parser.add_argument("--use-predictions", action="store_true", help="Use prediction polygons only for tasks without reviewed annotations.")
    parser.add_argument("--no-overwrite", action="store_true", help="Fail if a destination label file already exists.")
    args = parser.parse_args()

    tasks = json.loads(args.input.read_text(encoding="utf-8"))
    if not isinstance(tasks, list):
        raise SystemExit("Label Studio export must be a JSON array of tasks.")
    report = convert_tasks(
        tasks,
        output_dir=args.output,
        class_ids=parse_class_ids(args.classes),
        use_predictions=args.use_predictions,
        overwrite=not args.no_overwrite,
    )
    print(json.dumps(asdict(report), indent=2))
    if report.missing_annotations:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
