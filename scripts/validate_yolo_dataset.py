from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


@dataclass(frozen=True)
class DatasetIssue:
    severity: str
    path: str
    message: str


@dataclass(frozen=True)
class DatasetReport:
    images_train: int
    images_val: int
    labels_train: int
    labels_val: int
    golden_images: int
    golden_labels: int
    issues: list[DatasetIssue]

    @property
    def valid(self) -> bool:
        return not any(issue.severity == "error" for issue in self.issues)

    def as_dict(self) -> dict:
        payload = asdict(self)
        payload["valid"] = self.valid
        return payload


def parse_simple_yaml(path: Path) -> dict[str, object]:
    data: dict[str, object] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        value = value.strip()
        if value.startswith("[") and value.endswith("]"):
            items = [item.strip().strip("'\"") for item in value[1:-1].split(",") if item.strip()]
            data[key.strip()] = items
        elif value.isdigit():
            data[key.strip()] = int(value)
        else:
            data[key.strip()] = value.strip("'\"")
    return data


def resolve_dataset_root(data_yaml: Path, override_root: Path | None = None) -> Path:
    if override_root is not None:
        return override_root
    data = parse_simple_yaml(data_yaml)
    raw_path = Path(str(data.get("path", data_yaml.parent)))
    if raw_path.is_absolute() and raw_path.exists():
        return raw_path
    if raw_path.exists():
        return raw_path
    candidate = data_yaml.parent / raw_path
    if candidate.exists():
        return candidate
    repo_candidate = Path("data/yolo_dataset")
    if repo_candidate.exists():
        return repo_candidate
    return raw_path


def polygon_area(points: list[tuple[float, float]]) -> float:
    if len(points) < 3:
        return 0.0
    area = 0.0
    for index, (x1, y1) in enumerate(points):
        x2, y2 = points[(index + 1) % len(points)]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0


def parse_yolo_segmentation_line(line: str, *, class_count: int, min_points: int = 4, min_area: float = 1e-5) -> list[tuple[float, float]]:
    parts = line.split()
    if len(parts) < 1 + min_points * 2:
        raise ValueError(f"expected at least {min_points} polygon points")
    try:
        class_id = int(parts[0])
    except ValueError as exc:
        raise ValueError("class id must be an integer") from exc
    if class_id < 0 or class_id >= class_count:
        raise ValueError(f"class id {class_id} outside range 0..{class_count - 1}")
    coords = [float(value) for value in parts[1:]]
    if len(coords) % 2 != 0:
        raise ValueError("polygon coordinates must be x/y pairs")
    if any(not math.isfinite(value) for value in coords):
        raise ValueError("polygon coordinates must be finite")
    if any(value < 0.0 or value > 1.0 for value in coords):
        raise ValueError("polygon coordinates must be normalized to 0..1")
    points = list(zip(coords[0::2], coords[1::2]))
    if polygon_area(points) < min_area:
        raise ValueError("polygon area is too small")
    return points


def image_files(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    return sorted(path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)


def label_files(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    return sorted(path for path in directory.glob("*.txt") if path.is_file())


def validate_label_file(path: Path, *, class_count: int) -> list[DatasetIssue]:
    issues: list[DatasetIssue] = []
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        issues.append(DatasetIssue("error", str(path), "label file is empty"))
        return issues
    for line_number, line in enumerate(lines, 1):
        try:
            parse_yolo_segmentation_line(line, class_count=class_count)
        except ValueError as exc:
            issues.append(DatasetIssue("error", str(path), f"line {line_number}: {exc}"))
    return issues


def validate_dataset(data_yaml: Path, *, root: Path | None = None, golden_dir: Path = Path("data/golden_fixtures")) -> DatasetReport:
    data = parse_simple_yaml(data_yaml)
    class_count = int(data.get("nc", 1))
    dataset_root = resolve_dataset_root(data_yaml, root)
    train_images = image_files(dataset_root / "images" / "train")
    val_images = image_files(dataset_root / "images" / "val")
    train_labels = label_files(dataset_root / "labels" / "train")
    val_labels = label_files(dataset_root / "labels" / "val")
    golden_images = image_files(golden_dir)
    golden_labels = label_files(golden_dir / "labels")

    issues: list[DatasetIssue] = []
    for split, images, labels in (("train", train_images, train_labels), ("val", val_images, val_labels)):
        image_stems = {path.stem for path in images}
        label_stems = {path.stem for path in labels}
        if not images:
            issues.append(DatasetIssue("error", str(dataset_root / "images" / split), f"no {split} images found"))
        for stem in sorted(image_stems - label_stems):
            issues.append(DatasetIssue("error", str(dataset_root / "labels" / split / f"{stem}.txt"), "missing label file for image"))
        for stem in sorted(label_stems - image_stems):
            issues.append(DatasetIssue("error", str(dataset_root / "labels" / split / f"{stem}.txt"), "label file has no matching image"))
        for label in labels:
            issues.extend(validate_label_file(label, class_count=class_count))

    train_stems = {path.stem for path in train_images}
    val_stems = {path.stem for path in val_images}
    golden_stems = {path.stem for path in golden_images}
    for stem in sorted(train_stems & val_stems):
        issues.append(DatasetIssue("error", stem, "image stem appears in both train and val splits"))
    for stem in sorted((train_stems | val_stems) & golden_stems):
        issues.append(DatasetIssue("error", stem, "golden fixture must not appear in train or val splits"))
    golden_label_stems = {path.stem for path in golden_labels}
    for stem in sorted(golden_stems - golden_label_stems):
        issues.append(DatasetIssue("warning", str(golden_dir / "labels" / f"{stem}.txt"), "golden label missing"))

    return DatasetReport(
        images_train=len(train_images),
        images_val=len(val_images),
        labels_train=len(train_labels),
        labels_val=len(val_labels),
        golden_images=len(golden_images),
        golden_labels=len(golden_labels),
        issues=issues,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate PhoSca YOLO segmentation labels before training.")
    parser.add_argument("--data", type=Path, default=Path("data/data.yaml"))
    parser.add_argument("--root", type=Path, default=None, help="Override YOLO dataset root directory.")
    parser.add_argument("--golden-dir", type=Path, default=Path("data/golden_fixtures"))
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    report = validate_dataset(args.data, root=args.root, golden_dir=args.golden_dir)
    payload = report.as_dict()
    text = json.dumps(payload, indent=2)
    if args.output is not None:
        args.output.write_text(text, encoding="utf-8")
    print(text)
    if not report.valid:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
