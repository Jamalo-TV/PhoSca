from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
REPO_ROOT = Path(__file__).resolve().parents[1]


def image_files(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    return sorted(path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)


def label_studio_image_value(image_path: Path, *, document_root: Path, url_prefix: str) -> str:
    try:
        relative = image_path.resolve().relative_to(document_root.resolve())
    except ValueError:
        relative = Path(image_path.name)
    return f"{url_prefix}{relative.as_posix()}"


def detection_to_label_studio_result(detection: object, *, index: int, width: int, height: int) -> dict:
    polygon = detection.mask.get("polygon", [])
    points = [
        [
            round(float(point["x"]) * 100.0, 4),
            round(float(point["y"]) * 100.0, 4),
        ]
        for point in polygon
    ]
    return {
        "id": f"photo-{index}",
        "from_name": "label",
        "to_name": "image",
        "type": "polygonlabels",
        "original_width": width,
        "original_height": height,
        "image_rotation": 0,
        "score": float(detection.confidence),
        "value": {
            "points": points,
            "polygonlabels": ["photo"],
        },
    }


def preannotations_for_image(image_path: Path, *, min_confidence: float) -> tuple[list[dict], dict]:
    backend_path = str(REPO_ROOT / "backend")
    if backend_path not in sys.path:
        sys.path.insert(0, backend_path)
    import cv2

    from app.pipeline.segmentation import detect_photos_classical

    image = cv2.imread(str(image_path))
    if image is None:
        return [], {"warning": "image could not be read"}

    result = detect_photos_classical(image)
    height, width = image.shape[:2]
    detections = [detection for detection in result.detections if detection.confidence >= min_confidence]
    annotations = [
        detection_to_label_studio_result(detection, index=index, width=width, height=height)
        for index, detection in enumerate(detections, 1)
    ]
    return annotations, result.metadata


def build_task(
    image_path: Path,
    *,
    document_root: Path,
    url_prefix: str,
    preannotate: bool,
    min_confidence: float,
) -> dict:
    task: dict = {
        "data": {
            "image": label_studio_image_value(image_path, document_root=document_root, url_prefix=url_prefix),
        }
    }
    if not preannotate:
        return task

    annotations, metadata = preannotations_for_image(image_path, min_confidence=min_confidence)
    if annotations:
        score = sum(float(item["score"]) for item in annotations) / len(annotations)
        task["predictions"] = [
            {
                "model_version": str(metadata.get("model", "classical_hybrid_quad")),
                "score": round(score, 4),
                "result": annotations,
            }
        ]
    return task


def export_tasks(
    *,
    source_images: Path,
    output: Path,
    document_root: Path,
    url_prefix: str,
    preannotate: bool,
    min_confidence: float,
    limit: int = 0,
) -> list[dict]:
    images = image_files(source_images)
    if limit > 0:
        images = images[:limit]
    if not images:
        raise SystemExit(f"No images found in {source_images}")

    tasks = [
        build_task(
            image,
            document_root=document_root,
            url_prefix=url_prefix,
            preannotate=preannotate,
            min_confidence=min_confidence,
        )
        for image in images
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(tasks, indent=2), encoding="utf-8")
    return tasks


def main() -> None:
    parser = argparse.ArgumentParser(description="Export PhoSca album pages as Label Studio import tasks.")
    parser.add_argument("--source-images", type=Path, default=Path("data/raw_album_pages"))
    parser.add_argument("--output", type=Path, default=Path("data/label_studio_tasks.json"))
    parser.add_argument("--document-root", type=Path, default=Path("data"), help="Path mounted as Label Studio's local-files document root.")
    parser.add_argument("--url-prefix", default="/data/local-files/?d=", help="Label Studio local-files URL prefix.")
    parser.add_argument("--preannotate", action="store_true", help="Add classical segmentation predictions as editable polygons.")
    parser.add_argument("--min-confidence", type=float, default=0.45, help="Minimum classical detection confidence to export.")
    parser.add_argument("--limit", type=int, default=0, help="Optional maximum number of images to export.")
    args = parser.parse_args()

    tasks = export_tasks(
        source_images=args.source_images,
        output=args.output,
        document_root=args.document_root,
        url_prefix=args.url_prefix,
        preannotate=args.preannotate,
        min_confidence=args.min_confidence,
        limit=args.limit,
    )
    predictions = sum(len(task.get("predictions", [])) for task in tasks)
    print(json.dumps({"tasks": len(tasks), "prediction_groups": predictions, "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
