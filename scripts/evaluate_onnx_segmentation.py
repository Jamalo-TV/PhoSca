"""Evaluate the runtime ONNX segmentation path against locked YOLO fixtures."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

from validate_segmentation import match_detections, polygon_iou, read_yolo_segmentation


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def _runtime_settings(args: argparse.Namespace):
    repo_root = Path(__file__).resolve().parents[1]
    backend_root = repo_root / "backend"
    if str(backend_root) not in sys.path:
        sys.path.insert(0, str(backend_root))
    from app.config import Settings

    return Settings(
        DATABASE_URL="sqlite+aiosqlite:///./onnx-evaluation.db",
        REDIS_URL="redis://localhost:6379/0",
        YOLO_MODEL_PATH=args.model,
        MODEL_MANIFEST_PATH=args.manifest,
        REQUIRE_REGISTERED_MODELS=True,
        ENABLE_CLASSICAL_SEGMENTATION_FALLBACK=args.include_classical_fallback,
        YOLO_PROPOSAL_THRESHOLD=args.proposal_threshold,
        YOLO_CONFIDENCE_THRESHOLD=args.proposal_threshold,
    )


def _detected_polygon(detection) -> list[tuple[float, float]]:
    return [(float(point["x"]), float(point["y"])) for point in detection.mask.get("polygon", [])]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a registered ONNX segmentation model against locked fixtures.")
    parser.add_argument("--images", type=Path, default=Path("data/golden_fixtures"))
    parser.add_argument("--labels", type=Path, default=Path("data/golden_fixtures/labels"))
    parser.add_argument("--model", type=Path, default=Path("models/yolov8-seg-album.onnx"))
    parser.add_argument("--manifest", type=Path, default=Path("models/manifest.json"))
    parser.add_argument("--proposal-threshold", type=float, default=0.25)
    parser.add_argument("--include-classical-fallback", action="store_true")
    parser.add_argument("--iou-threshold", type=float, default=0.85)
    parser.add_argument("--output", type=Path, default=Path("data/onnx_golden_segmentation_report.json"))
    parser.add_argument("--min-mean-iou", type=float, default=0.85)
    parser.add_argument("--min-precision", type=float, default=0.90)
    parser.add_argument("--min-recall", type=float, default=0.90)
    args = parser.parse_args()

    if not args.model.is_file():
        raise SystemExit(f"ONNX model missing: {args.model}")
    settings = _runtime_settings(args)
    from app.pipeline.segmentation import segment_album_page

    totals = {"true_positive": 0, "false_positive": 0, "false_negative": 0, "duplicates": 0}
    all_ious: list[float] = []
    fixtures: list[dict] = []
    for image_path in sorted(path for path in args.images.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS):
        label_path = args.labels / f"{image_path.stem}.txt"
        expected = read_yolo_segmentation(label_path)
        if not expected:
            raise SystemExit(f"Golden label missing or empty: {label_path}")
        image = cv2.imread(str(image_path))
        if image is None:
            raise SystemExit(f"Fixture image could not be decoded: {image_path}")
        result = segment_album_page(image, settings)
        if result.metadata.get("onnx_error"):
            raise SystemExit(f"Runtime ONNX error for {image_path.name}: {result.metadata['onnx_error']}")
        detected = [_detected_polygon(item) for item in result.detections]
        ious = [max((polygon_iou(truth, candidate) for candidate in detected), default=0.0) for truth in expected]
        counts = match_detections(expected, detected, iou_threshold=args.iou_threshold)
        for name in totals:
            totals[name] += counts[name]
        all_ious.extend(ious)
        fixtures.append(
            {
                "fixture": image_path.stem,
                "expected": len(expected),
                "detected": len(detected),
                "ious": ious,
                "mean_iou": sum(ious) / len(ious),
                "page_perfect": counts["false_positive"] == 0 and counts["false_negative"] == 0,
                "runtime": result.metadata,
                **counts,
            }
        )

    precision_denominator = totals["true_positive"] + totals["false_positive"]
    recall_denominator = totals["true_positive"] + totals["false_negative"]
    report = {
        "model": str(args.model),
        "manifest": str(args.manifest),
        "iou_threshold": args.iou_threshold,
        "mean_iou": sum(all_ious) / len(all_ious) if all_ious else 0.0,
        "precision": totals["true_positive"] / precision_denominator if precision_denominator else 0.0,
        "recall": totals["true_positive"] / recall_denominator if recall_denominator else 0.0,
        "page_perfect": sum(1 for item in fixtures if item["page_perfect"]),
        "page_count": len(fixtures),
        **totals,
        "fixtures": fixtures,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if (
        report["mean_iou"] < args.min_mean_iou
        or report["precision"] < args.min_precision
        or report["recall"] < args.min_recall
    ):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
