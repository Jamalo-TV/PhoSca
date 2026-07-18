"""Evaluate the runtime ONNX segmentation path against locked YOLO fixtures."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
from validate_segmentation import read_yolo_segmentation

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def _runtime_value(value) -> float | None:
    try:
        runtime_ms = float(value)
    except (TypeError, ValueError):
        return None
    return runtime_ms if np.isfinite(runtime_ms) and runtime_ms >= 0.0 else None


def _page_runtime(metadata: dict, end_to_end_ms: float) -> dict[str, float | bool | None]:
    """Normalize one page's wall-clock and optional detector components."""
    model_ms = _runtime_value(metadata.get("model_time_ms"))
    if model_ms is None and metadata.get("model") == "yolov8_seg_onnx":
        model_ms = _runtime_value(metadata.get("inference_time_ms"))
    classical_ms = _runtime_value(metadata.get("classical_time_ms"))
    if classical_ms is None:
        classical = metadata.get("classical_metadata")
        if isinstance(classical, dict):
            classical_ms = _runtime_value(classical.get("classical_time_ms"))
    end_to_end_ms = max(0.0, float(end_to_end_ms))
    accounted_ms = sum(value for value in (model_ms, classical_ms) if value is not None)
    return {
        "segmentation_end_to_end_ms": round(end_to_end_ms, 2),
        "model_ms": model_ms,
        "classical_ms": classical_ms,
        "orchestration_ms": round(max(0.0, end_to_end_ms - accounted_ms), 2),
        "classical_invoked": classical_ms is not None,
    }


def _latency_statistics(values: list[float]) -> dict[str, float | int | None]:
    return {
        "samples": len(values),
        "mean_ms": float(np.mean(values)) if values else None,
        "median_ms": float(np.median(values)) if values else None,
        "p95_ms": float(np.percentile(values, 95)) if values else None,
    }


def _runtime_summary(pages: list[dict[str, float | bool | None]]) -> dict:
    end_to_end = [float(page["segmentation_end_to_end_ms"]) for page in pages]
    model = [float(page["model_ms"]) for page in pages if page["model_ms"] is not None]
    classical = [float(page["classical_ms"]) for page in pages if page["classical_ms"] is not None]
    orchestration = [float(page["orchestration_ms"]) for page in pages]
    end_to_end_stats = _latency_statistics(end_to_end)
    model_stats = _latency_statistics(model)
    classical_stats = _latency_statistics(classical)
    return {
        "timing_scope": "segment_album_page(image, settings); excludes image I/O and evaluator-only scoring",
        "segmentation_end_to_end": end_to_end_stats,
        "model": model_stats,
        "classical": classical_stats,
        "orchestration": _latency_statistics(orchestration),
        "classical_page_count": len(classical),
        "classical_page_rate": len(classical) / len(pages) if pages else 0.0,
        # Flat compatibility fields retained for dashboards consuming v2.
        "mean_end_to_end_ms": end_to_end_stats["mean_ms"],
        "median_end_to_end_ms": end_to_end_stats["median_ms"],
        "p95_end_to_end_ms": end_to_end_stats["p95_ms"],
        "mean_inference_ms": model_stats["mean_ms"],
        "median_inference_ms": model_stats["median_ms"],
        "p95_inference_ms": model_stats["p95_ms"],
        "mean_classical_ms": classical_stats["mean_ms"],
        "peak_memory_mb": None,
        "note": (
            "Wall clock includes preprocessing, detector work, risk assessment, and fusion. "
            "The first model sample may include cold ONNX session initialization."
        ),
    }


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
        YOLO_CONFIDENCE_THRESHOLD=args.acceptance_threshold,
    )


def _detected_polygon(detection) -> list[tuple[float, float]]:
    return [(float(point["x"]), float(point["y"])) for point in detection.mask.get("polygon", [])]


def _load_manifest(path: Path | None) -> dict[str, dict]:
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("pages", payload if isinstance(payload, list) else [])
    return {str(record["id"]): record for record in records}


def _perspective_stability(image, detections) -> tuple[int, list[dict]]:
    from app.pipeline.perspective import crop_and_correct_photo_with_metadata

    stable = 0
    evidence: list[dict] = []
    for index, detection in enumerate(detections):
        try:
            crop, metadata = crop_and_correct_photo_with_metadata(
                image,
                detection.bounding_box,
                detection.mask,
                trim_border=False,
                auto_orient=False,
                protected_regions=[],
                protect_regions=False,
            )
            assessment = metadata.get("geometry_assessment") or metadata.get("quad_validation") or {}
            passed = bool(crop.size and min(crop.shape[:2]) >= 2 and assessment.get("passed", True))
            stable += int(passed)
            evidence.append({"prediction_index": index, "stable": passed, "assessment": assessment})
        except (ValueError, cv2.error) as exc:
            evidence.append({"prediction_index": index, "stable": False, "reason": str(exc)})
    return stable, evidence


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a registered ONNX segmentation model against locked fixtures."
    )
    parser.add_argument("--images", type=Path, default=Path("data/golden_fixtures"))
    parser.add_argument("--labels", type=Path, default=Path("data/golden_fixtures/labels"))
    parser.add_argument("--model", type=Path, default=Path("models/yolov8-seg-album.onnx"))
    parser.add_argument("--manifest", type=Path, default=Path("models/manifest.json"))
    parser.add_argument("--proposal-threshold", type=float, default=0.25)
    parser.add_argument("--acceptance-threshold", type=float, default=0.70)
    parser.add_argument("--include-classical-fallback", action="store_true")
    parser.add_argument("--dataset-manifest", type=Path, default=Path("data/evaluation/cropping_dataset.json"))
    parser.add_argument("--iou-threshold", type=float, default=0.85)
    parser.add_argument("--output", type=Path, default=Path("data/onnx_golden_segmentation_report.json"))
    parser.add_argument("--min-mean-iou", type=float, default=0.85)
    parser.add_argument("--min-precision", type=float, default=0.90)
    parser.add_argument("--min-recall", type=float, default=0.90)
    parser.add_argument("--min-boundary-f-score", type=float, default=0.80)
    parser.add_argument("--max-content-cutoff", type=float, default=0.08)
    parser.add_argument("--max-background-retained", type=float, default=0.12)
    parser.add_argument("--max-serious-failure-rate", type=float, default=0.0)
    args = parser.parse_args()

    if not args.model.is_file():
        raise SystemExit(f"ONNX model missing: {args.model}")
    settings = _runtime_settings(args)
    from app.evaluation.cropping import evaluate_cropping_dataset, evaluate_page_crops
    from app.pipeline.confidence import calibrate_confidence
    from app.pipeline.decisions import component_review_reasons, confidence_components
    from app.pipeline.segmentation import segment_album_page

    manifest = _load_manifest(args.dataset_manifest if args.dataset_manifest.is_file() else None)
    page_metrics = []
    fixtures: list[dict] = []
    page_runtimes: list[dict[str, float | bool | None]] = []
    stable_transforms = 0
    transform_attempts = 0
    orientation_expected = 0
    orientation_correct = 0
    quality_gate_acceptances = 0
    evaluated_detections = 0
    fixture_paths = (
        path for path in args.images.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    for image_path in sorted(fixture_paths):
        label_path = args.labels / f"{image_path.stem}.txt"
        expected = read_yolo_segmentation(label_path)
        if not expected:
            raise SystemExit(f"Golden label missing or empty: {label_path}")
        image = cv2.imread(str(image_path))
        if image is None:
            raise SystemExit(f"Fixture image could not be decoded: {image_path}")
        page_started = perf_counter()
        result = segment_album_page(image, settings)
        page_runtime = _page_runtime(result.metadata, (perf_counter() - page_started) * 1000.0)
        page_runtimes.append(page_runtime)
        if result.metadata.get("onnx_error"):
            raise SystemExit(f"Runtime ONNX error for {image_path.name}: {result.metadata['onnx_error']}")
        detected = [_detected_polygon(item) for item in result.detections]
        accepted: list[bool] = []
        acceptance_evidence: list[dict] = []
        for item in result.detections:
            mask = item.mask or {}
            detector_source = str(mask.get("detector_source", "default"))
            calibrated, calibration = calibrate_confidence(
                item.confidence,
                detector_source,
                settings.segmentation_calibration_path,
            )
            boundary = mask.get("boundary_evidence") if isinstance(mask.get("boundary_evidence"), dict) else {}
            geometry = mask.get("geometry_evidence") if isinstance(mask.get("geometry_evidence"), dict) else {}
            components = confidence_components(
                detection_confidence=calibrated,
                boundary_confidence=boundary.get("score"),
                geometry_confidence=geometry.get("score"),
            )
            component_reasons = component_review_reasons(components)
            quality_accepted = bool(item.geometry_valid and not item.review_reasons and not component_reasons)
            production_accepted = bool(
                quality_accepted
                and calibrated >= settings.yolo_confidence_threshold
                and (calibration.get("calibrated") or not settings.require_confidence_calibration)
            )
            quality_gate_acceptances += int(quality_accepted)
            evaluated_detections += 1
            accepted.append(production_accepted)
            acceptance_evidence.append(
                {
                    "quality_gate_accepted": quality_accepted,
                    "production_accepted": production_accepted,
                    "calibrated": bool(calibration.get("calibrated")),
                    "confidence_components": components.as_dict(),
                    "review_reasons": list(dict.fromkeys([*item.review_reasons, *component_reasons])),
                }
            )
        metrics = evaluate_page_crops(
            expected,
            detected,
            width=image.shape[1],
            height=image.shape[0],
            iou_threshold=args.iou_threshold,
            accepted=accepted,
        )
        page_metrics.append(metrics)
        stable, transform_evidence = _perspective_stability(image, result.detections)
        stable_transforms += stable
        transform_attempts += len(result.detections)
        manifest_record = manifest.get(image_path.stem, {})
        if manifest_record.get("expected_orientation") is not None:
            orientation_expected += 1
            # Page-level orientation is intentionally not inferred from print
            # geometry.  This remains unmeasured until crop labels are present.
        fixtures.append(
            {
                "fixture": image_path.stem,
                "expected": len(expected),
                "detected": len(detected),
                "page_category": manifest_record.get("page_category"),
                "difficulty_tags": manifest_record.get("difficulty_tags", []),
                "annotation_provenance": manifest_record.get("annotation_provenance"),
                "metrics": metrics.as_dict(),
                "page_perfect": (
                    metrics.false_positive == 0 and metrics.false_negative == 0 and metrics.serious_failures == 0
                ),
                "perspective": {"stable": stable, "attempted": len(result.detections), "evidence": transform_evidence},
                "acceptance": acceptance_evidence,
                "runtime": result.metadata,
                "latency": page_runtime,
            }
        )

    aggregate = evaluate_cropping_dataset(page_metrics)
    runtime = _runtime_summary(page_runtimes)
    report = {
        "schema_version": 2,
        "model": str(args.model),
        "manifest": str(args.manifest),
        "dataset_manifest": str(args.dataset_manifest) if args.dataset_manifest.is_file() else None,
        "iou_threshold": args.iou_threshold,
        "classical_evidence_enabled": args.include_classical_fallback,
        "review_policy": {
            "acceptance_threshold": args.acceptance_threshold,
            "require_confidence_calibration": settings.require_confidence_calibration,
            "calibration_artifact_present": settings.segmentation_calibration_path.is_file(),
            "quality_gate_acceptance_rate": (
                quality_gate_acceptances / evaluated_detections if evaluated_detections else 0.0
            ),
            "production_acceptance_is_fail_closed": True,
        },
        "metrics": aggregate.as_dict(),
        # Compatibility summary for existing release scripts.
        "mean_iou": aggregate.polygon_iou,
        "precision": aggregate.detection_precision,
        "recall": aggregate.detection_recall,
        "page_perfect": sum(1 for item in fixtures if item["page_perfect"]),
        "page_count": aggregate.page_count,
        **aggregate.totals,
        "perspective_correction_stability": stable_transforms / transform_attempts if transform_attempts else 0.0,
        "orientation_accuracy": orientation_correct / orientation_expected if orientation_expected else None,
        "orientation_evaluated": orientation_expected,
        "runtime": runtime,
        "limitations": [
            "The locked pages are a same-domain, page-level holdout and are not album-independent.",
            "Orientation accuracy is null until manually verified crop-orientation labels exist.",
            "Mask and polygon IoU are rasterized at source resolution because no vector-geometry "
            "dependency is required.",
            "No reference final crops exist, so output appearance and page-curvature correction are not scored here.",
            "Automatic acceptance and review rates apply the production fail-closed calibration policy; "
            "the separate quality-gate rate excludes calibration availability.",
        ],
        "fixtures": fixtures,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if (
        aggregate.polygon_iou < args.min_mean_iou
        or aggregate.detection_precision < args.min_precision
        or aggregate.detection_recall < args.min_recall
        or aggregate.boundary_f_score < args.min_boundary_f_score
        or aggregate.content_cutoff > args.max_content_cutoff
        or aggregate.background_retained > args.max_background_retained
        or aggregate.serious_failure_rate > args.max_serious_failure_rate
    ):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
