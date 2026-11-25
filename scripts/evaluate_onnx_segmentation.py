"""Benchmark detector matches and the rectified images the crop pipeline writes."""

from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
from html import escape
from io import BytesIO
import json
import os
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import cv2
import numpy as np

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
GOLDEN_MANIFEST = Path("data/evaluation/cropping_dataset.json")
MODES = ("full_print", "image_only")


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
        "timing_scope": "segment_album_page(image, settings); excludes decode, cutout preparation, and scoring",
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
        VISIBLE_MASK_MODEL_PATH=args.visible_mask_model,
        MODEL_MANIFEST_PATH=args.manifest,
        REQUIRE_REGISTERED_MODELS=True,
        ENABLE_CLASSICAL_SEGMENTATION_FALLBACK=args.include_classical_fallback or args.classical_only,
        YOLO_PROPOSAL_THRESHOLD=args.proposal_threshold,
        YOLO_CONFIDENCE_THRESHOLD=args.acceptance_threshold,
        ENABLE_PRECISE_PHOTO_CROPS=not args.legacy_crop,
    )


def _detected_polygon(detection) -> list[tuple[float, float]]:
    mask = detection.mask or {}
    return _points(mask.get("visible_polygon") or mask.get("polygon") or [])


def _points(values) -> list[tuple[float, float]]:
    return [
        (float(point["x"]), float(point["y"])) if isinstance(point, dict) else tuple(map(float, point))
        for point in values
    ]


def _record_path(value: str, manifest: Path) -> Path:
    path = Path(value)
    relative = manifest.parent / path
    return path if not relative.is_file() and path.is_file() else relative


def _reference_bytes(path: Path, pinned_hash: str | None) -> bytes:
    payload = path.read_bytes()
    if pinned_hash is not None and sha256(payload).hexdigest() != pinned_hash:
        raise ValueError(f"Evaluation reference hash mismatch: {path}")
    return payload


def _has_disjoint_calibration_holdout(settings, records, detector_profile) -> bool:
    from app.pipeline.training_provenance import disjoint_sources

    try:
        return disjoint_sources(_page_source_index(records), _calibration_source_index(settings, detector_profile))
    except (OSError, ValueError, AttributeError):
        return False


def _page_source_index(records):
    from app.pipeline.training_provenance import TrainingProvenanceError, source_index

    images = source_index({record.get("image_sha256"): record.get("source_group") for record in records})
    if len(images) != len(records):
        raise TrainingProvenanceError("Page source inventory contains duplicate image digests.")
    return images


def _calibration_source_index(settings, detector_profile):
    from app.pipeline.confidence import load_confidence_calibration
    from app.pipeline.training_provenance import TrainingProvenanceError, source_index

    if not settings.require_confidence_calibration:
        raise TrainingProvenanceError("Final evaluation requires confidence calibration.")
    artifact = load_confidence_calibration(settings.segmentation_calibration_path, detector_profile)
    return source_index(artifact["source_images"])


def _load_cases(args) -> tuple[list[dict], dict]:
    if args.dataset_manifest is None and args.images == Path("data/golden_fixtures"):
        args.dataset_manifest = GOLDEN_MANIFEST
    if args.dataset_manifest:
        contents = args.dataset_manifest.read_bytes()
        args.dataset_manifest_sha256 = sha256(contents).hexdigest()
        payload = json.loads(contents)
        records = payload if isinstance(payload, list) else payload["pages"]
        split = payload.get("split_policy", {}) if isinstance(payload, dict) else {}
        if not records or len({str(record["id"]) for record in records}) != len(records):
            raise ValueError("Dataset manifest requires at least one page and unique IDs")
        for record in records:
            record["image"] = _record_path(record["image"], args.dataset_manifest)
            record["annotations"] = (
                _record_path(record["annotations"], args.dataset_manifest) if record.get("annotations") else None
            )
        return records, split
    return [
        {"id": image.stem, "image": image, "annotations": args.labels / f"{image.stem}.txt"}
        for image in sorted(args.images.iterdir()) if image.is_file() and image.suffix.lower() in IMAGE_EXTENSIONS
    ], {}


def _read_page_labels(record, parse_labels):
    label_path = record["annotations"]
    label_hash = record.get("annotations_sha256")
    if label_path and label_path.is_file():
        expected = parse_labels(_reference_bytes(label_path, label_hash).decode("utf-8"))
        pinned = label_hash is not None
    else:
        expected = []
        pinned = label_path is None
    count = record.get("expected_count")
    if (count is not None and len(expected) != count) or (not expected and count != 0):
        raise ValueError(f"Missing or inconsistent label file: {label_path or record['id']}")
    return expected, pinned


def _read_page_truth(record, parse_labels, hash_file):
    expected, pinned = _read_page_labels(record, parse_labels)
    if record.get("image_sha256") and hash_file(record["image"]) != record["image_sha256"]:
        raise ValueError(f"Evaluation image hash mismatch: {record['id']}")
    return expected, pinned


def _reference_scan_crop(payload, quad, shape, *, render):
    from app.pipeline.image_ops import load_image

    source = load_image(BytesIO(payload))
    invalid = "Reference scan needs four convex corners within the decoded raster"
    try:
        normalized = np.asarray(_points(quad), dtype=np.float32)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(invalid) from error
    if (normalized.shape != (4, 2) or not np.isfinite(normalized).all()
            or np.any(normalized < 0) or np.any(normalized > 1) or not cv2.isContourConvex(normalized)):
        raise ValueError(invalid)
    corners = normalized * np.float32([source.shape[1] - 1, source.shape[0] - 1])
    if cv2.contourArea(corners, oriented=True) < 4:
        raise ValueError("Reference scan corners must be clockwise and enclose a crop")
    height, width = shape
    if height * width > 2 * source.shape[0] * source.shape[1]:
        raise ValueError("Reference scan crop exceeds the source raster budget")
    if render:
        target = np.float32([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])
        transform = cv2.getPerspectiveTransform(corners, target)
        return cv2.warpPerspective(source, transform, (width, height))
    return None


def _reference_masks(
    record, expected, image_shape, manifest, *, include_reference_images=False, report_missing_sidecars=False,
):
    from app.evaluation.cropping import _mask
    from app.pipeline.crop_sampling import _outline_mask
    from app.schemas.common import PrintSizeMM

    height, width = image_shape[:2]
    crops = record.get("expected_final_crops")
    if crops is None:
        return [
            {"full_print": {"mask": _mask(polygon, width, height), "verified": False, "present": False}}
            for polygon in expected
        ]
    if len(crops) != len(expected):
        raise ValueError(f"{record['id']}: reference crops must follow annotation order")
    references = []
    for print_crops in crops:
        modes = {}
        for name, annotation in print_crops.items():
            if name not in MODES:
                raise ValueError(f"Unknown reference mode: {name}")
            sidecar_pinned = True
            issues = []
            if annotation.get("mask"):
                mask_path = _record_path(annotation["mask"], manifest)
                mask_hash = annotation.get("mask_sha256")
                if report_missing_sidecars and not mask_path.is_file():
                    modes[name] = {
                        "present": False, "sidecar_pinned": False, "reference_issues": ["MODE_MASK_FILE_MISSING"],
                    }
                    continue
                payload = _reference_bytes(mask_path, mask_hash)
                reference_mask = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_GRAYSCALE) if payload else None
                sidecar_pinned = mask_hash is not None
                if reference_mask is None or reference_mask.shape != (height, width):
                    raise ValueError(f"{record['id']}: invalid source-resolution reference mask")
            else:
                # Preserve the annotated boundary in source coordinates. Rounding
                # vertices before filling can add a row outside the physical edge.
                vertices = np.asarray(_points(annotation["polygon"]), dtype=np.float64) * [width - 1, height - 1]
                reference_mask = _outline_mask(vertices, (height, width))
            recoverable = annotation.get("recoverable")
            if not isinstance(recoverable, bool) or not np.any(reference_mask):
                raise ValueError(f"{record['id']}: {name} needs a recoverability flag and a nonempty reference mask")
            corners = np.array(_points(annotation["warp_quad"]), dtype=np.float32) if annotation.get("warp_quad") else None
            shape = tuple(annotation["shape"]) if annotation.get("shape") is not None else None
            if shape is not None and (len(shape) != 2 or not all(isinstance(size, int) and size >= 2 for size in shape)):
                raise ValueError(f"{record['id']}: {name} has invalid rectified reference dimensions")
            reference_image = None
            reference_quad = annotation.get("reference_image_quad")
            if reference_quad is not None and (shape is None or not annotation.get("reference_image")):
                raise ValueError(f"{record['id']}: {name} reference scan corners need a scan and rectified dimensions")
            reference_path = _record_path(annotation["reference_image"], manifest) if annotation.get("reference_image") else None
            if reference_path is not None and report_missing_sidecars and not reference_path.is_file():
                sidecar_pinned = False
                issues.append("MODE_REFERENCE_SCAN_MISSING")
            elif reference_path is not None:
                reference_hash = annotation.get("reference_image_sha256")
                payload = _reference_bytes(reference_path, reference_hash)
                sidecar_pinned &= reference_hash is not None
                if reference_quad is not None:
                    if (reference_hash or sha256(payload).hexdigest()) == record.get("image_sha256"):
                        raise ValueError(f"{record['id']}: {name} must use a separate scan as its metric reference")
                    reference_image = _reference_scan_crop(payload, reference_quad, shape, render=include_reference_images)
                else:
                    reference_image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR) if payload else None
                    if reference_image is None or min(reference_image.shape[:2]) < 2:
                        raise ValueError(f"{record['id']}: {name} has an invalid rectified reference image")
                    if shape is not None and reference_image.shape[:2] != shape:
                        raise ValueError(f"{record['id']}: {name} reference image does not match its stated shape")
                    shape = reference_image.shape[:2]
            measurement = annotation.get("physical_size_mm")
            ratio = PrintSizeMM.model_validate(measurement).aspect_ratio if measurement is not None else None
            if ratio is not None and shape is not None and abs((shape[1] - 1) / (shape[0] - 1) / ratio - 1) > .01:
                raise ValueError(f"{record['id']}: {name} rectified reference contradicts its physical dimensions")
            source_landmarks = annotation.get("source_landmarks")
            landmarks = annotation.get("landmarks")
            if source_landmarks is not None or landmarks is not None:
                source_landmarks = np.asarray(source_landmarks, dtype=float)
                landmarks = np.asarray(landmarks, dtype=float)
                if (
                    shape is None or source_landmarks.ndim != 2 or source_landmarks.shape[1] != 2
                    or source_landmarks.shape != landmarks.shape or not len(landmarks)
                    or not np.isfinite(source_landmarks).all() or not np.isfinite(landmarks).all()
                    or np.any((source_landmarks < 0) | (source_landmarks > [width - 1, height - 1]))
                    or np.any((landmarks < 0) | (landmarks > [shape[1] - 1, shape[0] - 1]))
                ):
                    raise ValueError(f"{record['id']}: {name} needs paired landmarks in the source and rectified pixel grids")
            valid_corners = bool(
                corners is not None and corners.shape == (4, 2) and np.isfinite(corners).all()
                and np.all((corners >= 0) & (corners <= 1)) and cv2.isContourConvex(corners)
            )
            if annotation.get("annotation_provenance") != "manually_verified":
                issues.append("MODE_ANNOTATION_UNVERIFIED")
            if recoverable and not valid_corners:
                issues.append("MODE_CORNERS_MISSING_OR_INVALID")
            if recoverable and ratio is None and shape is None:
                issues.append("INDEPENDENT_ASPECT_MISSING")
            modes[name] = {
                "mask": reference_mask,
                "native_instance_mask": name == "full_print" and bool(annotation.get("mask")),
                "sidecar_pinned": sidecar_pinned,
                "verified": record.get("annotation_provenance") == "manually_verified" and not issues,
                "reference_issues": issues,
                "recoverable": recoverable,
                "corners": corners * [width - 1, height - 1] if corners is not None else None,
                "shape": shape,
                "expected_aspect_ratio": ratio,
                "aspect_reference": "physical_size_mm" if ratio is not None else "rectified_pixels" if shape else None,
                "image": reference_image if include_reference_images else None,
                "source_landmarks": source_landmarks,
                "landmarks": landmarks,
            }
        references.append(modes)
    return references


def _check_references(args, records, split, parse_labels, load_image, hash_file, *, review_directory: Path | None = None):
    from app.evaluation.cropping import is_locked_album_manifest
    from app.evaluation.reference_review import render_reference_page, write_reference_review
    from app.pipeline.training_provenance import FINAL_SPLIT_ROLES

    totals = {
        name: Counter(present=0, verified=0, verified_recoverable=0, physical_dimensions=0) for name in MODES
    }
    positives = {name: set() for name in MODES}
    expected_photos = 0
    files_pinned = True
    page_issues = []
    mode_issues = []
    groups = set()
    hashes = set()
    sections = []
    calibration = split.get("role") == "calibration"
    if review_directory is not None:
        review_directory.mkdir(parents=True, exist_ok=True)
    for index, record in enumerate(records):
        digest = record.get("image_sha256")
        group = str(record.get("source_group") or "").strip().casefold()
        issues = []
        if group:
            groups.add(group)
        else:
            issues.append("SOURCE_GROUP_MISSING")
        if not digest:
            issues.append("PAGE_IMAGE_HASH_MISSING")
        elif digest in hashes:
            issues.append("DUPLICATE_PAGE_IMAGE")
        else:
            hashes.add(digest)
        if record.get("annotation_provenance") != "manually_verified":
            issues.append("PAGE_ANNOTATION_UNVERIFIED")
        image_missing = not record["image"].is_file()
        label_path = record["annotations"]
        labels_missing = not label_path.is_file() if label_path is not None else record.get("expected_count") != 0
        if image_missing or labels_missing:
            missing = []
            if image_missing:
                issues.append("PAGE_IMAGE_FILE_MISSING")
                missing.append({"kind": "image", "path": str(record["image"].resolve()),
                                "url": record.get("image_url")})
            if labels_missing:
                issues.append("LABEL_FILE_MISSING_OR_UNPINNED")
                missing.append({"kind": "annotations", "path": str(label_path.resolve()) if label_path else None})
                expected_photos += record.get("expected_count") or 0
            else:
                expected, _ = _read_page_labels(record, parse_labels)
                expected_photos += len(expected)
            files_pinned = False
            page_issues.append({"page_id": record["id"], "source_group": group or None,
                                "issues": issues, "missing_artifacts": missing})
            if review_directory is not None:
                name = str(record["id"])
                files = "".join(f'<li><code>{escape(item["path"] or item["kind"])}</code></li>' for item in missing)
                sections.append((name, f'<section id="page-{index:03d}" class="page"><h2>{escape(name)}</h2>'
                                       f'<p>Für die Prüfung fehlen Dateien.</p><ul>{files}</ul></section>'))
            continue
        expected, pinned = _read_page_truth(record, parse_labels, hash_file)
        expected_photos += len(expected)
        files_pinned &= pinned and bool(digest)
        page = load_image(record["image"])
        if not pinned:
            issues.append("LABEL_FILE_MISSING_OR_UNPINNED")
        if issues:
            page_issues.append({"page_id": record["id"], "source_group": group or None, "issues": issues})
        mode_issue_start = len(mode_issues)
        references = _reference_masks(
            record, expected, page.shape, args.dataset_manifest,
            include_reference_images=review_directory is not None and not calibration,
            report_missing_sidecars=True,
        )
        for instance_index, modes in enumerate(references):
            for name in MODES:
                reference = modes.get(name)
                identity = {"page_id": record["id"], "instance_index": instance_index, "mode": name}
                if not reference or not reference.get("present", True):
                    files_pinned &= not reference or reference.get("sidecar_pinned", True)
                    mode_issues.append(identity | {"issues": (reference or {}).get(
                        "reference_issues", ["MODE_ANNOTATION_MISSING"],
                    )})
                    continue
                sidecar_pinned = reference.get("sidecar_pinned", True)
                files_pinned &= sidecar_pinned
                reference_issues = reference["reference_issues"]
                if not sidecar_pinned and "MODE_REFERENCE_SCAN_MISSING" not in reference_issues:
                    reference_issues = [*reference_issues, "MODE_SIDECAR_UNPINNED"]
                if reference_issues:
                    mode_issues.append(identity | {"issues": reference_issues})
                totals[name]["present"] += 1
                totals[name]["verified"] += int(reference["verified"])
                if reference["verified"] and reference.get("recoverable") is True:
                    totals[name]["verified_recoverable"] += 1
                    if group:
                        positives[name].add(group)
                if reference.get("aspect_reference") == "physical_size_mm":
                    totals[name]["physical_dimensions"] += 1
        if review_directory is not None:
            issues_by_mode = {
                (item["instance_index"], item["mode"]): item["issues"]
                for item in mode_issues[mode_issue_start:]
            }
            section = render_reference_page(
                review_directory, index, record, page, expected, references, issues, issues_by_mode,
                calibration=calibration,
            )
            sections.append((str(record["id"]), section))
    final_role = split.get("role") in FINAL_SPLIT_ROLES
    locked = final_role and is_locked_album_manifest(records, split)
    dataset_issues = []
    if calibration:
        dataset_issues.append("CALIBRATION_SPLIT_RESERVED")
    elif not final_role:
        dataset_issues.append("FINAL_SPLIT_ROLE_REQUIRED")
    if split.get("album_independent") is not True:
        dataset_issues.append("ALBUM_INDEPENDENCE_UNCONFIRMED")
    if len(groups) < 2:
        dataset_issues.append("FEWER_THAN_TWO_SOURCE_GROUPS")
    insufficient_groups = [name for name in MODES if len(positives[name]) < 2]
    shared_groups = positives["full_print"] & positives["image_only"]
    if len(shared_groups) < 2:
        dataset_issues.append("FEWER_THAN_TWO_GROUPS_WITH_BOTH_MODES")
    report = {
        "reference_check_only": True,
        "dataset_manifest_sha256": args.dataset_manifest_sha256,
        "pages": len(records),
        "expected_photos": expected_photos,
        "reference_files_pinned": files_pinned,
        "locked_album_provenance": locked,
        "shared_verified_recoverable_groups": sorted(shared_groups),
        "ready_for_locked_evaluation": locked and files_pinned and len(shared_groups) >= 2 and all(
            totals[name]["verified"] == expected_photos for name in MODES
        ),
        "reference_issues": {
            "dataset": dataset_issues, "pages": page_issues, "modes": mode_issues,
            "modes_with_fewer_than_two_verified_recoverable_groups": insufficient_groups,
        },
        "modes": {name: dict(totals[name]) | {"verified_recoverable_groups": sorted(positives[name])} for name in MODES},
    }
    if review_directory is not None:
        report["reference_review_sheet"] = str(
            write_reference_review(review_directory, report, sections, calibration=calibration)
        )
    return report


def _write_reviewed_reference_manifest(source: Path, receipt_path: Path, output: Path, preflight: dict) -> None:
    from app.evaluation.reference_review import apply_reference_review

    manifest = json.loads(_reference_bytes(source, preflight["dataset_manifest_sha256"]))
    receipt = receipt_path.read_bytes()
    reviewed = apply_reference_review(manifest, json.loads(receipt), preflight, receipt_sha256=sha256(receipt).hexdigest())
    pages = reviewed if isinstance(reviewed, list) else reviewed["pages"]

    def rebase(record, keys):
        for key in keys:
            if record.get(key):
                absolute = _record_path(record[key], source).resolve()
                try:
                    record[key] = Path(os.path.relpath(absolute, output.parent.resolve())).as_posix()
                except ValueError:  # Windows paths on different volumes.
                    record[key] = str(absolute)

    for page in pages:
        rebase(page, ("image", "annotations"))
        for modes in page.get("expected_final_crops") or []:
            for mode in modes.values():
                rebase(mode, ("mask", "reference_image"))
    contents = json.dumps(reviewed, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as file:
        file.write(contents)


def _legacy_mode(vertices, image, source_shape, rotation):
    height, width = image.shape[:2]
    source = np.asarray(_points(vertices), dtype=np.float32) * np.float32(
        [source_shape[1] - 1, source_shape[0] - 1]
    )
    destination = np.float32([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])
    destination = np.roll(destination, -(int(rotation) // 90), axis=0)
    return {"status": "legacy", "warp_quad": vertices,
            "page_to_output": cv2.getPerspectiveTransform(source, destination).tolist()}


def _render_crops(image, proposals, settings, temporary, *, protected=None):
    from app.pipeline.image_ops import save_jpeg
    from app.pipeline.perspective import correct_photo_crop, page_plane_from_prints
    from app.pipeline.protected_regions import detect_protected_regions

    if protected is None:
        protected = detect_protected_regions(image) if settings.enable_protected_region_guard else []
    page_plane = None
    if settings.enable_precise_photo_crops:
        page_plane = page_plane_from_prints(image, None, [mask for _, mask, _ in proposals if mask])
    rendered = []
    for index, (box, mask, geometry_valid) in enumerate(proposals):
        try:
            result = correct_photo_crop(
                image, box, mask,
                trim_border=settings.enable_border_trim,
                auto_orient=settings.enable_photo_orientation_correction,
                orientation_model_path=settings.photo_orientation_model_path,
                protected_regions=protected,
                protect_regions=settings.enable_protected_region_guard,
                allow_page_face_overlap=geometry_valid,
                dual_mode=settings.enable_precise_photo_crops,
                page_plane=page_plane,
                enable_classical_retry=settings.enable_classical_segmentation_fallback,
                neighbor_masks=[
                    neighbor_mask for n, (_, neighbor_mask, _) in enumerate(proposals) if n != index and neighbor_mask
                ],
            )
        except (ValueError, cv2.error) as error:
            rendered.append({"reason": str(error), "modes": {}, "images": {}})
            continue
        if settings.enable_precise_photo_crops:
            modes = result.metadata["crop_modes"]
            images = {"full_print": result.image}
            if result.image_only is not None:
                images["image_only"] = result.image_only
        else:
            rotation = (result.metadata.get("orientation") or {}).get("rotation_degrees", 0)
            images = {"full_print": result.rectified_untrimmed}
            modes = {"full_print": _legacy_mode(
                result.metadata["source_quad"], result.rectified_untrimmed, image.shape, rotation,
            )}
            if result.metadata["border_trim"].get("applied"):
                images["image_only"] = result.image
                modes["image_only"] = _legacy_mode(result.metadata["warp_quad"], result.image, image.shape, rotation)
        save_jpeg(temporary / f"{index}-full.jpg", images["full_print"])
        if "image_only" in images and images["image_only"] is not images["full_print"]:
            save_jpeg(temporary / f"{index}-motif.jpg", images["image_only"])
        rendered.append({"images": images, "modes": modes, "metadata": result.metadata})
    return rendered


def _rendered_quality(references, matches, rendered, difficulty, eligible_modes):
    from app.evaluation.cropping import evaluate_rendered_crop

    rows = []
    by_truth = {match["truth_index"]: match for match in matches}
    for truth_index, truth_modes in enumerate(references):
        match = by_truth.get(truth_index)
        render = rendered[match["prediction_index"]] if match else None
        for name, reference in truth_modes.items():
            evidence = render["modes"].get(name, {}) if render else {}
            image = render["images"].get(name) if render else None
            state = evidence.get("status") or ("review_needed" if render and render.get("reason") else "unavailable")
            row = {
                "truth_index": truth_index, "prediction_index": match["prediction_index"] if match else None,
                "mode": name, "status": state, "valid_detection": bool(match and match["valid_detection"]),
                "matched_instance": bool(match and match["mask_iou"] >= .50),
                "verified": reference["verified"], "recoverable": reference.get("recoverable"),
                "independent_of_full_print": evidence.get("independent_of_full_print") is True,
                "aspect_reference": reference.get("aspect_reference") or ("rectified_pixels" if reference.get("shape") else None),
                "difficulty": difficulty, "reason_codes": evidence.get("reason_codes") or
                ([render["reason"]] if render and render.get("reason") else []),
            }
            matrix = evidence.get("page_to_output")
            if image is not None and matrix is not None:
                row.update(evaluate_rendered_crop(
                    reference["mask"], _points(evidence["warp_quad"]), matrix, image.shape[:2],
                    expected_corners=reference.get("corners"),
                    expected_shape=reference.get("shape"),
                    expected_aspect_ratio=reference.get("expected_aspect_ratio"),
                    source_landmarks=reference.get("source_landmarks"),
                    expected_landmarks=reference.get("landmarks"),
                    sampling=evidence.get("sampling"),
                ))
            row["geometry_passes"] = bool(
                row["recoverable"] is not False and row["matched_instance"] and "footprint_iou" in row
                and row["cutoff_beyond_1px"] == row["foreign_beyond_1px"] == row["outside_source_samples"] == 0
                and row["sampling_shortfall_px"] <= 1
                and row["warp_metadata_error_max_px"] <= 1
                and row.get("aspect_ratio_relative_error", 0) <= .01
                and row.get("landmark_error_p95_px", 0) <= 2
                and (row["foreign_source_samples"] == 0 if difficulty != "real" else (
                    row.get("orientation_matches_reference") is True and row.get("corner_error_max_px", np.inf) <= 2
                ))
            )
            row["passes"] = row["geometry_passes"] and state == "ready"
            row["automatic_release_passes"] = bool(row["passes"] and name in eligible_modes[match["prediction_index"]])
            rows.append(row)
    return rows


def _quality_summary(rows):
    summaries = {}
    for name in MODES:
        mode_rows = [row for row in rows if row["mode"] == name]
        verified = [row for row in mode_rows if row["verified"]]
        recoverable = [row for row in verified if row["recoverable"]]
        obstructed = [row for row in verified if row["recoverable"] is False]
        summary = {
            "verified": len(verified), "recoverable": len(recoverable), "obstructed": len(obstructed),
            "obstructed_routed_to_review": sum(row["status"] == "review_needed" for row in obstructed),
            "obstructed_marked_ready": sum(row["status"] == "ready" for row in obstructed),
            "verified_ready_failures": sum(
                row["status"] == "ready" and (not row["recoverable"] or not row["passes"]) for row in verified
            ),
            "ready": sum(row["status"] == "ready" for row in mode_rows),
            "passing_recoverable": sum(row["passes"] for row in recoverable),
            "passing_geometry_in_review": sum(
                row.get("geometry_passes") is True and row["status"] == "review_needed" for row in recoverable
            ),
            "geometry_failures_after_valid_detection": sum(
                row.get("valid_detection") is True and "footprint_iou" in row and not row["geometry_passes"]
                for row in recoverable
            ),
        }
        real_by_group = {}
        for row in recoverable:
            group = row.get("source_group")
            if row["difficulty"] == "real" and group:
                counts = real_by_group.setdefault(group, {"expected": 0, "automatic_release_passes": 0})
                counts["expected"] += 1
                counts["automatic_release_passes"] += int(row["automatic_release_passes"])
        summary["real_by_group"] = real_by_group
        for difficulty in ("clean", "hard", "real"):
            cases = [row for row in recoverable if row["difficulty"] == difficulty]
            errors = [error for row in cases if row["status"] == "ready" for error in row.get("corner_errors_px", [])]
            summary[difficulty] = {
                "expected": len(cases), "passes": sum(row["passes"] for row in cases),
                "automatic_release_passes": sum(row["automatic_release_passes"] for row in cases),
                "corner_error_p95_px": float(np.percentile(errors, 95)) if errors else None,
            }
        scored = [row for row in mode_rows if "footprint_iou" in row]
        for metric in ("footprint_iou", "content_cutoff", "foreign_fraction"):
            summary[f"mean_{metric}"] = float(np.mean([row[metric] for row in scored])) if scored else None
        summaries[name] = summary
    return summaries


def _rendered_mode_passes(mode, *, synthetic: bool, real_manifest_locked: bool, expected_photos: int) -> bool:
    if mode["obstructed_routed_to_review"] != mode["obstructed"] or mode["verified_ready_failures"]:
        return False
    if synthetic:
        clean, hard = mode["clean"], mode["hard"]
        return bool(
            clean["expected"] and clean["passes"] == clean["expected"]
            and clean["corner_error_p95_px"] is not None and clean["corner_error_p95_px"] <= 1
            and hard["expected"] and hard["passes"] / hard["expected"] >= .90
            and hard["corner_error_p95_px"] is not None and hard["corner_error_p95_px"] <= 2
        )
    from app.pipeline.crop_validation import real_mode_coverage_passes

    return real_manifest_locked and real_mode_coverage_passes(mode, expected_photos)


def _detection_passes(metrics, args) -> bool:
    if not metrics.expected_photos:
        return not metrics.detected_photos
    if args.dataset_manifest == GOLDEN_MANIFEST and (metrics.totals["false_positive"] or metrics.totals["false_negative"]):
        return False
    return (
        metrics.polygon_iou >= args.min_mean_iou
        and metrics.detection_precision >= args.min_precision and metrics.detection_recall >= args.min_recall
        and metrics.boundary_f_score >= args.min_boundary_f_score
        and metrics.content_cutoff <= args.max_content_cutoff
        and metrics.background_retained <= args.max_background_retained
        and metrics.serious_failure_rate <= args.max_serious_failure_rate
    )


def _contact_sheet(directory, image, expected, native_truth_masks, detected, rendered, rows, references, index, name):
    height, width = image.shape[:2]
    annotated = image.copy()
    for polygon, mask in zip(expected, native_truth_masks, strict=True):
        if mask is None:
            contours = [np.rint(np.asarray(polygon) * [width - 1, height - 1]).astype(np.int32)]
        else:
            contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cv2.polylines(annotated, contours, True, (65, 225, 25), max(2, width // 500))
    for polygon in detected:
        points = np.rint(np.asarray(polygon) * [width - 1, height - 1]).astype(np.int32)
        if len(points):
            cv2.polylines(annotated, [points], True, (255, 125, 25), max(2, width // 500))
    for item in rendered:
        for mode, color in (("full_print", (25, 75, 255)), ("image_only", (15, 210, 240))):
            quad = _points(item["modes"].get(mode, {}).get("warp_quad") or [])
            if quad:
                points = np.rint(np.asarray(quad) * [width - 1, height - 1]).astype(np.int32)
                cv2.polylines(annotated, [points], True, color, max(2, width // 500))
        for trace in item["modes"].get("image_only", {}).get("suggested_seam_traces", []):
            points = np.rint(np.asarray(_points(trace)) * [width - 1, height - 1]).astype(np.int32)
            cv2.polylines(annotated, [points], False, (235, 65, 190), max(1, width // 800), cv2.LINE_AA)
        refined = _points((item.get("metadata") or {}).get("refined_visible_polygon") or [])
        if refined:
            points = np.rint(np.asarray(refined) * [width - 1, height - 1]).astype(np.int32)
            cv2.polylines(annotated, [points], True, (185, 40, 205), max(1, width // 800))

    def figure(picture, slug, caption, max_width=440):
        width_px = min(max_width, picture.shape[1])
        thumbnail = cv2.resize(picture, (width_px, max(1, round(width_px * picture.shape[0] / picture.shape[1]))),
                               interpolation=cv2.INTER_AREA)
        path = f"{index:03d}-{slug}.jpg"
        cv2.imwrite(str(directory / path), thumbnail)
        return f'<figure><img src="{path}" alt=""><figcaption>{escape(caption)}</figcaption></figure>'

    items = [figure(annotated, "page", "Truth green · detector blue · visible mask magenta · full red · motif yellow · seam trace purple", 920)]
    for prediction_index, item in enumerate(rendered):
        for mode, crop in item["images"].items():
            row = next((r for r in rows if r["prediction_index"] == prediction_index and r["mode"] == mode), {})
            status = item["modes"].get(mode, {}).get("status", "unavailable")
            caption = f"{prediction_index + 1} · {mode} · {status}"
            if "content_cutoff" in row:
                caption += f' · cutoff {row["content_cutoff"]:.2%} · foreign {row["foreign_fraction"]:.2%}'
            slug = f"crop-{prediction_index}-{mode}"
            items.append(figure(crop, slug, caption))
            if "truth_index" in row:
                reference = references[row["truth_index"]][mode].get("image")
                if reference is not None:
                    items.append(figure(reference, f"{slug}-reference", f"{mode} · original reference"))
            edge = min(48, *crop.shape[:2])
            corners = np.hstack((crop[:edge, :edge], crop[:edge, -edge:], crop[-edge:, :edge], crop[-edge:, -edge:]))
            items.append(figure(cv2.resize(corners, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST),
                                f"{slug}-corners", "Corners ×2"))
    return f"<section><h2>{escape(name)}</h2><div>{''.join(items)}</div></section>"


def main() -> None:
    from validate_segmentation import parse_yolo_segmentation

    parser = argparse.ArgumentParser(
        description="Evaluate a registered ONNX segmentation model against locked fixtures."
    )
    parser.add_argument("--images", type=Path, default=Path("data/golden_fixtures"))
    parser.add_argument("--labels", type=Path, default=Path("data/golden_fixtures/labels"))
    parser.add_argument("--model", type=Path, default=Path("models/yolov8-seg-album.onnx"))
    parser.add_argument("--visible-mask-model", type=Path, help="Registered ONNX model for visible mask refinement.")
    parser.add_argument("--manifest", type=Path, default=Path("models/manifest.json"))
    parser.add_argument("--proposal-threshold", type=float, default=0.25)
    parser.add_argument("--acceptance-threshold", type=float, default=0.70)
    parser.add_argument("--include-classical-fallback", action="store_true")
    parser.add_argument("--classical-only", action="store_true", help="Benchmark the production fallback without a model.")
    parser.add_argument("--dataset-manifest", type=Path)
    parser.add_argument("--calibration-output", type=Path, help="Export labelled detector scores from a verified calibration split.")
    parser.add_argument("--check-references", action="store_true", help="Validate real reference files and report readiness without running inference.")
    parser.add_argument("--apply-reference-review", type=Path, help="Import a receipt exported by a source-only review sheet.")
    parser.add_argument("--reviewed-manifest", type=Path, help="Write reviewed annotations to a new manifest.")
    parser.add_argument("--synthetic", choices=("development", "locked"))
    parser.add_argument("--geometry-only", action="store_true", help="Seed approximate proposals on synthetic pages.")
    parser.add_argument("--legacy-crop", action="store_true", help="Measure the rollback detector and crop path.")
    parser.add_argument("--visuals", action="store_true", help="Write a contact sheet alongside the report.")
    parser.add_argument("--require-final-quality", action="store_true")
    parser.add_argument("--iou-threshold", type=float, default=0.85)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--min-mean-iou", type=float, default=0.85)
    parser.add_argument("--min-precision", type=float, default=0.90)
    parser.add_argument("--min-recall", type=float, default=0.90)
    parser.add_argument("--min-boundary-f-score", type=float, default=0.80)
    parser.add_argument("--max-content-cutoff", type=float, default=0.08)
    parser.add_argument("--max-background-retained", type=float, default=0.12)
    parser.add_argument("--max-serious-failure-rate", type=float, default=0.0)
    args = parser.parse_args()
    if args.output is None:
        args.output = Path("runs/crop-quality") / ("reference-check.json" if args.check_references else "evaluation.json")
    if args.dataset_manifest and args.output.resolve() == args.dataset_manifest.resolve():
        parser.error("--output must be separate from the input reference manifest")
    if args.calibration_output:
        if (not args.dataset_manifest or args.synthetic or args.geometry_only or args.legacy_crop or args.check_references
                or args.require_final_quality or args.iou_threshold != 0.85):
            parser.error("--calibration-output needs a separate real calibration manifest and the default IoU threshold")
        if args.calibration_output.resolve() in {args.dataset_manifest.resolve(), args.output.resolve()}:
            parser.error("--calibration-output must be separate from both the manifest and the report")

    if args.geometry_only and (not args.synthetic or args.legacy_crop):
        parser.error("--geometry-only requires a synthetic split and the precise crop path")
    if args.visible_mask_model and (args.legacy_crop or args.geometry_only or args.check_references):
        parser.error("--visible-mask-model requires detector inference with the precise crop path")
    if args.classical_only and (args.geometry_only or args.check_references or args.calibration_output
                                or args.visible_mask_model or args.model.is_file()):
        parser.error("--classical-only requires inference with an absent --model path and no learned refiner")
    if args.synthetic and args.dataset_manifest:
        parser.error("--synthetic and --dataset-manifest are separate inputs")
    if args.check_references and not args.dataset_manifest:
        parser.error("--check-references requires --dataset-manifest")
    if args.apply_reference_review or args.reviewed_manifest:
        if not (args.check_references and args.apply_reference_review and args.reviewed_manifest):
            parser.error("--apply-reference-review requires --check-references and --reviewed-manifest")
        paths = (args.dataset_manifest, args.apply_reference_review, args.reviewed_manifest, args.output)
        if len({path.resolve() for path in paths}) != len(paths):
            parser.error("Review input manifest, receipt, reviewed manifest and report need separate paths")
    if not (args.check_references or args.geometry_only or args.classical_only or args.model.is_file()):
        raise SystemExit(f"ONNX model missing: {args.model}")
    settings = _runtime_settings(args)
    from app.evaluation.cropping import evaluate_cropping_dataset, evaluate_page_crops, is_locked_album_manifest
    from app.pipeline.confidence import CALIBRATION_INPUT_SCHEMA, calibrate_confidence
    from app.pipeline.crop_validation import (
        CROP_VALIDATION_SCHEMA, CROP_VALIDATION_UNAVAILABLE, automatic_crop_review_reasons, crop_validation_profile,
        detector_calibration_profile, detector_training_provenance, geometrically_ready_crop_modes,
        route_ready_crop_modes_to_review,
    )
    from app.pipeline.decisions import component_review_reasons, confidence_components
    from app.pipeline.enhancement import prepare_for_cutout
    from app.pipeline.image_ops import load_image, sha256_file
    from app.pipeline.protected_regions import detect_protected_regions
    from app.pipeline.quality import segmentation_review_reasons
    from app.pipeline.segmentation import segment_album_page
    from app.pipeline.training_provenance import FINAL_SPLIT_ROLES, disjoint_sources

    if args.synthetic:
        from app.evaluation.crop_fixtures import geometry_seed, synthetic_crop_pages

        records = [{"id": page.name, "fixture": page, "annotation_provenance": "generated_fixture"}
                   for page in synthetic_crop_pages(args.synthetic)]
        split = {"synthetic": args.synthetic}
    else:
        records, split = _load_cases(args)
    if not records:
        raise SystemExit("No evaluation pages")
    expected_calibration_detector = None
    training = None
    reviewed_sources = None
    calibration_sources = None
    final_split = split.get("role") in FINAL_SPLIT_ROLES
    calibration_rows: list[dict] = []
    calibration_excluded_pages: list[str] = []
    if not args.check_references and not args.geometry_only:
        expected_calibration_detector = detector_calibration_profile(settings)
    if args.calibration_output:
        if split.get("role") != "calibration" or not is_locked_album_manifest(records, split):
            raise SystemExit("Calibration export requires a verified, album-independent calibration split.")
        if expected_calibration_detector is None:
            raise SystemExit("Calibration export requires the registered detector configuration.")
    if final_split and not args.check_references:
        if not args.require_final_quality:
            raise SystemExit("A reserved final split can only be evaluated with --require-final-quality.")
        if args.classical_only or args.legacy_crop or not is_locked_album_manifest(records, split):
            raise SystemExit("Final evaluation requires a registered precise-crop detector and verified, pinned album pages.")
    if args.calibration_output or (final_split and not args.check_references):
        try:
            training = detector_training_provenance(settings, expected_calibration_detector)
            reviewed_sources = _page_source_index(records)
            calibration_sources = (reviewed_sources if args.calibration_output else
                                   _calibration_source_index(settings, expected_calibration_detector))
        except (OSError, ValueError, AttributeError) as error:
            raise SystemExit(f"Album independence evidence unavailable: {error}") from error
        if not disjoint_sources(training.source_images, calibration_sources):
            raise SystemExit("Calibration shares an image or physical album with detector training or development.")
        if final_split:
            if not disjoint_sources(training.source_images, reviewed_sources):
                raise SystemExit("The final split shares an image or physical album with detector training or development.")
            if not disjoint_sources(calibration_sources, reviewed_sources):
                raise SystemExit("The final split shares an image or physical album with detector calibration.")
    if args.check_references:
        if args.apply_reference_review:
            preflight = _check_references(args, records, split, parse_yolo_segmentation, load_image, sha256_file)
            try:
                _write_reviewed_reference_manifest(
                    args.dataset_manifest, args.apply_reference_review, args.reviewed_manifest, preflight,
                )
            except (OSError, ValueError) as error:
                parser.error(str(error))
            args.dataset_manifest = args.reviewed_manifest
            records, split = _load_cases(args)
        review_directory = args.output.parent / f"{args.output.stem}-visuals" if args.visuals else None
        checked = _check_references(
            args, records, split, parse_yolo_segmentation, load_image, sha256_file,
            review_directory=review_directory,
        )
        if args.reviewed_manifest:
            checked["reviewed_manifest"] = str(args.reviewed_manifest)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        contents = json.dumps(checked, indent=2)
        args.output.write_text(contents, encoding="utf-8")
        print(contents)
        if args.require_final_quality and not checked["ready_for_locked_evaluation"]:
            raise SystemExit(3)
        return
    page_metrics = []
    primary_page_metrics = []
    final_instance_page_metrics = []
    fixtures: list[dict] = []
    page_runtimes: list[dict[str, float | bool | None]] = []
    pipeline_runtimes = []
    crop_rows = []
    mode_availability = {mode: Counter() for mode in MODES}
    sheets = []
    visuals = args.output.parent / f"{args.output.stem}-visuals" if args.visuals else None
    if visuals:
        visuals.mkdir(parents=True, exist_ok=True)
    stable_transforms = 0
    transform_attempts = 0
    quality_gate_acceptances = 0
    evaluated_detections = 0
    reference_files_locked = True
    with TemporaryDirectory(prefix="phosca-crop-benchmark-") as temporary:
        for page_index, record in enumerate(records):
            synthetic = record.get("fixture")
            if synthetic:
                expected = [item.visible_polygon for item in synthetic.prints]
                references = [
                    {mode: {"mask": ref.mask, "corners": ref.corners, "shape": ref.shape,
                            "source_landmarks": ref.source_landmarks, "landmarks": ref.landmarks,
                            "recoverable": ref.recoverable, "verified": True, "image": ref.image}
                     for mode, ref in item.reference_modes.items()}
                    for item in synthetic.prints
                ]
            else:
                expected, pinned = _read_page_truth(record, parse_yolo_segmentation, sha256_file)
                reference_files_locked &= pinned
            pipeline_started = perf_counter()
            image = synthetic.image if synthetic else load_image(record["image"])
            protected = detect_protected_regions(image) if settings.enable_protected_region_guard else []
            detection_metadata = {}
            ownership_unresolved = False
            calibration_detector = None
            acceptance_evidence: list[dict] = []
            if args.geometry_only:
                proposals = [(*geometry_seed(item, image.shape), True) for item in synthetic.prints]
                detected = [_points(mask["visible_polygon"]) for _, mask, _ in proposals]
                page_runtime = None
            else:
                prepared, _ = prepare_for_cutout(image)
                segmentation_started = perf_counter()
                result = segment_album_page(prepared, settings, source_image=image, protected_regions=protected)
                detection_metadata = result.metadata
                page_runtime = _page_runtime(detection_metadata, (perf_counter() - segmentation_started) * 1000.0)
                page_runtimes.append(page_runtime)
                if detection_metadata.get("onnx_error"):
                    raise SystemExit(f"Runtime ONNX error for {record['id']}: {detection_metadata['onnx_error']}")
                refinement = detection_metadata.get("visible_mask_refinement") or {}
                if refinement.get("error"):
                    raise SystemExit(f"Visible mask ONNX error for {record['id']}: {refinement['error']}")
                ownership_unresolved = refinement.get("ownership_unresolved") is True
                calibration_detector = detector_calibration_profile(settings, segmentation=detection_metadata)
                if args.calibration_output and not ownership_unresolved and calibration_detector != expected_calibration_detector:
                    raise SystemExit(f"Calibration detector configuration changed on {record['id']}.")
                proposals = [(item.bounding_box, item.mask, item.geometry_valid) for item in result.detections]
                detected = [_detected_polygon(item) for item in result.detections]

            rendered = _render_crops(image, proposals, settings, Path(temporary), protected=protected)
            if ownership_unresolved:
                for item in rendered:
                    route_ready_crop_modes_to_review(item["modes"], [CROP_VALIDATION_UNAVAILABLE])
            pipeline_ms = (perf_counter() - pipeline_started) * 1000
            pipeline_runtimes.append(pipeline_ms)
            for item in rendered:
                for mode in MODES:
                    unavailable = "review_needed" if item.get("reason") else "unavailable"
                    mode_availability[mode][item["modes"].get(mode, {}).get("status") or unavailable] += 1
            accepted: list[bool] = []
            eligible_modes: list[set[str]] = []
            if not args.geometry_only:
                for item, crop in zip(result.detections, rendered):
                    mask = item.mask or {}
                    detector_source = str(mask.get("detector_source", "default"))
                    calibrated, calibration = calibrate_confidence(
                        item.confidence, detector_source, settings.segmentation_calibration_path,
                        detector_profile=calibration_detector,
                    )
                    boundary = mask.get("boundary_evidence") or {}
                    geometry = mask.get("geometry_evidence") or {}
                    perspective = crop.get("metadata") or {}
                    detection_components = confidence_components(
                        detection_confidence=calibrated,
                        boundary_confidence=boundary.get("score"),
                        geometry_confidence=geometry.get("score"),
                    )
                    components = confidence_components(
                        detection_components.as_dict(),
                        boundary_confidence=perspective.get("boundary_confidence"),
                        geometry_confidence=perspective.get("geometry_confidence"),
                        perspective_confidence=perspective.get("perspective_confidence"),
                        orientation_confidence=perspective.get("orientation_confidence"),
                        border_trim_confidence=perspective.get("border_trim_confidence"),
                    )
                    full_mode = crop["modes"].get("full_print", {})
                    reasons = segmentation_review_reasons(
                        confidence=calibrated, confidence_threshold=settings.yolo_confidence_threshold, quality=item,
                    )
                    reasons.extend(component_review_reasons(detection_components))
                    reasons.extend(component_review_reasons(components))
                    reasons.extend(full_mode.get("reason_codes") or [])
                    orientation_reason = (perspective.get("orientation") or {}).get("reason_code")
                    if orientation_reason:
                        reasons.append(orientation_reason)
                    if (perspective.get("protected_regions") or {}).get("review_untrimmed"):
                        reasons.append("POSSIBLE_CONTENT_CUTOFF")
                    quality_accepted = bool(
                        item.geometry_valid and not reasons and full_mode.get("status") in {"ready", "legacy"}
                    )
                    crop_review = automatic_crop_review_reasons(
                        settings, calibrated, item.geometry_valid, calibration.get("calibrated") is True,
                        item.review_reasons, orientation_reason,
                    )
                    motif_image = crop["images"].get("image_only")
                    modes = geometrically_ready_crop_modes(
                        crop["modes"], image_only_present=motif_image is not None,
                        image_only_is_full_print=motif_image is crop["images"].get("full_print"),
                    ) if not crop_review else set()
                    eligible_modes.append(modes)
                    release_eligible = "full_print" in modes
                    quality_gate_acceptances += int(quality_accepted)
                    evaluated_detections += 1
                    accepted.append(release_eligible or bool(args.legacy_crop and quality_accepted and not crop_review))
                    acceptance_evidence.append({
                        "quality_gate_accepted": quality_accepted,
                        "eligible_before_album_validation": release_eligible,
                        "eligible_crop_modes": sorted(modes),
                        "calibrated": bool(calibration.get("calibrated")),
                        "confidence_components": components.as_dict(),
                        "review_reasons": list(dict.fromkeys(reasons)),
                    })
            if not synthetic:
                references = _reference_masks(record, expected, image.shape, args.dataset_manifest,
                                              include_reference_images=visuals is not None)
                reference_files_locked &= all(
                    reference.get("sidecar_pinned", True) for modes in references for reference in modes.values()
                )
            full_print_references = [modes.get("full_print", {}) for modes in references]
            native_truth_masks = [
                reference["mask"] if reference.get("native_instance_mask") else None
                for reference in full_print_references
            ]
            metrics = evaluate_page_crops(
                expected, detected, width=image.shape[1], height=image.shape[0],
                iou_threshold=args.iou_threshold, accepted=accepted or [False] * len(detected),
                truth_masks=native_truth_masks, diagnose_instances=True,
            )
            page_metrics.append(metrics)
            if args.calibration_output:
                if not pinned or any(
                    mask is not None and (
                        not reference["sidecar_pinned"] or "MODE_ANNOTATION_UNVERIFIED" in reference["reference_issues"]
                    )
                    for mask, reference in zip(native_truth_masks, full_print_references, strict=True)
                ):
                    raise SystemExit(f"Calibration requires verified, pinned instance annotations for {record['id']}.")
                if ownership_unresolved:
                    calibration_excluded_pages.append(record["id"])
                else:
                    correct = {match["prediction_index"] for match in metrics.matches
                               if match["valid_detection"] and not match["serious_failure"]}
                    calibration_rows.extend({
                        "page_id": record["id"], "image_sha256": record["image_sha256"],
                        "source_group": str(record["source_group"]).strip().casefold(), "prediction_index": index,
                        "source": (item.mask or {}).get("detector_source") or "default",
                        "score": item.confidence, "correct": index in correct,
                    } for index, item in enumerate(result.detections))
            primary_metrics = None
            if not args.geometry_only:
                primary = [_points(item.mask.get("polygon") or []) for item in result.detections]
                primary_metrics = metrics if primary == detected else evaluate_page_crops(
                    expected, primary, width=image.shape[1], height=image.shape[0],
                    iou_threshold=args.iou_threshold, accepted=accepted or [False] * len(primary),
                    truth_masks=native_truth_masks,
                )
                primary_page_metrics.append(primary_metrics)
            refined_polygons = [
                (item.get("metadata") or {}).get("refined_visible_polygon") for item in rendered
            ]
            refined_count = sum(bool(polygon) for polygon in refined_polygons)
            final_metrics = metrics
            if refined_count:
                final_polygons = [
                    _points(refined) if refined else raw for raw, refined in zip(detected, refined_polygons, strict=True)
                ]
                final_metrics = evaluate_page_crops(
                    expected, final_polygons, width=image.shape[1], height=image.shape[0],
                    iou_threshold=args.iou_threshold, accepted=accepted or [False] * len(detected),
                    truth_masks=native_truth_masks, diagnose_instances=True,
                )
            final_instance_page_metrics.append(final_metrics)
            rows = _rendered_quality(
                references, final_metrics.matches, rendered, synthetic.difficulty if synthetic else "real",
                eligible_modes or [frozenset()] * len(detected),
            )
            if not synthetic:
                for row in rows:
                    row["source_group"] = str(record.get("source_group") or "").strip().casefold()
            crop_rows.extend(rows)
            stable = sum(bool(item["images"]) for item in rendered)
            stable_transforms += stable
            transform_attempts += len(rendered)
            if visuals:
                sheets.append(_contact_sheet(
                    visuals, image, expected, native_truth_masks, detected, rendered, rows, references, page_index, record["id"],
                ))
            fixtures.append({
                "fixture": record["id"], "expected": len(expected), "detected": len(detected),
                "native_instance_mask_count": sum(mask is not None for mask in native_truth_masks),
                "page_category": record.get("page_category"), "difficulty_tags": record.get("difficulty_tags", []),
                "source_group": record.get("source_group"), "annotation_provenance": record.get("annotation_provenance"),
                "metrics": metrics.as_dict(),
                "primary_detector_metrics": primary_metrics.as_dict() if primary_metrics is not None else None,
                "final_instance_metrics": final_metrics.as_dict(),
                "refined_instances": refined_count,
                "page_perfect": metrics.false_positive == metrics.false_negative == metrics.serious_failures == 0,
                "perspective": {
                    "stable": stable, "attempted": len(rendered),
                    "evidence": [{key: value for key, value in item.items() if key != "images"} for item in rendered],
                },
                "crop_modes": rows,
                "acceptance": acceptance_evidence,
                "runtime": detection_metadata,
                "latency": (page_runtime or {}) | {"image_to_jpeg_ms": pipeline_ms},
            })

    aggregate = evaluate_cropping_dataset(page_metrics)
    final_instances = evaluate_cropping_dataset(final_instance_page_metrics)
    runtime = _runtime_summary(page_runtimes)
    runtime["image_to_jpeg"] = _latency_statistics(pipeline_runtimes)
    runtime["image_to_jpeg_scope"] = (
        "Decode, cutout prep, detection, protected regions, both crop modes and JPEG writes. "
        "Excludes database, later enhancement and evaluation. Geometry-only runs omit decode and detection."
    )
    quality = _quality_summary(crop_rows)
    detector_gate_passed = _detection_passes(aggregate, args)
    final_instance_gate_passed = _detection_passes(final_instances, args)
    calibration_holdout_disjoint = None if args.synthetic else _has_disjoint_calibration_holdout(
        settings, records, expected_calibration_detector,
    )
    training_calibration_disjoint = (
        bool(training and calibration_sources and disjoint_sources(training.source_images, calibration_sources))
        if not args.synthetic else None
    )
    training_release_disjoint = (
        bool(training and final_split and reviewed_sources and disjoint_sources(training.source_images, reviewed_sources))
        if not args.synthetic else None
    )
    real_manifest_locked = (
        not args.synthetic and reference_files_locked and calibration_holdout_disjoint
        and training_calibration_disjoint and training_release_disjoint
        and final_split and is_locked_album_manifest(records, split)
    )
    shared_groups = quality["full_print"]["real_by_group"].keys() & quality["image_only"]["real_by_group"].keys()
    shared_recoverable_album_groups = None if args.synthetic else len(shared_groups) >= 2
    rendered_gates = {
        name: _rendered_mode_passes(
            mode, synthetic=bool(args.synthetic),
            real_manifest_locked=real_manifest_locked and len(shared_groups) >= 2,
            expected_photos=aggregate.expected_photos,
        )
        for name, mode in quality.items()
    }
    validation_gates = {
        "detection": detector_gate_passed,
        "final_instance_segmentation": final_instance_gate_passed,
        "rendered_crop_modes": rendered_gates,
        "locked_real_album_manifest": None if args.synthetic else real_manifest_locked,
        "locked_reference_artifacts": None if args.synthetic else reference_files_locked,
        "calibration_holdout_disjoint": calibration_holdout_disjoint,
        "training_calibration_disjoint": training_calibration_disjoint,
        "training_release_disjoint": training_release_disjoint,
        "shared_recoverable_album_groups": shared_recoverable_album_groups,
        "standard_detection_thresholds": (
            args.iou_threshold >= 0.85 and args.min_mean_iou >= 0.85
            and args.min_precision >= 0.90 and args.min_recall >= 0.90
            and args.min_boundary_f_score >= 0.80 and args.max_content_cutoff <= 0.08
            and args.max_background_retained <= 0.12 and args.max_serious_failure_rate <= 0.0
        ),
    }
    report = {
        "schema_version": CROP_VALIDATION_SCHEMA,
        "model": None if args.classical_only else str(args.model),
        "model_sha256": sha256_file(args.model) if not (args.geometry_only or args.classical_only) else None,
        "manifest": str(args.manifest),
        "dataset_manifest": str(args.dataset_manifest) if args.dataset_manifest else None,
        "dataset_manifest_sha256": getattr(args, "dataset_manifest_sha256", None),
        "training_provenance_sha256": training.manifest_sha256 if training else None,
        "release_source_images": reviewed_sources if final_split else None,
        "split": split,
        "geometry_only": args.geometry_only,
        "legacy_crop": args.legacy_crop,
        "iou_threshold": args.iou_threshold,
        "classical_evidence_enabled": settings.enable_classical_segmentation_fallback,
        "classical_only": args.classical_only,
        "validation_profile": crop_validation_profile(settings),
        "primary_detector_metrics": (
            evaluate_cropping_dataset(primary_page_metrics).as_dict() if primary_page_metrics else None
        ),
        "settings": {
            "pipeline_version": settings.pipeline_version,
            "orientation_enabled": settings.enable_photo_orientation_correction,
            "border_analysis_enabled": settings.enable_border_trim,
            "protected_region_guard_enabled": settings.enable_protected_region_guard,
            "opencv_version": cv2.__version__,
        },
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
        "final_instance_metrics": final_instances.as_dict(),
        # Compatibility summary for existing release scripts.
        "mean_iou": aggregate.polygon_iou,
        "precision": aggregate.detection_precision,
        "recall": aggregate.detection_recall,
        "page_perfect": sum(1 for item in fixtures if item["page_perfect"]),
        "page_count": aggregate.page_count,
        **aggregate.totals,
        "perspective_correction_stability": stable_transforms / transform_attempts if transform_attempts else 0.0,
        "orientation_accuracy": None,
        "orientation_evaluated": 0,
        "crop_modes": quality,
        "validation_gates": validation_gates,
        "crop_mode_availability": {mode: dict(counts) for mode, counts in mode_availability.items()},
        "runtime": runtime,
        "limitations": [
            "The legacy golden split measures same-album regression. Synthetic splits cannot establish real accuracy.",
            "Crop appearance and aspect need independent per-mode references. Unverified outline measurements "
            "are diagnostic only.",
            "Mask and polygon IoU are rasterized at source resolution because no vector-geometry "
            "dependency is required.",
            "Automatic crop eligibility applies final geometry and segmentation confidence before enabling releases "
            "from the independently validated album report. Enhancement is evaluated separately.",
        ],
        "fixtures": fixtures,
    }
    orientation_matches = [row["orientation_matches_reference"] for row in crop_rows
                           if row["verified"] and "orientation_matches_reference" in row]
    if orientation_matches:
        report["orientation_accuracy"] = sum(orientation_matches) / len(orientation_matches)
        report["orientation_evaluated"] = len(orientation_matches)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if visuals:
        (visuals / "index.html").write_text(
            '<!doctype html><meta charset="utf-8"><title>PhoSca crop evaluation</title>'
            '<style>body{font:16px system-ui;background:#181818;color:#eee;margin:24px}'
            'section{margin-bottom:28px}section div{display:flex;gap:12px;flex-wrap:wrap}'
            'figure{margin:0}img{max-width:100%}</style><h1>PhoSca crop evaluation</h1>' + "".join(sheets),
            encoding="utf-8",
        )
    print(json.dumps({"report": str(args.output), "metrics": aggregate.as_dict(),
                      "final_instance_metrics": final_instances.as_dict(), "crop_modes": quality,
                      "validation_gates": validation_gates,
                      "crop_mode_availability": report["crop_mode_availability"],
                      "image_to_jpeg": runtime["image_to_jpeg"]}, indent=2))
    if args.calibration_output:
        args.calibration_output.parent.mkdir(parents=True, exist_ok=True)
        args.calibration_output.write_text(json.dumps({
            "schema_version": CALIBRATION_INPUT_SCHEMA,
            "role": "calibration", "dataset_manifest_sha256": args.dataset_manifest_sha256,
            "detector_profile": expected_calibration_detector,
            "source_images": {record["image_sha256"]: str(record["source_group"]).strip().casefold() for record in records},
            "excluded_pages": calibration_excluded_pages, "rows": calibration_rows,
        }, indent=2) + "\n", encoding="utf-8")
        return
    if not detector_gate_passed:
        raise SystemExit(2)
    if args.require_final_quality and (
        not final_instance_gate_passed or not all(rendered_gates.values())
        or (not args.synthetic and not validation_gates["standard_detection_thresholds"])
    ):
        raise SystemExit(3)


if __name__ == "__main__":
    main()
