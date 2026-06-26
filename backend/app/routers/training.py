from __future__ import annotations

import math
from pathlib import Path

import cv2
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from app.config import Settings, get_settings
from app.pipeline.segmentation import detect_photos_classical
from app.utils.file_security import read_and_validate_upload, validate_original_filename


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
IMAGE_MEDIA_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
}
REPO_ROOT = Path(__file__).resolve().parents[3]

router = APIRouter(prefix="/api/v1/training/segmentation", tags=["training"])


class TrainingPoint(BaseModel):
    x: float = Field(ge=0.0, le=1.0)
    y: float = Field(ge=0.0, le=1.0)


class TrainingPolygon(BaseModel):
    points: list[TrainingPoint] = Field(min_length=4)
    source: str | None = None
    confidence: float | None = None


class TrainingLabelPayload(BaseModel):
    polygons: list[TrainingPolygon] = Field(default_factory=list)


def _workspace(settings: Settings) -> Path:
    path = settings.segmentation_training_workspace
    if not path.is_absolute():
        path = REPO_ROOT / path
    path.mkdir(parents=True, exist_ok=True)
    for name in ("images", "labels", "exports"):
        (path / name).mkdir(parents=True, exist_ok=True)
    return path


def _image_dir(settings: Settings) -> Path:
    return _workspace(settings) / "images"


def _label_dir(settings: Settings) -> Path:
    return _workspace(settings) / "labels"


def _safe_image_path(filename: str, settings: Settings) -> Path:
    clean = validate_original_filename(filename)
    path = _image_dir(settings) / clean
    if path.suffix.lower() not in IMAGE_EXTENSIONS:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Unsupported image extension.")
    try:
        path.resolve().relative_to(_image_dir(settings).resolve())
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid image path.") from exc
    return path


def _label_path_for_image(image_path: Path, settings: Settings) -> Path:
    return _label_dir(settings) / f"{image_path.stem}.txt"


def _image_size(path: Path) -> tuple[int | None, int | None]:
    image = cv2.imread(str(path))
    if image is None:
        return None, None
    height, width = image.shape[:2]
    return int(width), int(height)


def _image_record(path: Path, settings: Settings) -> dict:
    width, height = _image_size(path)
    label_path = _label_path_for_image(path, settings)
    polygons = _read_yolo_label(label_path) if label_path.exists() else []
    return {
        "name": path.name,
        "url": f"/api/v1/training/segmentation/images/{path.name}",
        "width": width,
        "height": height,
        "bytes": path.stat().st_size,
        "labels": len(polygons),
        "has_label": label_path.exists(),
    }


def _image_files(settings: Settings) -> list[Path]:
    image_dir = _image_dir(settings)
    return sorted(path for path in image_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)


def _unique_upload_path(original_filename: str, extension: str, settings: Settings) -> Path:
    clean = validate_original_filename(original_filename)
    stem = Path(clean).stem[:180] or "album-page"
    candidate = _image_dir(settings) / f"{stem}{extension}"
    index = 2
    while candidate.exists():
        candidate = _image_dir(settings) / f"{stem}-{index}{extension}"
        index += 1
    return candidate


def _polygon_area(points: list[TrainingPoint]) -> float:
    area = 0.0
    for index, point in enumerate(points):
        other = points[(index + 1) % len(points)]
        area += point.x * other.y - other.x * point.y
    return abs(area) / 2.0


def _validate_polygon(points: list[TrainingPoint]) -> None:
    if len(points) < 4:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Each polygon needs at least four points.")
    if any(not math.isfinite(point.x) or not math.isfinite(point.y) for point in points):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Polygon coordinates must be finite.")
    if _polygon_area(points) < 1e-5:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Polygon area is too small.")


def _read_yolo_label(path: Path) -> list[dict]:
    polygons: list[dict] = []
    if not path.exists():
        return polygons
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split()
        if len(parts) < 9:
            continue
        coords = [float(value) for value in parts[1:]]
        points = [{"x": coords[index], "y": coords[index + 1]} for index in range(0, len(coords), 2)]
        polygons.append({"points": points, "source": "manual_yolo"})
    return polygons


def _write_yolo_label(path: Path, polygons: list[TrainingPolygon]) -> None:
    lines: list[str] = []
    for polygon in polygons:
        _validate_polygon(polygon.points)
        coords = " ".join(f"{point.x:.6f} {point.y:.6f}" for point in polygon.points)
        lines.append(f"0 {coords}")
    if lines:
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    elif path.exists():
        path.unlink()


@router.get("")
async def list_training_images(settings: Settings = Depends(get_settings)) -> dict:
    workspace = _workspace(settings)
    images = [_image_record(path, settings) for path in _image_files(settings)]
    return {
        "workspace": str(workspace),
        "images_dir": str(workspace / "images"),
        "labels_dir": str(workspace / "labels"),
        "exports_dir": str(workspace / "exports"),
        "images": images,
    }


@router.post("/images", status_code=status.HTTP_201_CREATED)
async def upload_training_images(
    files: list[UploadFile] | None = File(default=None),
    file: list[UploadFile] | None = File(default=None),
    settings: Settings = Depends(get_settings),
) -> dict:
    uploads = (files or []) + (file or [])
    if not uploads:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="At least one file is required.")

    request_size = 0
    saved: list[dict] = []
    for upload in uploads:
        validated, request_size = await read_and_validate_upload(
            upload,
            max_file_size=settings.max_upload_size,
            max_request_size=settings.max_request_size,
            current_request_size=request_size,
        )
        target = _unique_upload_path(validated.original_filename, validated.extension, settings)
        target.write_bytes(validated.content)
        saved.append(_image_record(target, settings))
    return {"images": saved}


@router.get("/images/{filename}")
async def get_training_image(filename: str, settings: Settings = Depends(get_settings)) -> FileResponse:
    path = _safe_image_path(filename, settings)
    if not path.exists():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Image not found.")
    return FileResponse(
        path,
        media_type=IMAGE_MEDIA_TYPES.get(path.suffix.lower(), "application/octet-stream"),
        filename=path.name,
        content_disposition_type="inline",
    )


@router.delete("/images/{filename}")
async def delete_training_image(filename: str, settings: Settings = Depends(get_settings)) -> dict:
    path = _safe_image_path(filename, settings)
    if not path.exists():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Image not found.")
    label_path = _label_path_for_image(path, settings)
    path.unlink()
    if label_path.exists():
        label_path.unlink()
    return {"deleted": path.name}


@router.get("/images/{filename}/labels")
async def get_training_labels(filename: str, settings: Settings = Depends(get_settings)) -> dict:
    image_path = _safe_image_path(filename, settings)
    if not image_path.exists():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Image not found.")
    label_path = _label_path_for_image(image_path, settings)
    return {"image": _image_record(image_path, settings), "polygons": _read_yolo_label(label_path)}


@router.put("/images/{filename}/labels")
async def save_training_labels(filename: str, payload: TrainingLabelPayload, settings: Settings = Depends(get_settings)) -> dict:
    image_path = _safe_image_path(filename, settings)
    if not image_path.exists():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Image not found.")
    label_path = _label_path_for_image(image_path, settings)
    _write_yolo_label(label_path, payload.polygons)
    return {"image": _image_record(image_path, settings), "polygons": _read_yolo_label(label_path)}


@router.post("/images/{filename}/detect")
async def detect_training_labels(filename: str, settings: Settings = Depends(get_settings)) -> dict:
    image_path = _safe_image_path(filename, settings)
    if not image_path.exists():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Image not found.")
    image = cv2.imread(str(image_path))
    if image is None:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Image could not be read.")
    result = detect_photos_classical(image)
    polygons = [
        {
            "points": detection.mask["polygon"],
            "source": detection.mask.get("source", "classical_hybrid_quad"),
            "confidence": detection.confidence,
        }
        for detection in result.detections
    ]
    return {"image": _image_record(image_path, settings), "polygons": polygons, "metadata": result.metadata}


@router.post("/prepare")
async def prepare_training_dataset(settings: Settings = Depends(get_settings)) -> dict:
    workspace = _workspace(settings)
    labels = sorted((workspace / "labels").glob("*.txt"))
    if not labels:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No labels found.")
    scripts_dir = REPO_ROOT / "scripts"
    import sys

    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    from prepare_yolo_dataset import prepare_dataset
    from validate_yolo_dataset import validate_dataset

    manifest = prepare_dataset(
        source_images=workspace / "images",
        source_labels=workspace / "labels",
        output_dataset=REPO_ROOT / "data" / "yolo_dataset",
        golden_dir=REPO_ROOT / "data" / "golden_fixtures",
        seed=42,
        val_count=5,
        golden_count=10,
        require_labels=True,
    )
    report = validate_dataset(REPO_ROOT / "data" / "data.yaml", root=REPO_ROOT / "data" / "yolo_dataset", golden_dir=REPO_ROOT / "data" / "golden_fixtures")
    return {"manifest": manifest.__dict__, "validation": report.as_dict()}
