"""Camera intrinsics registered against original capture bytes and decoded dimensions."""
import hashlib
import json
from dataclasses import dataclass
from functools import lru_cache
from io import BytesIO
from pathlib import Path

import numpy as np

from app.pipeline.confidence import is_sha256
from app.pipeline.image_ops import load_image


CAMERA_ASPECT_REVIEW = "CAMERA_ASPECT_REVIEW"


@dataclass(frozen=True)
class CameraCalibration:
    capture_sha256: str
    artifact_sha256: str
    calibration_id: str
    width: int
    height: int
    matrix: np.ndarray

    @property
    def profile_sha256(self) -> str:
        payload = json.dumps([self.width, self.height, self.matrix.tolist(), self.calibration_id])
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def provenance(self) -> dict:
        return {
            "capture_sha256": self.capture_sha256,
            "artifact_sha256": self.artifact_sha256,
            "calibration_id": self.calibration_id,
            "profile_sha256": self.profile_sha256,
        }


@lru_cache(maxsize=4)
def _read_calibrations(path: Path, modified_ns: int, size: int) -> tuple[dict, str]:
    payload = path.read_bytes()
    artifact = json.loads(payload)
    if (not isinstance(artifact, dict) or type(artifact.get("schema_version")) is not int
            or artifact["schema_version"] != 1 or not isinstance(artifact.get("captures"), dict)):
        raise ValueError("Invalid camera calibration artifact")
    return artifact["captures"], hashlib.sha256(payload).hexdigest()


def valid_camera_matrix(matrix: np.ndarray, width: int, height: int) -> bool:
    fx, skew, cx = matrix[0]
    cross_axis, fy, cy = matrix[1]
    side = max(width, height)
    return bool(
        np.isfinite(matrix).all() and skew == 0 and cross_axis == 0
        and np.array_equal(matrix[2], [0, 0, 1])
        and 0.1 * side <= min(fx, fy) <= max(fx, fy) <= 20 * side
        and 0 <= cx < width and 0 <= cy < height
    )


def capture_calibration(path: Path | None, capture_sha256: str | None) -> CameraCalibration | None:
    if path is None or not is_sha256(capture_sha256):
        return None
    try:
        stat = path.stat()
        captures, artifact_sha256 = _read_calibrations(path, stat.st_mtime_ns, stat.st_size)
        record = captures[capture_sha256]
        width, height = record["width"], record["height"]
        calibration_id = record["calibration_id"]
        if not (type(width) is int and type(height) is int and min(width, height) > 1
                and isinstance(calibration_id, str) and calibration_id.strip()):
            return None
        rows = record["camera_matrix"]
        if (not isinstance(rows, list) or len(rows) != 3
                or any(not isinstance(row, list) or len(row) != 3 for row in rows)
                or any(type(value) not in {int, float} for row in rows for value in row)):
            return None
        matrix = np.asarray(rows, dtype=np.float64)
        if not valid_camera_matrix(matrix, width, height):
            return None
    except (OSError, KeyError, ValueError, TypeError, OverflowError, RecursionError):
        return None
    matrix.setflags(write=False)
    return CameraCalibration(capture_sha256, artifact_sha256, calibration_id, width, height, matrix)


def load_crop_source(
    path: Path, capture_sha256: str | None, calibration_path: Path | None,
) -> tuple[np.ndarray, CameraCalibration | None]:
    calibration = capture_calibration(calibration_path, capture_sha256)
    if calibration is None:
        return load_image(path), None
    # Decode the same byte snapshot checked against the registered capture.
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != calibration.capture_sha256:
        raise ValueError("Camera calibration source hash does not match the original capture")
    image = load_image(BytesIO(payload))
    if image.shape[:2] != (calibration.height, calibration.width):
        return image, None
    return image, calibration
