"""Fit a near-pinhole camera and pin it to explicitly named album captures."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from tempfile import NamedTemporaryFile

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.pipeline.camera_calibration import valid_camera_matrix
from app.pipeline.image_ops import load_image, sha256_file


MIN_VIEWS = 6
MAX_DISTORTION_PX = 0.5
MAX_HELD_OUT_RMS_PX = 0.5
MAX_HELD_OUT_POINT_PX = 1.25
MAX_INTRINSIC_DRIFT_PX = 1.0


def checkerboard_points(columns: int, rows: int) -> np.ndarray:
    if min(columns, rows) < 4:
        raise ValueError("Use a checkerboard with at least four interior corners on each axis.")
    grid = np.zeros((columns * rows, 3), dtype=np.float32)
    grid[:, :2] = np.indices((rows, columns), dtype=np.float32)[::-1].reshape(2, -1).T
    return grid


def _fit(
    grid: np.ndarray, views: list[np.ndarray], size: tuple[int, int], *, pinhole: bool = False,
) -> tuple[float, np.ndarray, np.ndarray, tuple[np.ndarray, ...]]:
    flags = 0
    if pinhole:
        flags = cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3
    rms, camera, distortion, rotations, _ = cv2.calibrateCamera(
        [grid] * len(views), views, size, None, None, flags=flags,
        criteria=(cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 70, 1e-9),
    )
    width, height = size
    if (
        not np.isfinite(rms) or not np.isfinite(distortion).all()
        or not valid_camera_matrix(camera, width, height)
    ):
        raise ValueError("The calibration produced invalid camera intrinsics.")
    return rms, camera, distortion, rotations


def fit_camera(grid: np.ndarray, views: list[np.ndarray], size: tuple[int, int]) -> tuple[np.ndarray, dict]:
    """Fit on ordered OpenCV checkerboard corners in the decoded raster."""
    if len(views) < MIN_VIEWS:
        raise ValueError(f"At least {MIN_VIEWS} checkerboard views are needed.")
    views = [np.asarray(view, np.float32).reshape(-1, 2) for view in views]
    width, height = size
    bounds = np.array([width - 1, height - 1], dtype=np.float64)
    if min(bounds) <= 0 or any(
        len(view) != len(grid) or not np.isfinite(view).all() or np.any(view < 0) or np.any(view > bounds)
        for view in views
    ):
        raise ValueError("Each view needs a full set of checkerboard corners inside the decoded raster.")
    normalized = np.concatenate(views) / bounds
    frame_corners = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float64)
    if np.max(np.linalg.norm(normalized[None] - frame_corners[:, None], axis=2).min(axis=1)) > .27:
        raise ValueError("Checkerboard observations must cover all four image corners across the views.")

    _, lens, distortion, rotations = _fit(grid, views, size)
    normals = np.array([cv2.Rodrigues(rotation)[0][:, 2] for rotation in rotations])
    if not np.isfinite(normals).all():
        raise ValueError("The calibration produced invalid checkerboard poses.")
    normals *= np.where(normals[:, 2:] < 0, -1, 1)
    tilts = normals[:, :2]
    spread = np.linalg.svd(tilts - tilts.mean(axis=0), compute_uv=False) / np.sqrt(len(views))
    if spread[-1] < .05:
        raise ValueError("Tilt the checkerboard in both directions across distinct views.")

    grid_x, grid_y = np.meshgrid(np.linspace(0, width - 1, 21), np.linspace(0, height - 1, 21))
    pixels = np.column_stack((grid_x.ravel(), grid_y.ravel()))
    rays = (pixels - lens[:2, 2]) / lens.diagonal()[:2]
    distorted, _ = cv2.projectPoints(
        np.column_stack((rays, np.ones(len(rays)))), np.zeros(3), np.zeros(3), lens, distortion,
    )
    lens_shift = float(np.linalg.norm(distorted.reshape(-1, 2) - pixels, axis=1).max())
    if not lens_shift <= MAX_DISTORTION_PX:
        raise ValueError(f"Lens distortion shifts the full-frame grid by {lens_shift:.3f} px (limit {MAX_DISTORTION_PX} px).")

    rms, camera, _, _ = _fit(grid, views, size, pinhole=True)
    frame_corners *= bounds
    rays = (frame_corners - camera[:2, 2]) / camera.diagonal()[:2]
    held_out = []
    for index, view in enumerate(views):
        _, withheld_camera, _, _ = _fit(grid, views[:index] + views[index + 1:], size, pinhole=True)
        fitted, rvec, tvec = cv2.solvePnP(grid, view, withheld_camera, None)
        if not fitted:
            raise ValueError(f"Could not validate the pose of checkerboard view {index + 1}.")
        predicted, _ = cv2.projectPoints(grid, rvec, tvec, withheld_camera, None)
        errors = np.linalg.norm(predicted.reshape(-1, 2) - view, axis=1)
        projected_frame = rays * withheld_camera.diagonal()[:2] + withheld_camera[:2, 2]
        drift = float(np.linalg.norm(projected_frame - frame_corners, axis=1).max())
        view_rms = float(np.sqrt(np.mean(errors**2)))
        point_error = float(errors.max())
        if not (
            view_rms <= MAX_HELD_OUT_RMS_PX
            and point_error <= MAX_HELD_OUT_POINT_PX
            and drift <= MAX_INTRINSIC_DRIFT_PX
        ):
            raise ValueError(
                f"Checkerboard view {index + 1} failed held-out validation "
                f"(RMS {view_rms:.3f} px, worst corner {point_error:.3f} px, intrinsic drift {drift:.3f} px)."
            )
        held_out.append({"rms_px": view_rms, "max_corner_error_px": point_error, "intrinsic_drift_px": drift})
    return camera, {"fit_rms_px": float(rms), "max_sampled_distortion_px": lens_shift, "held_out_views": held_out}


def _read_capture(path: Path) -> tuple[str, np.ndarray]:
    digest = sha256_file(path)
    image = load_image(path)
    if digest != sha256_file(path):
        raise ValueError(f"The capture changed while being read: {path}")
    return digest, image


def calibration_artifact(columns: int, rows: int, checkerboards: list[Path], captures: list[Path]) -> dict:
    if len(checkerboards) < MIN_VIEWS:
        raise ValueError(f"At least {MIN_VIEWS} checkerboard views are needed.")
    if not captures:
        raise ValueError("Provide at least one album capture to register.")
    grid = checkerboard_points(columns, rows)
    size = None
    boards = {}
    for path in checkerboards:
        digest, image = _read_capture(path)
        if digest in boards:
            raise ValueError(f"Duplicate checkerboard capture: {path}")
        current = image.shape[1::-1]
        if size is not None and current != size:
            raise ValueError(f"Checkerboard raster has a different decoded size: {path}")
        size = current
        found, corners = cv2.findChessboardCornersSB(
            cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (columns, rows),
            cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY,
        )
        if not found:
            raise ValueError(f"Could not locate the complete checkerboard: {path}")
        boards[digest] = corners
    sources = sorted(boards)
    matrix, diagnostics = fit_camera(grid, [boards[digest] for digest in sources], size)
    record = {"width": size[0], "height": size[1], "camera_matrix": matrix.tolist()}
    evidence = {
        **record,
        "checkerboard_sha256": sources,
        "pattern_size": [columns, rows],
        "opencv_version": cv2.__version__,
        **diagnostics,
    }
    record["calibration_id"] = hashlib.sha256(json.dumps(evidence, sort_keys=True).encode()).hexdigest()
    entries = {}
    for path in captures:
        digest, image = _read_capture(path)
        if image.shape[1::-1] != size:
            raise ValueError(f"Album capture has a different decoded size: {path}")
        if digest in boards:
            raise ValueError(f"A checkerboard view cannot also be an album capture: {path}")
        entries[digest] = record
    return {"schema_version": 1, "captures": entries, "fit": evidence}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=(
            "Use a flat board with square cells. Cover all four image corners across the views. "
            "Keep the same camera, lens, focus, capture mode and orientation for boards and album captures."
        ),
    )
    parser.add_argument("--cols", type=int, required=True, help="Number of interior checkerboard columns.")
    parser.add_argument("--rows", type=int, required=True, help="Number of interior checkerboard rows.")
    parser.add_argument(
        "--checkerboards", type=Path, nargs="+", required=True,
        help=f"At least {MIN_VIEWS} distinct checkerboard photos, tilted in both axes.",
    )
    parser.add_argument(
        "--captures", type=Path, nargs="+", required=True,
        help="Album captures to pin to this calibration by file hash.",
    )
    parser.add_argument("--output", type=Path, required=True, help="Calibration JSON output.")
    args = parser.parse_args()
    try:
        if args.output.resolve() in {path.resolve() for path in (*args.checkerboards, *args.captures)}:
            raise ValueError("The calibration output must be separate from all source images.")
        artifact = calibration_artifact(args.cols, args.rows, args.checkerboards, args.captures)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with NamedTemporaryFile(mode="w", encoding="utf-8", dir=args.output.parent, delete=False) as stream:
                temporary = Path(stream.name)
                json.dump(artifact, stream, indent=2, allow_nan=False)
                stream.write("\n")
            temporary.replace(args.output)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    except (OSError, ValueError, cv2.error) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
