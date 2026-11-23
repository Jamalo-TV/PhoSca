import json
import sys

import cv2
import numpy as np
import pytest
from PIL import Image

from app.pipeline.image_ops import sha256_file
from scripts.calibrate_crop_camera import checkerboard_points, fit_camera, main


SIZE = (1280, 960)
CAMERA = np.array([[1020.0, 0.0, 637.0], [0.0, 1010.0, 482.0], [0.0, 0.0, 1.0]])
POSES = (
    ((.20, .24), (-.25, .21, .04), 24),
    ((.80, .24), (.19, -.25, -.10), 22),
    ((.20, .76), (.26, .15, .03), 25),
    ((.80, .76), (-.19, -.16, .08), 23),
    ((.50, .29), (.30, .07, -.11), 19),
    ((.50, .72), (-.12, .28, .04), 21),
    ((.50, .50), (.06, -.16, .03), 22),
    ((.30, .50), (-.30, -.20, .04), 20),
)


def _project(grid, k1=0.0, centered=False):
    for center, rotation, distance in POSES:
        ray = (np.array((.5, .5) if centered else center) * SIZE - CAMERA[:2, 2]) / CAMERA.diagonal()[:2]
        rotation = np.array(rotation)
        translation = np.r_[ray, 1.0] * distance - cv2.Rodrigues(rotation)[0] @ grid.mean(axis=0)
        distortion = np.array([k1, 0.0, 0.0, 0.0, 0.0])
        corners, _ = cv2.projectPoints(grid, rotation, translation, CAMERA, distortion)
        yield corners.reshape(-1, 2)


def test_camera_fit_recovers_known_intrinsics():
    grid = checkerboard_points(9, 6)
    camera, diagnostics = fit_camera(grid, list(_project(grid)), SIZE)
    np.testing.assert_allclose(camera, CAMERA, rtol=0, atol=.002)
    assert diagnostics["max_sampled_distortion_px"] < .002
    assert max(view["max_corner_error_px"] for view in diagnostics["held_out_views"]) < .002
    assert max(view["intrinsic_drift_px"] for view in diagnostics["held_out_views"]) < .002


def test_camera_fit_rejects_distortion_and_narrow_coverage():
    grid = checkerboard_points(9, 6)
    with pytest.raises(ValueError, match="Lens distortion"):
        fit_camera(grid, list(_project(grid, k1=.025)), SIZE)
    with pytest.raises(ValueError, match="all four image corners"):
        fit_camera(grid, list(_project(grid, centered=True)), SIZE)


def test_calibration_cli_detects_the_board_and_registers_decoded_capture(tmp_path, monkeypatch):
    columns, rows = 9, 6
    grid = checkerboard_points(columns, rows)
    square = 96
    board = np.full(((rows + 3) * square, (columns + 3) * square), 255, np.uint8)
    for row in range(rows + 1):
        for column in range((row + 1) % 2, columns + 1, 2):
            board[(row + 1) * square:(row + 2) * square, (column + 1) * square:(column + 2) * square] = 0
    outer_corners = [0, columns - 1, len(grid) - 1, len(grid) - columns]
    source = (grid[outer_corners, :2] + 2) * square - .5
    checkerboards = []
    for index, projected in enumerate(_project(grid)):
        # Render at 3x resolution so the reference uses consistent pixel centers.
        homography = cv2.getPerspectiveTransform(source, (projected[outer_corners] + .5) * 3 - .5)
        large = cv2.warpPerspective(board, homography, (SIZE[0] * 3, SIZE[1] * 3), borderValue=155)
        image = cv2.resize(large, SIZE, interpolation=cv2.INTER_AREA)
        path = tmp_path / f"checkerboard-{index}.png"
        assert cv2.imwrite(str(path), image)
        checkerboards.append(path)

    capture = tmp_path / "album.jpg"
    portrait = Image.new("RGB", SIZE[::-1])
    exif = portrait.getexif()
    exif[274] = 6
    portrait.save(capture, exif=exif)
    output = tmp_path / "calibration.json"
    monkeypatch.setattr(sys, "argv", [
        "calibrate_crop_camera.py", "--cols", str(columns), "--rows", str(rows),
        "--checkerboards", *(str(path) for path in checkerboards),
        "--captures", str(capture), "--output", str(output),
    ])
    main()

    artifact = json.loads(output.read_text(encoding="utf-8"))
    assert artifact["schema_version"] == 1
    digest = sha256_file(capture)
    assert set(artifact["captures"]) == {digest}
    record = artifact["captures"][digest]
    assert (record["width"], record["height"]) == SIZE
    assert len(record["calibration_id"]) == 64
    np.testing.assert_allclose(record["camera_matrix"], CAMERA, rtol=0, atol=1.0)
