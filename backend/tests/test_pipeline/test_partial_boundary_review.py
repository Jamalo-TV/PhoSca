from pathlib import Path

import cv2
import numpy as np
import pytest

from app.pipeline import perspective
from app.pipeline.geometry import normalized_vertices_to_pixels
from app.pipeline.print_edges import refine_print_boundary, suggest_partial_corner_edge


_GOLDEN = Path(__file__).resolve().parents[3] / "data" / "golden_fixtures"


@pytest.mark.parametrize(("scale", "quality"), [(1., None), (.75, 75)])
def test_overextended_corner_offers_one_source_edge_without_changing_either_crop(monkeypatch, scale, quality):
    page = cv2.imread(str(_GOLDEN / "page_31.jpg"))
    size = np.rint(np.array(page.shape[1::-1]) * scale).astype(int)
    if scale != 1:
        page = cv2.resize(page, tuple(size), interpolation=cv2.INTER_AREA)
    if quality is not None:
        ok, jpeg = cv2.imencode(".jpg", page, [cv2.IMWRITE_JPEG_QUALITY, quality])
        assert ok
        page = cv2.imdecode(jpeg, cv2.IMREAD_COLOR)
    # Source-normalized quad from the classical fallback on the lower-right print.
    seed = [
        {"x": .58211988, "y": .31773242},
        {"x": .82161278, "y": .48862085},
        {"x": .66969597, "y": .99254411},
        {"x": .38793629, "y": .79171926},
    ]
    mask = {"source": "classical_border_quad", "detector_source": "contrast_region", "polygon": seed}
    options = dict(dual_mode=True, enable_classical_retry=True, auto_orient=False, protect_regions=False)
    with monkeypatch.context() as patch:
        patch.setattr(perspective, "suggest_partial_corner_edge", lambda *_: None)
        baseline = perspective.correct_photo_crop(page, {}, mask, **options)
    measured = perspective.correct_photo_crop(page, {}, mask, **options)

    modes = measured.metadata["crop_modes"]
    full = modes["full_print"]
    assert {mode["status"] for mode in modes.values()} == {"review_needed"}
    assert "PRINT_BOUNDARY_UNSUPPORTED" in full["reason_codes"]
    trace, = full["suggested_edge_traces"]
    assert len(trace) == 2
    points = normalized_vertices_to_pixels([(p["x"], p["y"]) for p in trace], *size)
    reference = np.float32([[1612.234, 1851.031], [995.049, 1517.384]])
    reference *= (size - 1) / np.array([2559, 1919])
    axis = reference[1] - reference[0]
    distances = (points - reference[0]) @ np.array([-axis[1], axis[0]]) / np.linalg.norm(axis)
    assert np.abs(distances).max() < 4 * scale
    assert np.linalg.norm(points[1] - points[0]) / np.linalg.norm(axis) == pytest.approx(.8, abs=.02)
    assert {key: value for key, value in full.items() if key != "suggested_edge_traces"} == (
        baseline.metadata["crop_modes"]["full_print"]
    )
    assert modes["image_only"] == baseline.metadata["crop_modes"]["image_only"]
    assert np.array_equal(measured.rectified_untrimmed, baseline.rectified_untrimmed)


@pytest.mark.parametrize("open_bottom", [False, True])
def test_partial_corner_measurement_requires_a_new_distributed_edge(open_bottom):
    paper = (195, 200, 207)
    y, x = np.mgrid[:680, :512]
    fade = np.clip((x - 350) / 135, 0, 1)
    if open_bottom:
        fade = np.maximum(fade, np.where(x > 235, np.clip((y - 525) / 135, 0, 1), 0))
    pixels = (1 - fade[..., None]) * np.array([95, 120, 145]) + fade[..., None] * np.array(paper)
    stock = np.float32([[780, 365], [1340, 677], [1000, 1310], [440, 998]])
    raster = np.float32([[0, 0], [511, 0], [511, 679], [0, 679]])
    transform = cv2.getPerspectiveTransform(raster, stock)
    page = cv2.warpPerspective(pixels.astype(np.uint8), transform, (1800, 1600), borderValue=paper)
    if open_bottom:
        stock[2] += (90, 50)
    outline, evidence, reasons = refine_print_boundary(page, stock)
    assert "PRINT_BOUNDARY_UNSUPPORTED" in reasons
    assert suggest_partial_corner_edge(page, outline, evidence) is None
