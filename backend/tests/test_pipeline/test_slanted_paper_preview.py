from pathlib import Path

import cv2
import numpy as np

from app.pipeline.crop_sampling import _outline_mask
from app.pipeline.image_ops import load_image
from app.pipeline.perspective import _neighbors_for_edge_comparison
from app.pipeline.print_outline_suggestion import suggest_slanted_print_edge_comparison


def test_real_slanted_edge_preserves_draft_centres_and_rejects_an_obscured_end():
    directory = Path(__file__).resolve().parents[3] / "data/golden_fixtures"
    page = load_image(directory / "page_05.jpg")
    stored = np.float32([[1299, 1153], [2316, 1153], [2332, 1868], [1310, 1849]])
    evidence = {"bottom": {"matched_print_rim": True, "support": .7589, "ambiguous": False}}

    candidate, (edge,) = suggest_slanted_print_edge_comparison(page, stored, evidence)
    assert np.array_equal(candidate[:2], stored[:2])
    assert np.allclose(edge, [[2331.145, 1829.781], [1310.035, 1851.223]], atol=.21)
    assert np.allclose(candidate[2:, 1] - edge[:, 1], 1., atol=.04)
    annotation = np.loadtxt(directory / "labels/page_05.txt")[3, 1:].reshape(4, 2)
    truth = annotation * (np.array(page.shape[1::-1]) - 1)
    before, after, draft = [_outline_mask(points, page.shape[:2]).astype(bool) for points in (stored, candidate, truth)]
    assert not (before & draft & ~after).any()
    assert (before & ~after & ~draft).sum() > 16_000

    a, b = edge
    axis = (b - a) / np.linalg.norm(b - a)
    normal = np.float32([-axis[1], axis[0]])
    covered = page.copy()
    cap = np.float32([b, b - 42 * axis, b - 42 * axis + 24 * normal, b + 24 * normal, b + 23 * axis])
    cv2.fillPoly(covered, [np.rint(cap).astype(np.int32)], (174, 165, 162), cv2.LINE_AA)
    assert suggest_slanted_print_edge_comparison(covered, stored, evidence) is None

    peer = np.float32([[284, 1030], [253, 1378], [264, 1730], [1309, 1733], [1309, 1159], [1288, 1022]])
    scale = np.array(page.shape[1::-1]) - 1
    mask = {
        "source": "yolo_seg_mask", "detector_source": "yolo_onnx",
        "polygon": [{"x": float(x), "y": float(y)} for x, y in peer / scale],
    }
    assert _neighbors_for_edge_comparison(page, stored, candidate, [mask]) == []
    manual = {**mask, "warp_quad": mask["polygon"]}
    assert len(_neighbors_for_edge_comparison(page, stored, candidate, [manual])) == 1
    blocked = page.copy()
    blocked[1310:1510, 1292:1303] = 25
    assert len(_neighbors_for_edge_comparison(blocked, stored, candidate, [mask])) == 1
    wider = candidate.copy()
    wider[[0, 3], 0] -= 2.
    assert len(_neighbors_for_edge_comparison(page, stored, wider, [mask])) == 1
