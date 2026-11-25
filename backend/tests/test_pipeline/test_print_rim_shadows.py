import cv2
import numpy as np
import pytest

from app.evaluation.cropping import evaluate_rendered_crop
from app.pipeline.geometry import normalized_vertices_to_pixels
from app.pipeline.perspective import correct_photo_crop
from app.pipeline.print_edges import PRINT_SHADOW_EDGE_REVIEW


_PAPER = np.float32([[221, 212], [1060, 212], [1060, 850], [221, 850]])
_MOTIF = _PAPER + np.float32([[48, 48], [-48, 48], [-48, -48], [48, -48]])


def _scan(shadow=None, *, warm_shadow=False):
    y, x = np.indices((1040, 1400), dtype=np.float32)
    distances = np.stack([y - 212, 1060 - x, 850 - y, x - 221])
    depth = distances.min(axis=0)
    paper = depth >= 0
    motif = depth >= 48
    lab = np.empty((*depth.shape, 3), np.float32)
    lab[:] = (251, 128, 128)
    lab[paper] = (219, 129, 135) if shadow else (184, 128, 128)
    texture = 17 * np.sin(x / 19) * np.cos(y / 24) * np.clip((depth - 48) / 20, 0, 1)
    lab[motif, 0] = (180 if shadow else 226) + texture[motif]
    if shadow:
        amount, width, center = shadow
        shade = np.where(distances < 0, amount * np.exp(-.5 * ((distances - center) / width)**2), 0).max(axis=0)
        lab[~paper, 0] -= shade[~paper]
        if warm_shadow:
            lab[~paper, 1:] += shade[~paper, None] * np.float32([.005, .09])
    lab[..., 0] += np.random.default_rng(8317).normal(0, .45, depth.shape)
    image = cv2.cvtColor(np.rint(lab).astype(np.uint8), cv2.COLOR_LAB2BGR)
    return cv2.GaussianBlur(image, (0, 0), .8), {"full_print": paper, "image_only": motif}


def _crop(image, seed):
    scale = np.float32(image.shape[1::-1]) - 1
    polygon = [{"x": float(x), "y": float(y)} for x, y in seed / scale]
    return correct_photo_crop(
        image, {}, {"polygon": polygon}, dual_mode=True, auto_orient=False, protect_regions=False,
        print_aspect_ratio=839 / 638,
    )


@pytest.mark.parametrize(("shadow", "warm_shadow"), [((65, 7, -3), False), ((70, 8, -4), True)])
def test_broad_shadow_cannot_take_ownership_of_the_outer_print(shadow, warm_shadow):
    image, truth = _scan(shadow, warm_shadow=warm_shadow)
    crop = _crop(image, _PAPER)
    assert any(side["outward_search"].get("shadow_shoulder_support", 0) >= .70
               for side in crop.metadata["boundary_refinement"].values())
    for mode, target, expected in (("full_print", crop.image, _PAPER), ("image_only", crop.image_only, _MOTIF)):
        evidence = crop.metadata["crop_modes"][mode]
        assert evidence["status"] == "review_needed" and PRINT_SHADOW_EDGE_REVIEW in evidence["reason_codes"]
        vertices = [(point["x"], point["y"]) for point in evidence["warp_quad"]]
        corners = normalized_vertices_to_pixels(vertices, *image.shape[1::-1])
        assert np.max(np.linalg.norm(corners - expected, axis=1)) < 1
        metrics = evaluate_rendered_crop(truth[mode], vertices, evidence["page_to_output"], target.shape[:2],
                                         sampling=evidence["sampling"])
        assert metrics["foreign_source_samples"] == metrics["cutoff_beyond_1px"] == 0


def test_a_dark_neutral_paper_rim_surrounding_a_bright_motif_still_recovers():
    image, _ = _scan()
    crop = _crop(image, _MOTIF)
    assert all(mode["status"] == "ready" for mode in crop.metadata["crop_modes"].values())
    assert all(side["outward_search"]["print_rim_supported"] for side in crop.metadata["boundary_refinement"].values())
    assert crop.image.shape[:2] == (639, 840)
