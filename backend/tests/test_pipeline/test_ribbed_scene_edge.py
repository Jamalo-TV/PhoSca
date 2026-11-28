from copy import deepcopy

import cv2
import numpy as np
import pytest

from app.pipeline.near_capture_review import quarter_turn
from app.pipeline.perspective import correct_photo_crop
from app.pipeline.projective_prints import PROJECTIVE_PRINT_OUTLINE_REVIEW
from app.pipeline.ribbed_scene_edge import RIBBED_SCENE_EDGE, measure_ribbed_scene_edge, ribbed_scene_edge_review
from app.pipeline.tile_print_review import CONTESTED_TILE_FOLLOWUP, TILED_PRINT_PROPOSAL


def _photo_on_ribbed_backing(case="separate"):
    height, width = 620, 610
    y = np.arange(height, dtype=np.float32)[:, None]
    lab = np.empty((height, width, 3), dtype=np.uint8)
    lab[:] = [207, 128, 143]
    lab[..., 0] = 207 + 26 * np.cos(2 * np.pi * y / 11)
    if case == "remote_pattern":
        lab[:, 442:, 0] = 207 + 26 * np.cos(2 * np.pi * y / 7)
    right = 414 if case == "two_transitions" else 432
    scene = lab[139:434, 158:right]
    scene[:] = [82, 136, 124]
    scene[..., 0] = 82 + np.random.default_rng(34).integers(-34, 35, size=scene.shape[:2])
    if case == "flat_rim":
        lab[139:434, 141:158] = [212, 128, 143]
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR), np.float32([[140, 139], [432, 139], [432, 433], [140, 433]])


@pytest.mark.parametrize("turn", range(4))
def test_open_scene_edge_rotates_with_native_source(turn):
    image, quad = _photo_on_ribbed_backing()
    scale = np.float32(image.shape[1::-1]) - 1
    rotated = np.ascontiguousarray(np.rot90(image, turn))
    rotated_scale = np.float32(rotated.shape[1::-1]) - 1
    evidence = measure_ribbed_scene_edge(rotated, quarter_turn(quad / scale, turn) * rotated_scale)

    assert evidence is not None
    trace, = evidence.spans
    original = quarter_turn(trace / rotated_scale, (-turn) % 4) * scale
    np.testing.assert_allclose(original[:, 0], [158, 158], atol=1.)
    assert all(168 < point[1] < 404 for point in original)
    assert np.linalg.norm(original[0] - original[1]) > 215
    assert len(evidence.scopes) == 3


@pytest.mark.parametrize("case", ["flat_rim", "remote_pattern", "two_transitions"])
def test_open_scene_edge_requires_distinct_and_matched_backing_witnesses(case):
    image, quad = _photo_on_ribbed_backing(case)

    assert measure_ribbed_scene_edge(image, quad) is None


@pytest.mark.parametrize("case", ["recovered", "neighbor", "deferred", "flat_backing"])
def test_recovered_ribbed_side_changes_the_held_raster_only_with_clear_native_evidence(case):
    height, width = 620, 610
    y = np.arange(height, dtype=np.float32)[:, None]
    lab = np.full((height, width, 3), [207, 128, 143], dtype=np.uint8)
    if case != "flat_backing":
        lab[..., 0] = 207 + 35 * np.cos(2 * np.pi * y / 11)
    scene = lab[139:434, 158:432]
    scene[:] = [82, 136, 124]
    scene[..., 0] = 82 + np.random.default_rng(34).integers(-24, 25, size=scene.shape[:2])
    image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    scale = np.float32([width - 1, height - 1])

    def polygon(points):
        return [{"x": float(x), "y": float(y)} for x, y in np.float32(points) / scale]

    visible = [[158, 139], [432, 139], [432, 433], [158, 433]]
    seed = [[158, 139], [432.2, 139], [418, 433], [158, 433]]
    claim = {"polygon": polygon([[434, 247], [448, 247], [448, 306], [434, 306]])}
    mask = {"source": "yolo_seg_mask", "detector_source": "yolo_onnx",
            "polygon": polygon(seed), "visible_polygon": polygon(visible)}
    options = dict(dual_mode=True, auto_orient=False, protected_regions=[],
                   neighbor_masks=[claim] if case == "neighbor" else [],
                   deferred_print_proposals=[claim] if case == "deferred" else [])
    before = correct_photo_crop(image, {}, mask, **options)
    after = correct_photo_crop(image, {}, mask, enable_classical_retry=True, **options)

    if case != "recovered":
        assert before.metadata["crop_modes"] == after.metadata["crop_modes"]
        np.testing.assert_array_equal(before.image, after.image)
        return
    assert after.metadata["boundary_refinement"]["right"]["review_ribbed_side_fit"]
    before_modes, after_modes = before.metadata["crop_modes"], after.metadata["crop_modes"]
    assert all(mode["status"] == "review_needed" and "PRINT_BOUNDARY_UNSUPPORTED" in mode["reason_codes"]
               for mode in after_modes.values())
    assert after.image_only is None and before.image_only is None
    original, candidate = [np.float32([(p["x"], p["y"]) for p in modes["full_print"]["warp_quad"]]) * scale
                           for modes in (before_modes, after_modes)]
    np.testing.assert_allclose(candidate[[0, 3]], original[[0, 3]], atol=.001)
    np.testing.assert_allclose(candidate[1:3, 0], [431, 431], atol=.25)
    assert candidate[2, 0] - original[2, 0] > 12
    assert after_modes["full_print"]["sampling"] in {"owned_bilinear", "owned_polygon_bilinear"}


@pytest.mark.parametrize("case", [
    "exact", "generic_proposal", "different_source", "different_corners", "released_mode",
    "competing_hint", "claimed_remote_witness", "protected_local_witness",
])
def test_open_scene_edge_review_is_bound_to_the_held_source(case):
    image, quad = _photo_on_ribbed_backing()
    scale = np.float32(image.shape[1::-1]) - 1

    def polygon(points):
        return [{"x": float(x), "y": float(y)} for x, y in np.float32(points) / scale]

    outline = polygon(quad)
    sha = "1" * 64
    provenance = {
        "deferred_print_proposal": {
            "id": 3, "reason": TILED_PRINT_PROPOSAL, "source": CONTESTED_TILE_FOLLOWUP, "polygon": outline,
        },
        "manual_creation": {"deferred_proposal_id": 3, "confirm_standalone_print": True, "warp_quad": outline},
        "prediction_provenance": {"page_sha256": sha},
    }
    mask = {"manual_added": True, "warp_quad": outline}
    modes = {"full_print": {"status": "review_needed", "reason_codes": [PROJECTIVE_PRINT_OUTLINE_REVIEW], "warp_quad": outline},
             "image_only": {"status": "review_needed", "reason_codes": [PROJECTIVE_PRINT_OUTLINE_REVIEW]}}
    claims = []
    protected = []
    if case == "generic_proposal":
        provenance["deferred_print_proposal"].pop("source")
    elif case == "different_source":
        sha = "2" * 64
    elif case == "different_corners":
        mask["warp_quad"] = polygon(quad + [1, 0])
    elif case == "released_mode":
        modes["image_only"]["status"] = "ready"
    elif case == "competing_hint":
        modes["full_print"]["suggested_edge_traces"] = [outline[:2]]
    elif case == "claimed_remote_witness":
        claims = [{"source_mask": {"visible_polygon": polygon([[444, 240], [449, 240], [449, 265], [444, 265]])}}]
    elif case == "protected_local_witness":
        protected = [{"x1": 154, "y1": 220, "x2": 166, "y2": 278}]
    before = deepcopy(modes)

    result = ribbed_scene_edge_review(image, mask, provenance, sha, modes, claims, protected)

    assert modes == before
    if case == "exact":
        assert result["suggested_edge_source"] == RIBBED_SCENE_EDGE
        assert len(result["suggested_edge_traces"]) == 1
        assert result.keys() == {"suggested_edge_source", "suggested_edge_traces", "ribbed_scene_edge_scope"}
    else:
        assert result is None
