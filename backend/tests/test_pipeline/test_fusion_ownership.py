from pathlib import Path

import cv2
import numpy as np
import pytest

from app.config import Settings
from app.evaluation.crop_fixtures import _photograph
from app.pipeline.enhancement import prepare_for_cutout
from app.pipeline.image_ops import load_image
from app.pipeline.segmentation import _make_detection, fuse_candidate_evidence


_PAGE_04 = Path(__file__).resolve().parents[3] / "data" / "golden_fixtures" / "page_04.jpg"
# Native proposals captured from the mixed development checkpoint and the
# classical detector. The image evidence, not these coordinates, decides ownership.
_MODEL_OUTLINES = (
    [(832, 236), (832, 1381), (868, 1368), (872, 1096), (886, 1378), (1593, 1381),
     (1612, 1362), (1630, 1381), (1622, 238)],
    [(1594, 718), (1594, 1379), (1614, 1387), (1594, 1815), (2300, 1803), (2308, 1611),
     (2355, 1445), (2352, 734), (1668, 718), (1627, 736)],
    [(83, 746), (148, 1876), (857, 1865), (863, 740)],
)
_FALLBACK_OUTLINES = (
    [(1797, 730), (2362, 724), (2559, 1919), (1855, 1818)],
    [(130, 1863), (1620, 729), (2356, 853), (2299, 1860)],
)


def _proposals(image, turn):
    height, width = image.shape[:2]
    settings = Settings(DATABASE_URL="sqlite://", REDIS_URL="redis://")

    def make(outline, confidence, source, detector):
        points = np.float32(outline)
        if turn:
            points = np.column_stack((points[:, 1], height - 1 - points[:, 0]))
        item = _make_detection(
            box={}, points=points, confidence=confidence, image_width=width, image_height=height,
            settings=settings, source=source, detector_source=detector,
        )
        assert item is not None, detector
        return item

    models = [make(quad, confidence, "yolo_seg_mask", "yolo_onnx")
              for quad, confidence in zip(_MODEL_OUTLINES, (.735, .862, .919), strict=True)]
    classical = [make(quad, confidence, "classical_border_quad", family)
                 for quad, confidence, family in zip(_FALLBACK_OUTLINES, (.576, .701),
                                                     ("contrast_region", "edge_contour"), strict=True)]
    return models, classical


@pytest.mark.parametrize("turn", [0, 1])
def test_page_04_fallback_outlines_are_uncounted_but_preserved_for_review(turn):
    original = np.ascontiguousarray(np.rot90(load_image(_PAGE_04), turn))
    prepared, _ = prepare_for_cutout(original)
    models, classical = _proposals(original, turn)

    selected, evidence = fuse_candidate_evidence(prepared, models, classical, ownership_source=original)

    assert len(selected) == 3
    assert all(model.mask["polygon"] in [item.mask["polygon"] for item in selected] for model in models)
    assert [item["reason"] for item in evidence["deferred_print_proposals"]] == [
        "shared_confirmed_edge", "crossing_modeled_prints",
    ]
    assert [item["id"] for item in evidence["deferred_print_proposals"]] == [3, 4]
    assert all(len(item["polygon"]) == 4 for item in evidence["deferred_print_proposals"])


@pytest.mark.parametrize("outline_index", [0, 1])
def test_genuine_print_with_same_overlap_keeps_classical_candidate(outline_index):
    original = load_image(_PAGE_04)
    prepared, _ = prepare_for_cutout(original)
    models, classical = _proposals(original, 0)
    # Keep the modeled outlines and their prior evidence, then give the
    # classical footprint its own physical boundary and photographic texture.
    _, baseline = fuse_candidate_evidence(prepared, models, classical)
    model_evidence = {index: entry["boundary_evidence"] for index, entry in enumerate(baseline["clusters"][:3])}
    quad = np.float32(_FALLBACK_OUTLINES[outline_index])
    paper = np.full((420, 512, 3), 228, dtype=np.uint8)
    paper[15:-15, 15:-15] = _photograph(390, 482, np.random.default_rng(2819), white=False)
    transform = cv2.getPerspectiveTransform(np.float32([[0, 0], [511, 0], [511, 419], [0, 419]]), quad)
    height, width = original.shape[:2]
    stock = cv2.warpPerspective(paper, transform, (width, height))
    footprint = cv2.warpPerspective(np.ones(paper.shape[:2], np.uint8), transform, (width, height), flags=cv2.INTER_NEAREST)
    original[footprint != 0] = stock[footprint != 0]

    selected, evidence = fuse_candidate_evidence(
        prepare_for_cutout(original)[0], models, [classical[outline_index]],
        model_evidence=model_evidence, ownership_source=original,
    )

    assert len(selected) == 4
    assert not evidence.get("deferred_print_proposals")
