from pathlib import Path

import cv2
import numpy as np
import pytest

from app.config import Settings
from app.evaluation.crop_fixtures import _photograph
from app.pipeline.enhancement import prepare_for_cutout
from app.pipeline.frame_backing_review import FRAME_BACKING_COUNT_REVIEW, FRAME_BACKING_PROPOSAL
from app.pipeline.fusion_ownership import (
    BORROWED_FALLBACK_EDGE, NESTED_FALLBACK_CONTOUR, NESTED_FALLBACK_CONTOUR_REVIEW,
    deferred_neighbor_masks, nested_fallback_contour_spans,
)
from app.pipeline.image_ops import load_image
from app.pipeline.segmentation import _make_detection, fuse_candidate_evidence


_PAGE_04 = Path(__file__).resolve().parents[3] / "data" / "golden_fixtures" / "page_04.jpg"
_PAGE_26 = _PAGE_04.with_name("page_26.jpg")
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
_NESTED_MODEL = [(1273, 843), (1175, 860), (625, 860), (617, 1187), (642, 1512), (615, 1812),
                 (1302, 1787), (1302, 1274)]
_NESTED_CHILDREN = (
    [(617, 855), (1278, 848), (1181, 1249), (625, 1195)],
    [(638, 1502), (1283, 1159), (1288, 1789), (647, 1770)],
)
_NEARBY_PRINT = [(400, 142), (1390, 125), (1395, 803), (431, 817)]


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


@pytest.mark.parametrize("added_paper", [False, True])
def test_nested_fallback_contours_from_rotated_album_offer_individual_count_review(added_paper):
    source = np.ascontiguousarray(np.rot90(load_image(_PAGE_26), 2))
    encoded, jpeg = cv2.imencode(".jpg", source, [cv2.IMWRITE_JPEG_QUALITY, 75])
    assert encoded
    source = cv2.imdecode(jpeg, cv2.IMREAD_COLOR)
    height, width = source.shape[:2]
    settings = Settings(DATABASE_URL="sqlite://", REDIS_URL="redis://")

    def make(points, confidence, *, model=False):
        item = _make_detection(
            box={}, points=np.float32(points), confidence=confidence, image_width=width, image_height=height,
            settings=settings, source="yolo_seg_mask" if model else "classical_border_quad",
            detector_source="yolo_onnx" if model else "edge_contour",
        )
        assert item is not None
        return item

    parent = make(_NESTED_MODEL, .923, model=True)
    first, second = [make(quad, confidence) for quad, confidence in zip(_NESTED_CHILDREN, (.753, .759), strict=True)]
    neighbor = make(_NEARBY_PRINT, .873)
    prepared, _ = prepare_for_cutout(source)
    previous, baseline = fuse_candidate_evidence(prepared, [parent], [first, second, neighbor])
    if added_paper:
        # The same projected outline now covers a visible independent paper layer.
        claimed = np.zeros(source.shape[:2], np.uint8)
        cv2.fillConvexPoly(claimed, np.int32(_NESTED_CHILDREN[0]), 1)
        source[claimed != 0] = np.clip(source[claimed != 0].astype(np.int16) + 20, 0, 255)
    selected, evidence = fuse_candidate_evidence(
        prepared, [parent], [first, second, neighbor], ownership_source=source,
        model_evidence={0: baseline["clusters"][0]["boundary_evidence"]},
    )
    if added_paper:
        assert selected == previous and not evidence.get("deferred_print_proposals")
        return

    assert [item.mask["polygon"] for item in selected] == [neighbor.mask["polygon"], parent.mask["polygon"]]
    assert selected[1].review_reasons == [NESTED_FALLBACK_CONTOUR_REVIEW]
    assert selected[1].mask == next(item.mask for item in previous if item.mask["fusion"]["cluster_id"] == 0)
    offers = evidence["deferred_print_proposals"]
    assert [(item["id"], item["parent_cluster_id"], item["reason"]) for item in offers] == [
        (1, 0, NESTED_FALLBACK_CONTOUR), (2, 0, NESTED_FALLBACK_CONTOUR),
    ]
    assert all(len(item["polygon"]) == 4 and len(item["interior_edge_traces"][0]) == 2 for item in offers)
    assert [evidence["clusters"][index]["decision"] for index in (1, 2)] == ["deferred_for_count_review"] * 2


def test_an_oblique_one_pixel_gap_preserves_two_prints_with_inset_contours():
    height, width = 1004, 1080
    x, y = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
    noise = np.random.default_rng(408726).normal(0, 9, (50, 55)).astype(np.float32)
    texture = cv2.resize(noise, (width, height)) + 8 * np.sin(x / 55) + 5 * np.sin(y / 63)
    shading = 48 / (1 + np.exp((y - 552 + .21 * (x - 220)) / 42))
    colors = np.stack([124 + texture + shading, 103 + .8 * texture + shading, 81 + .5 * texture + .4 * shading], axis=2)
    source = np.full((height, width, 3), 221, np.uint8)
    source[111:948, 225:856] = np.uint8(np.clip(colors[111:948, 225:856], 0, 255))
    scale = np.float32([width - 1, height - 1])
    parent = np.float32([[225, 111], [855, 111], [855, 947], [225, 947]]) / scale
    children = [np.float32(points) / scale for points in (
        [[225, 127], [855, 127], [855, 458], [225, 320]],
        [[290, 698], [855, 560], [855, 947], [225, 947]],
    )]

    def jpeg75(image):
        encoded, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 75])
        assert encoded
        return cv2.imdecode(buffer, cv2.IMREAD_COLOR)

    assert len(nested_fallback_contour_spans(jpeg75(source), parent, children)) == 2
    columns = np.arange(225, 856)
    position = (columns - 540) / 315
    for bow in (0, 8):
        separated = source.copy()
        rows = np.rint(530 + .075 * (columns - 540) + bow * (1 - position ** 2)).astype(int)
        separated[rows, columns] = 221
        assert not nested_fallback_contour_spans(jpeg75(separated), parent, children)


@pytest.mark.parametrize("overprint", [None, "across_surface", "on_album"])
def test_frame_backing_crossing_is_reviewable_and_keeps_distinct_print_evidence(overprint):
    image = np.ascontiguousarray(np.rot90(load_image(_PAGE_26), 2))
    ok, jpeg = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 75])
    assert ok
    source = cv2.imdecode(jpeg, cv2.IMREAD_COLOR)
    height, width = source.shape[:2]
    settings = Settings(DATABASE_URL="sqlite://", REDIS_URL="redis://")

    def make(points, confidence, source_name, detector):
        item = _make_detection(
            box={}, points=np.float32(points), confidence=confidence, image_width=width, image_height=height,
            settings=settings, source=source_name, detector_source=detector,
        )
        assert item is not None
        return item

    model = make([(1619, 822), (1623, 1762), (1673, 1778), (2286, 1778), (2258, 801)],
                 .927, "yolo_seg_mask", "yolo_onnx")
    whole = make([(1618, 819), (2255, 810), (2256, 1744), (1631, 1767)], .883,
                 "classical_border_quad", "edge_contour")
    frame = make([(1644, 823), (2266, 799), (2181, 1919), (1249, 1919)], .612,
                 "classical_border_quad", "contrast_region")
    prepared, _ = prepare_for_cutout(source)
    previous, _ = fuse_candidate_evidence(prepared, [model], [whole, frame])
    assert len(previous) == 2
    if overprint == "across_surface":
        # A partially captured photo crosses the album edge inside the broader proposal.
        overlay = np.int32([[945, 0], [1055, 0], [1046, 545], [951, 550]])
        cv2.fillConvexPoly(source, [width - 1, height - 1] - overlay, (171, 156, 149))
    elif overprint == "on_album":
        # A pale layer below the album edge leaves the entire transverse line intact.
        overlay = np.int32([[952, 195], [1110, 195], [1080, 530], [954, 528]])
        cv2.fillConvexPoly(source, [width - 1, height - 1] - overlay, (207, 213, 225))

    selected, evidence = fuse_candidate_evidence(prepared, [model], [whole, frame], ownership_source=source)
    if overprint:
        assert selected == previous and not evidence.get("deferred_print_proposals")
        return

    parent, = selected
    assert parent.mask == next(row.mask for row in previous if row.mask["polygon"] == whole.mask["polygon"])
    assert parent.review_reasons == [FRAME_BACKING_COUNT_REVIEW]
    offer, = evidence["deferred_print_proposals"]
    assert (offer["reason"], offer["id"], offer["parent_cluster_id"]) == (FRAME_BACKING_PROPOSAL, 1, 0)
    assert offer["polygon"] == frame.mask["polygon"]
    trace, = offer["surface_edge_traces"]
    points = np.array([[p["x"], p["y"]] for p in trace]) * [width - 1, height - 1]
    np.testing.assert_allclose(points, [[2311, 1782.42], [1119, 1807.45]], atol=2)


@pytest.mark.parametrize("separate_rim", [None, "bright", "faint_tinted"])
def test_model_with_weak_score_can_anchor_a_count_review_without_releasing_neighbor_claim(separate_rim):
    source = np.ascontiguousarray(np.rot90(load_image(_PAGE_04), 3))
    ok, jpeg = cv2.imencode(".jpg", source, [cv2.IMWRITE_JPEG_QUALITY, 75])
    assert ok
    source = cv2.imdecode(jpeg, cv2.IMREAD_COLOR)
    height, width = source.shape[:2]
    settings = Settings(DATABASE_URL="sqlite://", REDIS_URL="redis://")

    def make(points, confidence, detector):
        detection = _make_detection(
            box={}, points=np.float32(points), confidence=confidence, image_width=width, image_height=height,
            settings=settings, source="yolo_seg_mask" if detector == "yolo_onnx" else "classical_border_quad",
            detector_source=detector,
        )
        assert detection is not None
        return detection

    owner = make([(98, 1605), (98, 2298), (451, 2309), (536, 2353), (1188, 2352), (1195, 1605),
                  (561, 1605), (1021, 1632), (578, 1654), (400, 1605)], .432, "yolo_onnx")
    other = make([(40, 860), (537, 852), (1171, 1866), (60, 2300)], .768, "edge_contour")
    outline = [(101, 1856), (1189, 1797), (1195, 2362), (0, 2559)]
    suspect = make(outline, .579, "contrast_region")
    prepared, _ = prepare_for_cutout(source)
    before, baseline = fuse_candidate_evidence(prepared, [owner], [other, suspect])
    if separate_rim == "bright":
        paper = np.zeros(source.shape[:2], np.uint8)
        cv2.fillConvexPoly(paper, np.int32(outline), 255)
        rim = paper - cv2.erode(paper, np.ones((3, 3), np.uint8))
        source[rim != 0] = (209, 223, 234)
    elif separate_rim == "faint_tinted":
        # A perspective-shifted reproduction with the same artwork but a
        # narrow tinted perimeter survives JPEG mainly in chroma.
        physical_outline = np.float32([[89.638, 1849.683], [1200.317, 1790.253], [1194.663, 2366.828], [0, 2559]])
        paper = np.zeros(source.shape[:2], np.uint8)
        cv2.fillConvexPoly(paper, np.rint(physical_outline * 256).astype(np.int32), 255, cv2.LINE_AA, shift=8)
        rim = paper - cv2.erode(paper, np.ones((13, 13), np.uint8))
        exposed = rim > 0
        source[exposed] = np.uint8(np.clip(source[exposed] + rim[exposed, None] / 255 * [-13.5, 0, 11.25], 0, 255))
        blur = cv2.GaussianBlur(source, (0, 0), .85)
        ok, jpeg = cv2.imencode(".jpg", blur, [cv2.IMWRITE_JPEG_QUALITY, 75])
        assert ok
        source = cv2.imdecode(jpeg, cv2.IMREAD_COLOR)

    after, evidence = fuse_candidate_evidence(prepared, [owner], [other, suspect], ownership_source=source)
    if separate_rim:
        assert after == before
        assert "deferred_print_proposals" not in evidence
        return
    assert after == [item for item in before if item.mask["fusion"]["cluster_id"] != 2]
    assert evidence["clusters"][:2] == baseline["clusters"][:2]
    assert evidence["clusters"][2]["decision"] == "deferred_for_count_review"
    offer, = evidence["deferred_print_proposals"]
    assert (offer["id"], offer["parent_cluster_id"], offer["reason"]) == (2, 0, BORROWED_FALLBACK_EDGE)
    assert offer["polygon"] == suspect.mask["polygon"]
    for key in ("borrowed_edge_traces", "interior_edge_traces", "exposed_edge_traces"):
        trace, = offer[key]
        assert len(trace) == 2 and trace[0] != trace[1]
    assert deferred_neighbor_masks({"fusion": evidence}) == [
        {key: suspect.mask[key] for key in ("source", "detector_source", "polygon")},
    ]
