import cv2
import numpy as np
import pytest

from app.config import Settings
from app.evaluation.cropping import evaluate_page_crops
from app.pipeline.album_windows import ALBUM_WINDOW_OCCLUDED, ALBUM_WINDOW_SOURCE, ALBUM_WINDOW_STOCK_UNCERTAIN, album_window_leaves
from app.pipeline.enhancement import prepare_for_cutout
from app.pipeline.perspective import correct_photo_crop
from app.pipeline.segmentation import SegmentationDetection, SegmentationResult, _refine_album_windows, segment_album_page


def _album_scene(kind: str, tilted: bool) -> tuple[np.ndarray, list[np.ndarray]]:
    image = np.full((640, 880, 3), (25, 29, 32), np.uint8)
    frame = np.float32([[0, 0], [359, 0], [359, 459], [0, 459]])
    offsets = ([[-14, -4], [0, 16], [0, 25], [6, -29]],
               [[0, 16], [14, -15], [20, -25], [0, 25]])
    decoration = np.int32([[51, 48], [308, 48], [308, 408], [51, 408]])
    motif = np.float32([[101, 98], [259, 98], [259, 358], [101, 358]])
    if kind == "arch":
        angles = np.linspace(np.pi, 2 * np.pi, 23)
        arc = np.column_stack((180 + 79 * np.cos(angles), 142 + 44 * np.sin(angles)))
        motif = np.vstack((arc, motif[2:])).astype(np.float32)
    expected = []
    for index, x in enumerate((70, 435)):
        quad = frame + np.float32([x, 81])
        if tilted:
            quad += offsets[index]
        leaf = np.full((460, 360, 3), (195, 215, 230), np.uint8)
        if kind in {"blank", "rectangle", "arch"}:
            cv2.polylines(leaf, [decoration], True, (101, 130, 146), 3, cv2.LINE_AA)
        if kind in {"print", "mat"}:
            cv2.fillConvexPoly(leaf, decoration, (241, 245, 250))
        if kind == "mat":
            cv2.rectangle(leaf, (97, 94), (259, 358), (116, 132, 142), -1)
        if kind != "blank":
            mask = cv2.fillPoly(np.zeros(leaf.shape[:2], np.uint8), [np.rint(motif).astype(np.int32)], 1)
            y, _ = np.indices(mask.shape)
            tones = 135 + 16 * np.sin(y / 33 + index)
            photograph = np.uint8(np.stack((tones - 16, tones, tones + 13), axis=2))
            cv2.ellipse(photograph, (184, 230), (38, 51), 0, 0, 360, (73, 94, 122), -1)
            leaf[mask != 0] = photograph[mask != 0]
        matrix = cv2.getPerspectiveTransform(frame, quad)
        width, height = image.shape[1::-1]
        raster = cv2.warpPerspective(leaf, matrix, (width, height))
        occupied = cv2.warpPerspective(np.ones(leaf.shape[:2], np.uint8), matrix, (width, height), flags=cv2.INTER_NEAREST)
        image[occupied != 0] = raster[occupied != 0]
        if kind != "blank":
            boundary = decoration if kind == "print" else motif
            expected.append(cv2.perspectiveTransform(np.float32(boundary)[None], matrix)[0] / [width - 1, height - 1])
    return image, expected


@pytest.mark.parametrize("kind", ["blank", "rectangle", "arch", "print"])
@pytest.mark.parametrize("tilted", [False, True])
def test_bound_album_detection_keeps_visible_prints_and_paper_separate(tmp_path, kind, tilted):
    image, expected = _album_scene(kind, tilted)
    settings = Settings(DATABASE_URL="sqlite+aiosqlite://", REDIS_URL="redis://localhost:6379/0",
                        YOLO_MODEL_PATH=tmp_path / "no-detector.onnx", VISIBLE_MASK_MODEL_PATH=None,
                        ENABLE_PRECISE_PHOTO_CROPS=True, ENABLE_CLASSICAL_SEGMENTATION_FALLBACK=True)

    result = segment_album_page(prepare_for_cutout(image)[0], settings, source_image=image)

    height, width = image.shape[:2]
    polygons = [[(p["x"], p["y"]) for p in detection.mask["polygon"]] for detection in result.detections]
    metrics = evaluate_page_crops(expected, polygons, width=width, height=height)
    assert len(result.detections) == metrics.true_positive == len(expected)
    if kind == "blank":
        assert set(result.metadata["album_window_rejections"].values()) == {"empty_album_leaf"}
        return

    assert metrics.mask_iou >= .96
    assert metrics.content_cutoff < .02 and metrics.background_retained < .035
    for detection in result.detections:
        assert detection.mask["source"] == ALBUM_WINDOW_SOURCE
        crop = correct_photo_crop(image, detection.bounding_box, detection.mask,
                                  auto_orient=False, protect_regions=False, dual_mode=True)
        reason = ALBUM_WINDOW_STOCK_UNCERTAIN if kind == "print" else ALBUM_WINDOW_OCCLUDED
        assert crop.image_only is None
        for mode in crop.metadata["crop_modes"].values():
            assert mode["status"] == "review_needed" and reason in mode["reason_codes"]


def test_two_warm_bordered_photos_without_separate_mount_evidence_are_preserved():
    image, _ = _album_scene("unmounted", tilted=False)

    assert album_window_leaves(image) == []


def _existing_detection(x1, y1, x2, y2):
    box = {"x1": x1 / 879, "y1": y1 / 639, "x2": x2 / 879, "y2": y2 / 639}
    polygon = [{"x": x / 879, "y": y / 639} for x, y in ((x1, y1), (x2, y1), (x2, y2), (x1, y2))]
    return SegmentationDetection(box, {"polygon": polygon, "source": "yolo_seg_mask"}, .96, 1., True, [])


def test_spread_is_replaced_by_its_two_apertures():
    image, _ = _album_scene("rectangle", tilted=False)
    spread = _existing_detection(70, 81, 794, 540)

    result = _refine_album_windows(image, SegmentationResult([spread], {}))

    assert len(result.detections) == 2
    assert result.metadata["album_window_rejections"] == {0: "album_spread"}


def test_small_detected_photo_on_decorated_leaf_is_preserved():
    image, _ = _album_scene("blank", tilted=False)
    image[230:298, 210:246] = (57, 86, 129)
    small = _existing_detection(210, 230, 245, 297)

    result = _refine_album_windows(image, SegmentationResult([small], {}))

    assert result.detections == [small]


def test_bright_insert_with_a_beveled_aperture_cannot_release_cardboard():
    image, _ = _album_scene("mat", tilted=False)
    result = _refine_album_windows(image, SegmentationResult([], {}))

    assert len(result.detections) == 2
    for detection in result.detections:
        crop = correct_photo_crop(image, detection.bounding_box, detection.mask,
                                  auto_orient=False, protect_regions=False, dual_mode=True)
        assert crop.image_only is None
        for mode in crop.metadata["crop_modes"].values():
            assert mode["status"] == "review_needed" and ALBUM_WINDOW_STOCK_UNCERTAIN in mode["reason_codes"]


def test_learned_neighbor_cannot_claim_fixed_window_pixels(tmp_path, monkeypatch):
    from app.pipeline import segmentation, visible_mask_refinement
    from app.pipeline.geometry import normalized_polygon_overlap

    image, _ = _album_scene("rectangle", tilted=False)
    seed = _existing_detection(40, 160, 165, 399)
    learned = segmentation._detection_polygon(_existing_detection(40, 160, 200, 399))
    initial = SegmentationResult([seed], {})
    source_result = _refine_album_windows(image, initial)
    known = [item for item in source_result.detections if item.mask["source"] == ALBUM_WINDOW_SOURCE]
    monkeypatch.setattr(segmentation, "_segment_album_page", lambda *_: initial)
    monkeypatch.setattr(visible_mask_refinement, "refine_visible_masks", lambda *_: ([learned], "a" * 64))
    settings = Settings(DATABASE_URL="sqlite+aiosqlite://", REDIS_URL="redis://localhost:6379/0",
                        VISIBLE_MASK_MODEL_PATH=tmp_path / "local.onnx", ENABLE_PRECISE_PHOTO_CROPS=True,
                        ENABLE_CLASSICAL_SEGMENTATION_FALLBACK=True)

    result = segment_album_page(image, settings, source_image=image)

    assert [item for item in result.detections if item.mask["source"] == ALBUM_WINDOW_SOURCE] == known
    neighbor, = [item for item in result.detections if item.mask["source"] != ALBUM_WINDOW_SOURCE]
    visible = neighbor.mask.get("visible_polygon") or neighbor.mask["polygon"]
    points = [(point["x"], point["y"]) for point in visible]
    assert all(normalized_polygon_overlap(points, segmentation._detection_polygon(item)).iou == 0 for item in known)
