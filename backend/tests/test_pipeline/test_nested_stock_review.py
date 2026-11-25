from dataclasses import replace

import cv2
import numpy as np
import pytest

from app.config import Settings
from app.pipeline import segmentation
from app.pipeline.crop_sampling import OWNED_BILINEAR
from app.pipeline.geometry import normalized_vertices_to_pixels
from app.pipeline.multipanel_prints import GRID_OUTLINE_REVIEW_SOURCE, MULTIPANEL_PRINT_REVIEW, nested_print_on_shared_stock
from app.pipeline.perspective import _proposal_polygon, apply_refined_visibility, correct_photo_crop
from app.pipeline.print_edges import PRINT_EDGE_CONFIRMATION


_OUTER = np.float32([[78, 75], [629, 75], [629, 477], [78, 477]])
_INNER = np.float32([[119, 115], [588, 115], [588, 436], [119, 436]])


def _contact_sheet(stock: tuple[int, int, int] = (223, 227, 230)) -> np.ndarray:
    page = np.full((550, 713, 3), (91, 106, 120), np.uint8)
    cv2.fillConvexPoly(page, _OUTER.astype(np.int32), stock)
    cv2.fillConvexPoly(page, _INNER.astype(np.int32), (42, 42, 42))
    for row in range(3):
        for column in range(4):
            left, top = 126 + column * 115, 123 + row * 104
            color = 105 + 20 * (row + column)
            cv2.rectangle(page, (left, top), (left + 108, top + 95), (color, color, color), -1)
    return page


def test_link_requires_enclosed_grid_on_consistent_pale_stock():
    page = _contact_sheet()
    assert nested_print_on_shared_stock(page, _OUTER, _INNER)
    assert not nested_print_on_shared_stock(page, _INNER, _INNER)
    assert not nested_print_on_shared_stock(_contact_sheet((103, 141, 171)), _OUTER, _INNER)


@pytest.mark.parametrize("shift", [np.float32([510, 0]), np.float32([540, 45])])
def test_independent_adjacent_print_does_not_link(shift):
    page = np.pad(_contact_sheet(), ((0, 0), (0, 520), (0, 0)), constant_values=100)
    assert not nested_print_on_shared_stock(page, _OUTER, _INNER + shift)


@pytest.mark.parametrize("shift", [np.float32([20, 0]), np.float32([0, 20])])
def test_enclosed_quad_needs_parallel_edges_in_both_axes(shift):
    inner = _INNER.copy()
    inner[2] += shift
    assert not nested_print_on_shared_stock(_contact_sheet(), _OUTER, inner)


def test_projected_grid_keeps_a_matching_stock_link():
    image = _contact_sheet()
    page_quad = np.float32([[0, 0], [712, 0], [712, 549], [0, 549]])
    perspective_quad = np.float32([[7, 8], [695, 26], [702, 530], [30, 540]])
    transform = cv2.getPerspectiveTransform(page_quad, perspective_quad)
    warped = cv2.warpPerspective(image, transform, image.shape[1::-1])
    outer, inner = cv2.perspectiveTransform(np.stack((_OUTER, _INNER)), transform)

    assert nested_print_on_shared_stock(warped, outer, inner)


@pytest.mark.parametrize("blur", [0, .85])
def test_distributed_paper_shadow_keeps_an_inset_print_out_of_the_link(blur):
    page = _contact_sheet()
    layer = np.ones(page.shape[:2], np.uint8)
    cv2.fillConvexPoly(layer, _INNER.astype(np.int32), 0)
    distance = cv2.distanceTransform(layer, cv2.DIST_L2, 5)
    shadow = 9 * (.5 + .5 * np.cos(np.minimum(distance / 22, 1) * np.pi)) * layer
    separate = np.uint8(np.clip(page - shadow[:, :, None], 0, 255))
    if blur:
        separate = cv2.GaussianBlur(separate, (0, 0), blur)

    assert not nested_print_on_shared_stock(separate, _OUTER, _INNER)


def test_already_held_contour_can_be_a_parent_but_not_a_fragment():
    page = _contact_sheet()
    height, width = page.shape[:2]
    settings = Settings(DATABASE_URL="sqlite://", REDIS_URL="redis://")
    parent, child = [
        segmentation._make_detection(
            box={}, points=quad, confidence=.9, image_width=width, image_height=height, settings=settings,
            source="classical_border_quad", detector_source="edge_contour",
        ) for quad in (_OUTER, _INNER)
    ]

    def hold(item):
        return replace(item, mask={**item.mask, "paper_edge_confirmation": True},
                       review_reasons=[PRINT_EDGE_CONFIRMATION])

    child_held = segmentation.SegmentationResult([parent, hold(child)], {})
    assert segmentation._defer_nested_print_candidates(page, child_held) is child_held
    parent_held = segmentation.SegmentationResult([hold(parent), child], {})
    retained, = segmentation._defer_nested_print_candidates(page, parent_held).detections
    assert retained.review_reasons == [MULTIPANEL_PRINT_REVIEW, PRINT_EDGE_CONFIRMATION]


def _partial_grid_page(layout):
    height, width = 1140, 900
    page = np.full((height, width, 3), (88, 96, 102), np.uint8)
    page[50:1090, 50:850] = (214, 217, 219)
    page[80:1088, 118:782] = (45, 49, 47)
    rails = (89, 195, 229, 325, 359, 463, 499, 594, 630, 731, 768, 863, 901, 1006)
    for row in rails:
        if layout == "broken_cadence" and row > 480:
            continue
        for x in range(124, 773, 17):
            cv2.rectangle(page, (x, row + 46), (x + 6, row + 54), (236, 238, 239), -1)

    outer = np.float32([[50, 50], [849, 50], [849, 1089], [50, 1089]])
    inner = np.float32([[109, 534], [492, 533], [791, 535], [793, 782],
                        [792, 1083], [395, 1083], [106, 1082]])
    if layout == "second_rim":
        page[535:544, 109:791] = (214, 217, 219)
    elif layout in {"raised_print", "top_shadow", "side_shadow"}:
        shadow = np.zeros(page.shape[:2], np.uint8)
        cv2.fillConvexPoly(shadow, cv2.convexHull(np.rint(inner).astype(np.int32)), 1)
        distance = cv2.distanceTransform(1 - shadow, cv2.DIST_L2, 5)
        reach = 10 if layout == "top_shadow" else 25
        shade = 20 * (distance > 0) * (.5 + .5 * np.cos(np.pi * np.minimum(1, distance / reach)))
        if layout == "top_shadow":
            shade[534:] = 0
        elif layout == "side_shadow":
            shade[:, 110:] = 0
        page = np.uint8(np.clip(page - shade[..., None], 0, 255))
    elif layout == "perspective_jpeg":
        aligned = np.float32([[80, 61], [838, 32], [860, 1088], [40, 1060]])
        transform = cv2.getPerspectiveTransform(outer, aligned)
        page = cv2.warpPerspective(page, transform, (width, height))
        inner = cv2.perspectiveTransform(inner[None], transform)[0]
        outer = aligned
        ok, jpeg = cv2.imencode(".jpg", page, [cv2.IMWRITE_JPEG_QUALITY, 75])
        assert ok
        page = cv2.rotate(cv2.imdecode(jpeg, cv2.IMREAD_COLOR), cv2.ROTATE_90_CLOCKWISE)
        outer, inner = [np.column_stack((height - 1 - quad[:, 1], quad[:, 0])) for quad in (outer, inner)]
        height, width = page.shape[:2]

    return page, outer, inner


@pytest.mark.parametrize("layout", [
    "continuous", "perspective_jpeg", "second_rim", "raised_print", "top_shadow", "side_shadow", "broken_cadence",
])
def test_partial_grid_count_requires_original_pixel_evidence(layout):
    page, outer, inner = _partial_grid_page(layout)
    height, width = page.shape[:2]
    settings = Settings(DATABASE_URL="sqlite://", REDIS_URL="redis://")
    detections = []
    for points, source, detector, score in (
        (outer, "classical_border_quad", "edge_contour", .89),
        (inner, "yolo_seg_mask", "yolo_onnx", .54),
    ):
        item = segmentation._make_detection(
            box={}, points=points, confidence=.89, image_width=width, image_height=height,
            settings=settings, source=source, detector_source=detector,
        )
        item.mask["boundary_evidence"] = {"score": score}
        detections.append(item)

    proposals = segmentation.SegmentationResult(detections, {})
    result = segmentation._defer_nested_print_candidates(page, proposals)
    if layout in {"continuous", "perspective_jpeg"}:
        photo, = result.detections
        assert photo.mask["polygon"] == detections[0].mask["polygon"]
        assert len(photo.mask["nested_print_suggestion"]) == 4
        assert photo.review_reasons == [MULTIPANEL_PRINT_REVIEW]
        assert result.metadata["nested_print_count_suggestions"] == 1
    else:
        assert result is proposals


def _grid_with_weak_outer_proposal(settings, cut_edge=False):
    page, outer, inner = _partial_grid_page("continuous")
    paper = np.full(page.shape[:2], 0, np.uint8)
    cv2.fillConvexPoly(paper, outer.astype(np.int32), 1)
    page[paper == 0] = (111, 150, 185)
    page[1066:1090, 50:850] = (214, 217, 219)
    if cut_edge:
        cv2.line(page, tuple(inner[0].astype(int)), tuple(inner[2].astype(int)), (214, 217, 219), 2)
    seed = np.float32([[85, 61], [830, 107], [832, 1084], [79, 1084]])
    height, width = page.shape[:2]
    detections = []
    for points, source, detector_source, score in (
        (seed, "classical_border_quad", "contrast_region", .56),
        (inner, "yolo_seg_mask", "yolo_onnx", .64),
    ):
        item = segmentation._make_detection(
            box={}, points=points, confidence=.88, image_width=width, image_height=height, settings=settings,
            source=source, detector_source=detector_source,
        )
        item.mask["boundary_evidence"] = {"score": score}
        detections.append(item)
    return page, seed, segmentation.SegmentationResult(detections, {})


@pytest.mark.parametrize("cut_edge", [False, True])
def test_recovering_grid_outline_retains_the_crop_seed_and_checks_thin_separating_edges(cut_edge):
    settings = Settings(DATABASE_URL="sqlite://", REDIS_URL="redis://")
    page, seed, proposals = _grid_with_weak_outer_proposal(settings, cut_edge)
    height, width = page.shape[:2]
    detections = proposals.detections
    result = segmentation._defer_nested_print_candidates(page, proposals)
    if cut_edge:
        assert result is proposals
        return

    photo, = result.detections
    assert photo.mask["visible_polygon_source"] == GRID_OUTLINE_REVIEW_SOURCE
    assert photo.review_reasons == [MULTIPANEL_PRINT_REVIEW]
    assert photo.mask["polygon"] == detections[0].mask["polygon"]
    suggestion, = result.metadata["fusion"]["deferred_print_proposals"]
    assert suggestion["reason"] == "shared_print_grid" and len(suggestion["polygon"]) == 4
    assert np.allclose(_proposal_polygon(photo.mask, width, height), seed)
    assert apply_refined_visibility(photo.mask, {}) == photo.mask
    options = dict(dual_mode=True, auto_orient=False, protect_regions=False)
    baseline = correct_photo_crop(page, photo.bounding_box, detections[0].mask, **options)
    review = correct_photo_crop(page, photo.bounding_box, photo.mask, **options)
    assert np.array_equal(review.rectified_untrimmed, baseline.rectified_untrimmed)
    assert review.metadata["warp_quad"] == baseline.metadata["warp_quad"]
    modes = review.metadata["crop_modes"]
    assert modes["full_print"]["suggested_warp_quad"] == photo.mask["visible_polygon"]
    assert all(mode["status"] == "review_needed" and PRINT_EDGE_CONFIRMATION in mode["reason_codes"]
               for mode in modes.values())
    outline_preview = modes["full_print"]["outline_preview"]
    preview_height, preview_width = review.full_print_preview.shape[:2]
    assert outline_preview["sampling"] == OWNED_BILINEAR
    assert outline_preview["shape"] == {"height": preview_height, "width": preview_width}
    assert review.image_only_preview is baseline.image_only_preview is None
    source_quad = normalized_vertices_to_pixels(
        [[point["x"], point["y"]] for point in outline_preview["source_outline"]], width, height,
    )
    projected = cv2.perspectiveTransform(source_quad[None], np.asarray(outline_preview["page_to_output"]))[0]
    assert np.allclose(projected, [[0, 0], [preview_width - 1, 0],
                                  [preview_width - 1, preview_height - 1], [0, preview_height - 1]], atol=.01)

    # A reviewer can restore a deferred candidate as a neighboring print.
    disputed = correct_photo_crop(page, photo.bounding_box, photo.mask, neighbor_masks=[detections[1].mask], **options)
    full_mode = disputed.metadata["crop_modes"]["full_print"]
    assert full_mode["status"] == "review_needed"
    assert full_mode["suggested_warp_quad"] == photo.mask["visible_polygon"]
    assert disputed.full_print_preview is None and "outline_preview" not in full_mode


async def test_restoring_grid_fragment_invalidates_the_overlapping_review_png(client, monkeypatch):
    from uuid import uuid4

    from app.config import get_settings
    from app.services import pipeline

    monkeypatch.setenv("AUTO_ANALYZE_UPLOADS", "true")
    monkeypatch.setenv("MAX_UPLOAD_SIZE", "1048576")
    monkeypatch.setenv("MAX_REQUEST_SIZE", "2097152")
    monkeypatch.setenv("ENABLE_PHOTO_ORIENTATION_CORRECTION", "false")
    monkeypatch.setenv("ENABLE_PROTECTED_REGION_GUARD", "false")
    get_settings.cache_clear()
    page, _, proposals = _grid_with_weak_outer_proposal(get_settings())
    monkeypatch.setattr(segmentation, "_segment_album_page", lambda *_: proposals)
    monkeypatch.setattr(pipeline, "run_paddle_ocr", lambda *_: [])
    album = await client.post("/api/v1/albums", json={"name": "Kontaktabzug"})
    assert album.status_code == 201, album.text
    album_id = album.json()["id"]
    ok, payload = cv2.imencode(".png", page)
    assert ok
    upload = await client.post(f"/api/v1/albums/{album_id}/pages/upload", files={
        "files": ("album.png", payload.tobytes(), "image/png"),
    })
    assert upload.status_code == 201, upload.text
    page_id = upload.json()["pages"][0]["page_id"]
    queue, = (await client.get("/api/v1/review/queue", params={"type": "segmentation", "album_id": album_id})).json()
    count = queue["payload"]
    assert count["detection_count"] == 1
    proposal, = count["deferred_print_proposals"]
    url = f"/api/v1/photos/{count['detections'][0]['photo_id']}"
    photo = (await client.get(url)).json()
    preview = await client.get(photo["urls"]["full_print_preview"])
    assert preview.status_code == 200 and preview.headers["content-type"] == "image/png"
    original = (await client.get(photo["urls"]["source_corrected_untrimmed"])).content

    add = await client.post(f"/api/v1/pages/{page_id}/manual-photos", json={
        "request_id": str(uuid4()), "warp_quad": proposal["polygon"], "deferred_proposal_id": proposal["id"],
        "photo_count_revision": count["photo_count_revision"],
    })
    assert add.status_code == 201, add.text
    for reprocess in (False, True):
        if reprocess:
            assert (await client.post(f"{url}/reprocess")).status_code == 200
        updated = (await client.get(url)).json()
        assert updated["urls"]["full_print_preview"] is None
        assert (await client.get(photo["urls"]["full_print_preview"])).status_code == 404
        modes = updated["enhancement_applied"]["correction"]["crop_modes"]
        assert modes["full_print"]["suggested_warp_quad"] == photo["segmentation_mask"]["visible_polygon"]
        assert all(mode["status"] == "review_needed" for mode in modes.values())
        assert updated["crop_download_urls"] == {"full_print": None, "image_only": None}
        if not reprocess:
            assert (await client.get(updated["urls"]["source_corrected_untrimmed"])).content == original


@pytest.mark.parametrize("separate_print", [False, True])
async def test_nested_grid_requires_review_in_the_upload_and_export_flow(client, monkeypatch, separate_print):
    from uuid import uuid4

    from app.config import get_settings
    from app.pipeline import segmentation
    from app.pipeline.image_ops import pixel_box_to_normalized
    from app.pipeline.multipanel_prints import MULTIPANEL_PRINT_REVIEW
    from app.services import pipeline

    monkeypatch.setenv("AUTO_ANALYZE_UPLOADS", "true")
    monkeypatch.setenv("MAX_UPLOAD_SIZE", "1048576")
    monkeypatch.setenv("MAX_REQUEST_SIZE", "2097152")
    monkeypatch.setenv("ENABLE_PHOTO_ORIENTATION_CORRECTION", "false")
    monkeypatch.setenv("ENABLE_PROTECTED_REGION_GUARD", "false")
    get_settings.cache_clear()
    page = _contact_sheet()
    height, width = page.shape[:2]

    def proposal(quad):
        box = pixel_box_to_normalized(*cv2.boundingRect(quad), width, height)
        return segmentation._make_detection(
            box=box, points=quad, confidence=.9, image_width=width, image_height=height,
            settings=get_settings(), source="classical_border_quad", detector_source="edge_contour",
        )

    stock, grid = proposal(_OUTER), proposal(_INNER)
    monkeypatch.setattr(segmentation, "_segment_album_page", lambda *_: segmentation.SegmentationResult([stock, grid], {}))
    monkeypatch.setattr(pipeline, "run_paddle_ocr", lambda *_: [])
    album = await client.post("/api/v1/albums", json={"name": "Kontaktabzug"})
    assert album.status_code == 201, album.text
    album_id = album.json()["id"]
    album_url = f"/api/v1/albums/{album_id}"
    ok, payload = cv2.imencode(".png", page)
    assert ok
    upload = await client.post(f"{album_url}/pages/upload", files=[("files", ("album.png", payload.tobytes(), "image/png"))])
    assert upload.status_code == 201, upload.text
    photo, = (await client.get(f"{album_url}/photos")).json()
    assert photo["segmentation_mask"]["polygon"] == stock.mask["polygon"]
    assert photo["segmentation_mask"]["nested_print_suggestion"] == grid.mask["polygon"]
    assert photo["enhancement_applied"]["segmentation_review_reasons"] == [MULTIPANEL_PRINT_REVIEW]
    assert photo["crop_download_urls"] == {"full_print": None, "image_only": None}
    count_review, = (await client.get("/api/v1/review/queue", params={"type": "segmentation", "album_id": album_id})).json()
    assert count_review["payload"]["detection_count"] == 1
    count_entry, = count_review["payload"]["detections"]
    assert count_entry["segmentation_mask"]["nested_print_suggestion"] == grid.mask["polygon"]

    url = f"/api/v1/photos/{photo['id']}"
    if separate_print:
        # At limited contrast a flush second print can look like part of the
        # larger print. The reviewer may explicitly add it from the suggestion.
        addition = {
            "request_id": str(uuid4()), "warp_quad": grid.mask["polygon"],
            "adjacent_suggestion": {"photo_id": photo["id"], "polygon": grid.mask["polygon"], "kind": "nested"},
        }
        page_id = upload.json()["pages"][0]["page_id"]
        add_url = f"/api/v1/pages/{page_id}/manual-photos"
        added = await client.post(add_url, json=addition)
        assert added.status_code == 201, added.text
        retry = await client.post(add_url, json=addition)
        assert retry.status_code == 200 and retry.json()["id"] == added.json()["id"]
        duplicate = await client.post(add_url, json={**addition, "request_id": str(uuid4())})
        assert duplicate.status_code == 409
        assert duplicate.json()["detail"] == "adjacent_print_suggestion_saved"
        updated = (await client.get(url)).json()
        assert not updated["segmentation_mask"].get("nested_print_suggestion")
        assert updated["crop_download_urls"] == {"full_print": None, "image_only": None}
        queue = await client.get("/api/v1/review/queue", params={"type": "segmentation", "album_id": album_id})
        assert queue.json()[0]["payload"]["detection_count"] == 2
        return

    vertices = stock.mask["polygon"]
    reviewed = await client.patch(f"{url}/bounding-box", json={
        "bounding_box": stock.bounding_box,
        "segmentation_mask": {"polygon": vertices, "warp_quad": vertices, "image_quad": grid.mask["polygon"],
                              "source": "manual_quad", "print_size_mm": {"width": 551, "height": 402}},
    })
    assert reviewed.status_code == 200, reviewed.text
    assert not reviewed.json()["segmentation_mask"].get("nested_print_suggestion")
    for mode, reference in (("full_print", page[75:478, 78:630]), ("image_only", page[115:437, 119:589])):
        download = await client.get(reviewed.json()["crop_download_urls"][mode])
        assert download.status_code == 200, download.text
        crop = cv2.imdecode(np.frombuffer(download.content, np.uint8), cv2.IMREAD_COLOR)
        assert crop.shape == reference.shape
        assert np.abs(crop.astype(float) - reference).mean() < 3
