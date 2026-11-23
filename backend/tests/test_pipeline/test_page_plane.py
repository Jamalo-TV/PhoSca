import cv2
import numpy as np
import pytest

from app.pipeline.page_plane import PRINT_ASPECT_REFERENCE_REVIEW, MeasuredPrintReference, measured_page_aspect
from app.pipeline.perspective import correct_photo_crop


def _scene(hinge=0, third=False):
    width, height = 2200, 1680
    page = np.full((height, width, 3), (61, 76, 98), np.uint8)
    camera = np.float64([[1940, 0, (width - 1) / 2], [0, 1940, (height - 1) / 2], [0, 0, 1]])
    locations = [((135, 102), (-85, -49, 0), 0), ((147, 95), (90, 19, 0), hinge)]
    if third:
        locations.append(((118, 84), (25, 110, 0), 0))
    result = []
    for (w, h), location, tilt in locations:
        local = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        corners = np.column_stack((local - [w / 2, h / 2], np.zeros(4)))
        rotation = cv2.Rodrigues(np.float64([np.deg2rad(tilt), 0, 0]))[0]
        world = corners @ rotation.T + location
        target, _ = cv2.projectPoints(world, np.float64([np.deg2rad(41), 0, 0]), np.float64([0, 0, 500]), camera, None)
        quad = target.reshape(4, 2).astype(np.float32)
        source = np.full((h + 1, w + 1, 3), 238, np.uint8)
        source[12:h - 11, 10:w - 9] = (58, 109, 147)
        matrix = cv2.getPerspectiveTransform(local, quad)
        raster = cv2.warpPerspective(source, matrix, (width, height), borderMode=cv2.BORDER_REPLICATE)
        x1, y1 = np.floor(quad.min(axis=0)).astype(int)
        x2, y2 = np.ceil(quad.max(axis=0)).astype(int) + 1
        yy, xx = np.ogrid[y1:y2, x1:x2]
        visible = np.ones((y2 - y1, x2 - x1), dtype=bool)
        for start, end in zip(quad, np.roll(quad, -1, axis=0), strict=True):
            dx, dy = end - start
            visible &= dx * (yy - start[1]) >= dy * (xx - start[0])
        page[y1:y2, x1:x2][visible] = raster[y1:y2, x1:x2][visible]
        inner = np.float32([[10, 12], [w - 10, 12], [w - 10, h - 12], [10, h - 12]])
        motif = cv2.perspectiveTransform(inner[None], matrix)[0]
        result.append((quad, motif, (w, h)))
    return page, result


def _mask(page, outer, motif):
    width, height = page.shape[1::-1]

    def vertices(points):
        return [{"x": float(x) / (width - 1), "y": float(y) / (height - 1)} for x, y in points]

    return {"source": "manual_quad", "polygon": vertices(outer), "warp_quad": vertices(outer), "image_quad": vertices(motif)}


def test_review_preview_recovers_both_aspects_from_measured_neighbor():
    page, ((peer, _, physical), (target, motif, expected)) = _scene()
    mask = _mask(page, target, motif)
    options = dict(dual_mode=True, auto_orient=False, protect_regions=False)
    donor = MeasuredPrintReference("peer", peer, physical[0] / physical[1])
    baseline = correct_photo_crop(page, {}, mask, **options)

    crop = correct_photo_crop(page, {}, mask, measured_page_references=[donor], **options)

    baseline_mode = baseline.metadata["crop_modes"]["full_print"]
    assert baseline_mode["reason_codes"] == ["PRINT_ASPECT_UNCERTAIN"]
    assert abs((baseline.image.shape[1] - 1) / (baseline.image.shape[0] - 1) / (expected[0] / expected[1]) - 1) > .15
    assert crop.metadata["aspect_ratio_estimation"]["method"] == "measured_page_reference"
    assert crop.metadata["aspect_ratio_estimation"]["page_reference_estimates"][0]["photo_id"] == "peer"
    for key, image, ratio in (("full_print", crop.image, expected[0] / expected[1]),
                              ("image_only", crop.image_only, (expected[0] - 20) / (expected[1] - 24))):
        assert (image.shape[1] - 1) / (image.shape[0] - 1) == pytest.approx(ratio, rel=.006)
        assert crop.metadata["crop_modes"][key]["reason_codes"] == [PRINT_ASPECT_REFERENCE_REVIEW]
        assert crop.metadata["crop_modes"][key]["status"] == "review_needed"


def test_page_reference_rejects_hinged_targets_and_unobserved_reference_edges():
    page, ((peer, _, physical), (target, _, _)) = _scene(hinge=5)
    donor = MeasuredPrintReference("peer", peer, physical[0] / physical[1])
    assert measured_page_aspect(page, target, [donor]) == (None, [])

    page, ((peer, _, physical), (target, _, _)) = _scene()
    wrong = MeasuredPrintReference("shifted", peer + np.float32([4, 0]), physical[0] / physical[1])
    assert measured_page_aspect(page, target, [wrong]) == (None, [])


def test_conflicting_measured_peers_prevent_aspect_transfer():
    page, ((peer, _, dimensions), (target, _, _), (other, _, other_size)) = _scene(third=True)
    measurements = [MeasuredPrintReference("first", peer, dimensions[0] / dimensions[1]),
                    MeasuredPrintReference("conflict", other, 1.05 * other_size[0] / other_size[1])]

    ratio, estimates = measured_page_aspect(page, target, measurements)

    assert ratio is None
    assert {entry["photo_id"] for entry in estimates} == {"first", "conflict"}


def test_calibrated_page_plane_abstains_on_mixed_planes():
    from app.evaluation.crop_fixtures import _projected_corners, synthetic_crop_pages
    from app.pipeline.page_plane import fit_calibrated_page_plane

    page = next(page for page in synthetic_crop_pages("development") if page.name == "dense_gutters")
    first = [_projected_corners(110, 90, (x, 113), (18, -12, -1)) for x in (90, 245, 420, 590)]
    other = [_projected_corners(110, 90, (x, 490), (-34, 29, 2)) for x in (90, 245, 420, 590)]
    assert fit_calibrated_page_plane(first + other, page.camera_matrix, page.image.shape) is None

    flat = [item.reference_modes["full_print"].corners for item in page.prints]
    plane = fit_calibrated_page_plane(flat, page.camera_matrix, page.image.shape)
    assert plane is not None
    assert plane.estimate_aspect(other[-1]) is None


def test_uncalibrated_plane_abstains_for_aligned_prints_and_mixed_planes():
    from app.evaluation.crop_fixtures import _projected_corners
    from app.pipeline.page_plane import fit_uncalibrated_page_plane

    aligned = [_projected_corners(120, 90, (x, y), (0, 0, 0), page_tilt=25)
               for x in (120, 310, 500, 690) for y in (128, 340, 500)]
    assert fit_uncalibrated_page_plane(aligned, (640, 840, 3)) is None

    mixed = [_projected_corners(120, 90, (x, y), (0, 0, roll), page_tilt=tilt)
             for y, tilt in ((130, 28), (450, -33))
             for x, roll in ((120, -25), (310, 16), (500, -22), (690, 18))]
    assert fit_uncalibrated_page_plane(mixed, (640, 840, 3)) is None


@pytest.mark.parametrize("calibrated", (True, False), ids=("camera-model", "different-print-angles"))
async def test_page_plane_stabilizes_both_previews_without_automatic_release(client, tmp_path, monkeypatch, calibrated):
    import json

    from app.config import get_settings
    from app.database import get_session_factory
    from app.evaluation.crop_fixtures import geometry_seed, synthetic_crop_pages
    from app.models import Album, ExtractedPhoto, Page
    from app.pipeline.image_ops import sha256_file

    monkeypatch.setenv("ENABLE_PRECISE_PHOTO_CROPS", "true")
    monkeypatch.setenv("ENABLE_PHOTO_ORIENTATION_CORRECTION", "false")
    monkeypatch.setenv("ENABLE_PROTECTED_REGION_GUARD", "false")
    name = "dense_gutters" if calibrated else "rotated_prints"
    fixture = next(page for page in synthetic_crop_pages("development") if page.name == name)
    path = tmp_path / "page.png"
    assert cv2.imwrite(str(path), fixture.image)
    capture_hash = sha256_file(path)
    height, width = fixture.image.shape[:2]
    calibration = tmp_path / "camera-calibration.json"
    if calibrated:
        calibration.write_text(json.dumps({"schema_version": 1, "captures": {capture_hash: {
            "width": width, "height": height, "camera_matrix": fixture.camera_matrix.tolist(), "calibration_id": "test",
        }}}), encoding="utf-8")
    monkeypatch.setenv("CAMERA_CALIBRATION_PATH", str(calibration))
    get_settings.cache_clear()

    async with get_session_factory()() as session:
        album = Album(name="Calibrated album")
        session.add(album)
        await session.flush()
        capture = Page(album_id=album.id, original_filename=path.name, storage_path=str(path), file_hash_sha256=capture_hash)
        session.add(capture)
        await session.flush()
        photos = []
        for item in fixture.prints:
            box, mask = geometry_seed(item, fixture.image.shape)
            photos.append(ExtractedPhoto(page_id=capture.id, album_id=album.id, storage_path=str(path),
                                        bounding_box=box, segmentation_mask=mask, segmentation_confidence=.95))
        session.add_all(photos)
        await session.commit()
        target_index = 4 if calibrated else 11
        target_id = photos[target_index].id

    response = await client.post(f"/api/v1/photos/{target_id}/reprocess")
    assert response.status_code == 200, response.text
    photo = response.json()
    correction = photo["enhancement_applied"]["correction"]
    estimate = correction["aspect_ratio_estimation"]
    if calibrated:
        assert correction["camera_calibration"]["capture_sha256"] == capture_hash
        assert estimate["method"] == "calibrated_camera"
        reason = "CAMERA_ASPECT_REVIEW"
    else:
        assert estimate["method"] == "estimated_page_plane"
        assert estimate["page_plane"]["independent_aspect_difference"] <= .01
        reason = PRINT_ASPECT_REFERENCE_REVIEW
    assert estimate["page_plane"]["supporting_prints"] >= 6
    for mode_name, reference in fixture.prints[target_index].reference_modes.items():
        mode = correction["crop_modes"][mode_name]
        assert reason in mode["reason_codes"]
        h, w = (mode["shape"][axis] for axis in ("height", "width"))
        rh, rw = reference.shape
        assert (w - 1) / (h - 1) == pytest.approx((rw - 1) / (rh - 1), rel=.01)
        assert photo["crop_download_urls"][mode_name] is None


async def test_reviewed_peer_changes_preview_but_cannot_release_either_crop(client, tmp_path, monkeypatch):
    from app.config import get_settings
    from app.database import get_session_factory
    from app.models import Album, ExtractedPhoto, Page
    from app.pipeline.image_ops import sha256_file

    monkeypatch.setenv("ENABLE_PRECISE_PHOTO_CROPS", "true")
    monkeypatch.setenv("ENABLE_PHOTO_ORIENTATION_CORRECTION", "false")
    monkeypatch.setenv("ENABLE_PROTECTED_REGION_GUARD", "false")
    get_settings.cache_clear()
    page, (reference, target) = _scene()
    path = tmp_path / "tilted-album.png"
    assert cv2.imwrite(str(path), page)
    edits = [
        {"bounding_box": {"x1": 0, "y1": 0, "x2": 1, "y2": 1}, "segmentation_mask": _mask(page, quad, motif)}
        for quad, motif, _ in (reference, target)
    ]
    async with get_session_factory()() as session:
        album = Album(name="Measured page")
        session.add(album)
        await session.flush()
        capture = Page(album_id=album.id, original_filename=path.name, storage_path=str(path), file_hash_sha256=sha256_file(path))
        session.add(capture)
        await session.flush()
        photos = [ExtractedPhoto(page_id=capture.id, album_id=album.id, storage_path=str(path), **body) for body in edits]
        session.add_all(photos)
        await session.commit()
        reference_url, target_url = [f"/api/v1/photos/{item.id}" for item in photos]

    baseline = await client.patch(f"{target_url}/bounding-box", json=edits[1])
    assert baseline.status_code == 200, baseline.text
    assert baseline.json()["enhancement_applied"]["correction"]["crop_modes"]["full_print"]["reason_codes"] == ["PRINT_ASPECT_UNCERTAIN"]

    ref_size = dict(zip(("width", "height"), reference[2], strict=True))
    edits[0]["segmentation_mask"]["print_size_mm"] = ref_size
    measured = await client.patch(f"{reference_url}/bounding-box", json=edits[0])
    assert measured.status_code == 200, measured.text
    preview = await client.post(f"{target_url}/reprocess")
    assert preview.status_code == 200, preview.text
    data = preview.json()
    correction = data["enhancement_applied"]["correction"]
    assert correction["aspect_ratio_estimation"]["page_reference_estimates"][0]["photo_id"] == measured.json()["id"]
    assert correction["aspect_ratio_estimation"]["estimated_ratio"] == pytest.approx(target[2][0] / target[2][1], rel=.002)
    for name in ("full_print", "image_only"):
        assert correction["crop_modes"][name]["reason_codes"] == [PRINT_ASPECT_REFERENCE_REVIEW]
        assert data["crop_download_urls"][name] is None
        assert (await client.get(f"{target_url}/image?variant={name}")).status_code == 404

    edits[0]["segmentation_mask"]["print_size_mm"] = None
    assert (await client.patch(f"{reference_url}/bounding-box", json=edits[0])).status_code == 200
    refreshed = await client.post(f"{target_url}/reprocess")
    assert refreshed.status_code == 200, refreshed.text
    refreshed_mode = refreshed.json()["enhancement_applied"]["correction"]["crop_modes"]["full_print"]
    assert refreshed_mode["reason_codes"] == ["PRINT_ASPECT_UNCERTAIN"]

    edits[1]["segmentation_mask"]["print_size_mm"] = dict(zip(("width", "height"), target[2], strict=True))
    confirmed = await client.patch(f"{target_url}/bounding-box", json=edits[1])
    assert confirmed.status_code == 200, confirmed.text
    for name in ("full_print", "image_only"):
        assert (await client.get(f"{target_url}/image?variant={name}")).status_code == 200
