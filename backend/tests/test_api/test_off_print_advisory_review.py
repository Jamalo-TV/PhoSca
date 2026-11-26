from __future__ import annotations

from copy import deepcopy
from uuid import UUID, uuid4

import cv2
import numpy as np
import pytest


WIDTH, HEIGHT = 640, 480
HAAR_SOURCE = "opencv_haarcascade_frontalface_default"
DOWNLOADS = ("full_print", "image_only", "source_corrected_trimmed")


def _quad(left, top, right, bottom):
    return [
        {"x": x / (WIDTH - 1), "y": y / (HEIGHT - 1)}
        for x, y in ((left, top), (right, top), (right, bottom), (left, bottom))
    ]


def _bounds(quad):
    return {"x1": quad[0]["x"], "y1": quad[0]["y"], "x2": quad[2]["x"], "y2": quad[2]["y"]}


def _manual_edit(paper=None):
    paper = paper if paper is not None else _quad(270, 85, 592, 394)
    return {
        "bounding_box": _bounds(paper),
        "segmentation_mask": {
            "source": "manual_quad",
            "polygon": paper,
            "warp_quad": paper,
            "image_quad": _quad(285, 100, 577, 379),
            "print_size_mm": {"width": 161, "height": 154.5},
        },
    }


def _correction(photo):
    return photo["enhancement_applied"]["correction"]


async def _upload_scene(client, monkeypatch, *, precise=True, overlapping_peer=False):
    from app.config import get_settings
    from app.core.auth import hash_api_key
    from app.database import get_session_factory
    from app.models import ApiKey, ApiRole, Tenant
    from app.pipeline.segmentation import SegmentationDetection, SegmentationResult
    from app.services import pipeline

    monkeypatch.setenv("AUTH_REQUIRED", "true")
    monkeypatch.setenv("AUTO_ANALYZE_UPLOADS", "true")
    monkeypatch.setenv("MAX_UPLOAD_SIZE", "1048576")
    monkeypatch.setenv("MAX_REQUEST_SIZE", "2097152")
    monkeypatch.setenv("ENABLE_PRECISE_PHOTO_CROPS", str(precise).lower())
    monkeypatch.setenv("ENABLE_PROTECTED_REGION_GUARD", "true")
    monkeypatch.setenv("ENABLE_PHOTO_ORIENTATION_CORRECTION", "false")
    monkeypatch.setattr(pipeline, "run_paddle_ocr", lambda *_: [])
    get_settings.cache_clear()

    async with get_session_factory()() as session:
        tenant = Tenant(name="Paper review tenant")
        session.add(tenant)
        await session.flush()
        session.add(ApiKey(
            tenant_id=tenant.id, label="paper editor", role=ApiRole.editor,
            key_hash=hash_api_key("off-print-test-key"),
        ))
        await session.commit()
    client.headers["X-API-Key"] = "off-print-test-key"

    page = np.full((HEIGHT, WIDTH, 3), (85, 101, 110), np.uint8)
    page[85:395, 270:593] = (227, 237, 242)
    page[100:380, 285:578] = (76, 105, 142)
    cv2.ellipse(page, (436, 242), (106, 72), -18, 0, 360, (92, 151, 196), -1)
    cv2.line(page, (303, 349), (560, 302), (130, 167, 183), 3)

    # Both native advisory boxes cross the left paper edge. Their centers
    # lie on the backing. The PNG preserves the edge at every station.
    regions = [
        {
            "type": "face", "source": HAAR_SOURCE, "confidence": 0.75, "margin_ratio": 0.18,
            "box": {"x1": 233, "y1": top, "x2": 278, "y2": top + 46},
        }
        for top in (129, 287)
    ]
    monkeypatch.setattr(pipeline, "detect_protected_regions", lambda _: deepcopy(regions))
    quads = [_quad(270, 85, 592, 394)]
    if overlapping_peer:
        quads.append(_quad(151, 108, 294, 352))
    detections = [
        SegmentationDetection(
            _bounds(quad), {"source": "yolo_seg_mask", "detector_source": "yolo_onnx", "polygon": quad},
            0.91, 1.04, True, [],
        )
        for quad in quads
    ]
    monkeypatch.setattr(pipeline, "segment_album_page", lambda *_, **__: SegmentationResult(detections, {}))

    created = await client.post("/api/v1/albums", json={"name": "Off-print advisory faces"})
    assert created.status_code == 201, created.text
    album = f"/api/v1/albums/{created.json()['id']}"
    success, png = cv2.imencode(".png", page)
    assert success
    uploaded = await client.post(
        f"{album}/pages/upload", files={"files": ("straight-paper.png", png.tobytes(), "image/png")},
    )
    assert uploaded.status_code == 201, uploaded.text
    listed = await client.get(f"{album}/photos")
    assert listed.status_code == 200, listed.text
    assert len(listed.json()) == len(quads)
    photo = max(listed.json(), key=lambda row: row["bounding_box"]["x2"])
    url = f"/api/v1/photos/{photo['id']}"
    response = await client.get(url)
    assert response.status_code == 200, response.text
    return url, response.json(), regions


async def _downloads(client, url):
    return [await client.get(f"{url}/image", params={"variant": name}) for name in DOWNLOADS]


async def test_off_print_advisories_require_the_current_complete_manual_confirmation(client, monkeypatch):
    from app.database import get_session_factory
    from app.models import Page

    url, before, regions = await _upload_scene(client, monkeypatch)
    correction = _correction(before)
    assert all(correction["crop_modes"][mode]["status"] == "review_needed" for mode in DOWNLOADS[:2])
    assert correction["protected_regions"]["review_untrimmed"] is True
    assert all(response.status_code == 404 for response in await _downloads(client, url))
    offer = correction["album_page_mark_review"]
    assert offer["source"] == "off_print_advisory"
    assert offer["source_size"] == {"width": WIDTH, "height": HEIGHT}
    assert [item["region"]["box"] for item in offer["regions"]] == [region["box"] for region in regions]
    for item in offer["regions"]:
        box = item["region"]["box"]
        assert item["bounds"] == {
            "x1": box["x1"] / (WIDTH - 1), "y1": box["y1"] / (HEIGHT - 1),
            "x2": box["x2"] / (WIDTH - 1), "y2": box["y2"] / (HEIGHT - 1),
        }
        assert (box["x1"] + box["x2"]) / 2 < 270 < box["x2"]

    paper = correction["crop_modes"]["full_print"]["warp_quad"]

    async def save(**confirmation):
        return await client.patch(f"{url}/bounding-box", json={**_manual_edit(paper), **confirmation})

    guessed = await save()
    assert guessed.status_code == 422, guessed.text
    for indices in (None, [0], [0, 0], [1, 0]):
        rejected = await save(album_page_mark_review_id=offer["id"], album_page_mark_review_indices=indices)
        assert rejected.status_code == 409, rejected.text
        assert rejected.json()["detail"] == "album_page_mark_choices_required"
    stale = await save(album_page_mark_review_id=str(uuid4()), album_page_mark_review_indices=[0, 1])
    assert stale.status_code == 409, stale.text
    moved = await save(
        **_manual_edit(_quad(272, 85, 592, 394)),
        album_page_mark_review_id=offer["id"], album_page_mark_review_indices=[0, 1],
    )
    assert moved.status_code == 422, moved.text
    assert moved.json()["detail"] == "album_page_mark_outline_changed"
    assert (await client.get(url)).json() == before

    accepted = await save(album_page_mark_review_id=offer["id"], album_page_mark_review_indices=[0, 1])
    assert accepted.status_code == 200, accepted.text
    after = accepted.json()
    reviewed = after["enhancement_applied"]["review_provenance"]["reviewed_album_mark"]
    assert reviewed["id"] == offer["id"]
    assert reviewed["regions"] == offer["regions"]
    assert reviewed["warp_quad"] == paper
    assert reviewed["confirmed_indices"] == [0, 1]
    assert _correction(after)["protected_regions"]["reviewed_album_mark_id"] == offer["id"]
    assert all(_correction(after)["crop_modes"][mode]["status"] == "ready" for mode in DOWNLOADS[:2])
    initial = await _downloads(client, url)
    assert all(response.status_code == 200 for response in initial)

    replayed = await client.post(f"{url}/reprocess")
    assert replayed.status_code == 200, replayed.text
    repeated = await _downloads(client, url)
    assert [(response.status_code, response.content) for response in repeated] == [
        (200, response.content) for response in initial
    ]
    replay = await save(album_page_mark_review_id=offer["id"], album_page_mark_review_indices=[0, 1])
    assert replay.status_code == 409, replay.text

    async with get_session_factory()() as session:
        page = await session.get(Page, UUID(before["page_id"]))
        updated = deepcopy(page.processing_metadata)
        updated["steps"]["protected_regions"]["regions"].append({
            "type": "text", "source": "manual", "box": {"x1": 245, "y1": 244, "x2": 276, "y2": 263},
        })
        page.processing_metadata = updated
        await session.commit()
    assert all(response.status_code == 404 for response in await _downloads(client, url))
    assert (await client.get(url)).json()["crop_download_urls"] == {"full_print": None, "image_only": None}
    stale_reprocess = await client.post(f"{url}/reprocess")
    assert stale_reprocess.status_code == 200, stale_reprocess.text
    assert all(response.status_code == 404 for response in await _downloads(client, url))


async def test_off_print_advisory_is_retired_when_a_new_claim_touches_only_its_padding(client, monkeypatch):
    url, before, _ = await _upload_scene(client, monkeypatch)
    correction = _correction(before)
    offer = correction["album_page_mark_review"]
    paper = correction["crop_modes"]["full_print"]["warp_quad"]
    accepted = await client.patch(f"{url}/bounding-box", json={
        **_manual_edit(paper), "album_page_mark_review_id": offer["id"], "album_page_mark_review_indices": [0, 1],
    })
    assert accepted.status_code == 200, accepted.text
    assert all(response.status_code == 200 for response in await _downloads(client, url))

    # The claim stops short of the first raw face box and far short of the paper.
    # It enters only the existing protected padding along the left side of the box.
    page_id = before["page_id"]
    claim = await client.post(f"/api/v1/pages/{page_id}/manual-photos", json={
        "request_id": str(uuid4()), "warp_quad": _quad(195, 136, 230, 168),
    })
    assert claim.status_code == 201, claim.text
    detail = (await client.get(url)).json()
    assert "album_page_mark_review_stale" in _correction(detail)["crop_modes"]["full_print"]["reason_codes"]
    assert all(response.status_code == 404 for response in await _downloads(client, url))
    assert (await client.post(f"{url}/reprocess")).status_code == 200
    assert all(response.status_code == 404 for response in await _downloads(client, url))


@pytest.mark.parametrize(("precise", "overlapping_peer"), [(True, True), (False, False)])
async def test_off_print_advisory_offer_does_not_apply_to_an_unrelated_claim_or_legacy_crop(
    client, monkeypatch, precise, overlapping_peer,
):
    url, photo, _ = await _upload_scene(client, monkeypatch, precise=precise, overlapping_peer=overlapping_peer)
    correction = _correction(photo)
    assert "album_page_mark_review" not in correction
    if precise:
        guessed = await client.patch(f"{url}/bounding-box", json=_manual_edit())
        assert guessed.status_code == 422, guessed.text
        assert all(response.status_code == 404 for response in await _downloads(client, url))
    else:
        assert not correction.get("crop_modes")
        original = await client.get(f"{url}/image", params={"variant": "source_corrected_untrimmed"})
        assert original.status_code == 200, original.text
