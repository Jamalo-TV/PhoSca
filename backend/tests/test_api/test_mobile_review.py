from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path
from uuid import UUID, uuid4

import cv2
import numpy as np
import pytest


async def _seed_review_data(image=None):
    from app.config import get_settings
    from app.database import get_session_factory
    from app.models import Album, ExtractedPhoto, Page, PageStatus, PhotoStatus

    settings = get_settings()
    if image is None:
        image = np.full((120, 160, 3), 120, dtype=np.uint8)
    page_path = settings.storage_path / "uploads" / "review-page.jpg"
    original_path = settings.storage_path / "processed" / "review-original.jpg"
    enhanced_path = settings.storage_path / "processed" / "review-enhanced.jpg"
    for path in (page_path, original_path, enhanced_path):
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), image)
    album = Album(name="Mobile review")
    async with get_session_factory()() as session:
        session.add(album)
        await session.flush()
        page = Page(
            album_id=album.id,
            original_filename=page_path.name,
            storage_path=str(page_path),
            file_hash_sha256=hashlib.sha256(page_path.read_bytes()).hexdigest(),
            status=PageStatus.review_needed,
        )
        session.add(page)
        await session.flush()
        photo = ExtractedPhoto(
            page_id=page.id,
            album_id=album.id,
            storage_path=str(original_path),
            original_storage_path=str(original_path),
            bounding_box={"x1": 0.1, "y1": 0.2, "x2": 0.8, "y2": 0.9},
            segmentation_mask={
                "source": "classical_boundary",
                "polygon": [
                    {"x": 0.1, "y": 0.2},
                    {"x": 0.45, "y": 0.18},
                    {"x": 0.8, "y": 0.2},
                    {"x": 0.82, "y": 0.55},
                    {"x": 0.8, "y": 0.9},
                    {"x": 0.1, "y": 0.9},
                ],
                "warp_quad": [{"x": 0.1, "y": 0.2}, {"x": 0.8, "y": 0.2}, {"x": 0.8, "y": 0.9}, {"x": 0.1, "y": 0.9}],
            },
            segmentation_confidence=0.72,
            aspect_ratio=1.2,
            geometry_valid=True,
            status=PhotoStatus.review_needed,
            enhancement_applied={
                "variants": {
                    "original": {"path": str(original_path)},
                    "enhanced": {"path": str(enhanced_path), "metrics": {"source_similarity": 0.91}},
                },
                "correction": {"orientation": {"rotation_degrees": 0}, "perspective_confidence": 0.84},
            },
        )
        session.add(photo)
        await session.commit()
        return album.id, page.id, photo.id


async def test_scene_edge_guide_is_retired_by_a_new_manual_claim_on_the_remote_witness(client, monkeypatch):
    from app.config import get_settings
    from app.database import get_session_factory
    from app.models import ExtractedPhoto
    from app.pipeline.ribbed_scene_edge import RIBBED_SCENE_EDGE, RIBBED_SCENE_EDGE_SCOPE

    monkeypatch.setenv("ENABLE_PRECISE_PHOTO_CROPS", "true")
    get_settings.cache_clear()
    _, page_id, photo_id = await _seed_review_data(np.full((600, 660, 3), 180, np.uint8))

    def quad(left, top, right, bottom):
        return [{"x": x, "y": y} for x, y in ((left, top), (right, top), (right, bottom), (left, bottom))]

    retained_geometry = quad(.10, .20, .79, .90)
    trace = [quad(.14, .32, .15, .56)[::3]]
    scope = [quad(.13, .30, .17, .60), quad(.84, .30, .85, .60)]
    async with get_session_factory()() as session:
        photo = await session.get(ExtractedPhoto, photo_id)
        photo.enhancement_applied = {"correction": {"crop_modes": {"full_print": {
            "warp_quad": retained_geometry, "status": "review_needed", "suggested_edge_source": RIBBED_SCENE_EDGE,
            "suggested_edge_traces": trace, RIBBED_SCENE_EDGE_SCOPE: scope,
        }}}}
        await session.commit()

    add_url = f"/api/v1/pages/{page_id}/manual-photos"
    unrelated = await client.post(add_url, json={"request_id": str(uuid4()), "warp_quad": quad(.88, .13, .97, .29)})
    assert unrelated.status_code == 201, unrelated.text
    async with get_session_factory()() as session:
        photo = await session.get(ExtractedPhoto, photo_id)
        assert photo.enhancement_applied["correction"]["crop_modes"]["full_print"]["suggested_edge_traces"] == trace

    overlaps = await client.post(add_url, json={"request_id": str(uuid4()), "warp_quad": quad(.82, .34, .93, .62)})
    assert overlaps.status_code == 201, overlaps.text
    async with get_session_factory()() as session:
        photo = await session.get(ExtractedPhoto, photo_id)
        full = photo.enhancement_applied["correction"]["crop_modes"]["full_print"]
        assert full == {"warp_quad": retained_geometry, "status": "review_needed"}


@pytest.mark.parametrize("edit_existing", [False, True])
async def test_neighboring_source_guide_is_retired_only_by_a_material_manual_claim(client, monkeypatch, edit_existing):
    from app.config import get_settings
    from app.database import get_session_factory
    from app.models import Page
    from app.pipeline.contested_scene_edge import CONTESTED_SCENE_EDGE_TRACES
    from app.pipeline.near_capture_review import NEAR_CAPTURE_PRINT_TRACES
    from app.pipeline.mounted_window_review import MOUNTED_WINDOW_REVIEW_TRACES
    from app.pipeline.neighboring_print_review import FOLLOWUP_PRINT_EDGE_GUIDE, NEIGHBORING_PRINT_EDGE_TRACES

    monkeypatch.setenv("ENABLE_PRECISE_PHOTO_CROPS", "true")
    monkeypatch.setenv("ENABLE_PHOTO_ORIENTATION_CORRECTION", "false")
    get_settings.cache_clear()
    album_id, page_id, photo_id = await _seed_review_data(np.full((600, 660, 3), 180, np.uint8))

    def quad(left, top, right, bottom):
        return [{"x": x, "y": y} for x, y in ((left, top), (right, top), (right, bottom), (left, bottom))]

    detected = quad(.80, .25, .96, .75)
    traces = [detected[:2], detected[1:3], detected[2:]]
    other_traces = [quad(.02, .45, .08, .63)[::3]]
    scene_traces = [quad(.910, .34, .915, .65)[::3]]
    window_traces = [quad(.900, .37, .905, .62)[::3]]
    followup_guide = {"traces": [quad(.310, .06, .315, .14)[::3]], "witnesses": [quad(.28, .06, .33, .14)]}
    async with get_session_factory()() as session:
        page = await session.get(Page, page_id)
        page.processing_metadata = {"steps": {"segmentation": {
            NEIGHBORING_PRINT_EDGE_TRACES: traces, NEAR_CAPTURE_PRINT_TRACES: other_traces,
            FOLLOWUP_PRINT_EDGE_GUIDE: followup_guide,
            CONTESTED_SCENE_EDGE_TRACES: scene_traces,
            MOUNTED_WINDOW_REVIEW_TRACES: window_traces,
        }}}
        await session.commit()

    review_url = f"/api/v1/review/queue?album_id={album_id}&type=segmentation"

    async def review_payload():
        item, = (await client.get(review_url)).json()
        return item["payload"]

    initial = await review_payload()
    assert initial[NEIGHBORING_PRINT_EDGE_TRACES] == traces
    assert initial[NEAR_CAPTURE_PRINT_TRACES] == other_traces
    assert initial[FOLLOWUP_PRINT_EDGE_GUIDE] == followup_guide
    assert initial[CONTESTED_SCENE_EDGE_TRACES] == scene_traces
    assert initial[MOUNTED_WINDOW_REVIEW_TRACES] == window_traces
    assert initial["detection_count"] == 1 and initial["deferred_print_proposals"] == []
    manual_url = f"/api/v1/pages/{page_id}/manual-photos"
    separate = await client.post(manual_url, json={"request_id": str(uuid4()), "warp_quad": quad(.08, .03, .29, .15)})
    assert separate.status_code == 201, separate.text
    assert (await review_payload())[NEIGHBORING_PRINT_EDGE_TRACES] == traces
    assert FOLLOWUP_PRINT_EDGE_GUIDE not in (await review_payload())
    assert (await review_payload())[CONTESTED_SCENE_EDGE_TRACES] == scene_traces
    assert (await review_payload())[MOUNTED_WINDOW_REVIEW_TRACES] == window_traces

    # Only this part of the saved quad intersects a measured span: no inferred
    # corner or automatic photo is created from the open trace.
    claim = quad(.92, .36, .98, .61)
    if edit_existing:
        saved = await client.patch(f"/api/v1/photos/{photo_id}/bounding-box", json={
            "bounding_box": {"x1": .92, "y1": .36, "x2": .98, "y2": .61},
            "segmentation_mask": {"source": "manual_quad", "polygon": claim, "warp_quad": claim},
        })
    else:
        saved = await client.post(manual_url, json={"request_id": str(uuid4()), "warp_quad": claim})
    assert saved.status_code == (200 if edit_existing else 201), saved.text
    review_provenance = saved.json()["enhancement_applied"]["review_provenance"]
    assert "paper_trace_review" not in review_provenance
    final = await review_payload()
    assert NEIGHBORING_PRINT_EDGE_TRACES not in final
    assert CONTESTED_SCENE_EDGE_TRACES not in final
    assert MOUNTED_WINDOW_REVIEW_TRACES not in final
    assert final[NEAR_CAPTURE_PRINT_TRACES] == other_traces
    assert final["detection_count"] == (2 if edit_existing else 3)


async def test_embossed_carrier_cannot_restore_a_manually_updated_child_outline(client, monkeypatch):
    from app.config import get_settings
    from app.database import get_session_factory
    from app.models import ExtractedPhoto, Page
    from app.pipeline.embossed_album_pockets import EMBOSSED_CARRIER_PROPOSAL

    monkeypatch.setenv("ENABLE_PRECISE_PHOTO_CROPS", "true")
    get_settings.cache_clear()
    album_id, page_id, photo_id = await _seed_review_data()

    def rectangle(left, top, right, bottom):
        return [{"x": x, "y": y} for x, y in ((left, top), (right, top), (right, bottom), (left, bottom))]

    saved = rectangle(.14, .18, .78, .88)
    pending = {"id": 1, "reason": EMBOSSED_CARRIER_PROPOSAL, "polygon": rectangle(.07, .09, .90, .96),
               "contained_print_polygon": rectangle(.22, .29, .72, .81)}
    revision = str(uuid4())
    async with get_session_factory()() as session:
        page = await session.get(Page, page_id)
        page.processing_metadata = {"photo_count_revision": revision, "steps": {
            "segmentation": {"fusion": {"deferred_print_proposals": [pending]}},
        }}
        photo = await session.get(ExtractedPhoto, photo_id)
        photo.segmentation_mask = {**photo.segmentation_mask, "source": "manual_quad", "warp_quad": saved}
        await session.commit()

    result = await client.post(f"/api/v1/pages/{page_id}/manual-photos", json={
        "request_id": str(uuid4()), "warp_quad": saved, "deferred_proposal_id": 1,
        "photo_count_revision": revision, "confirm_album_pocket_print": True,
    })
    assert result.status_code == 422 and result.json()["detail"] == "album_carrier_matches_detected_print"
    assert len((await client.get(f"/api/v1/albums/{album_id}/photos")).json()) == 1
    async with get_session_factory()() as session:
        page = await session.get(Page, page_id)
        assert page.processing_metadata["steps"]["segmentation"]["fusion"]["deferred_print_proposals"] == [pending]


async def test_nested_edge_count_decision_keeps_evidence_and_retires_only_its_parent_reason(client):
    from app.database import get_session_factory
    from app.models import ExtractedPhoto, Page, ReviewEvent
    from app.pipeline.fusion_ownership import NESTED_EDGE_CONTOUR, NESTED_EDGE_CONTOUR_REVIEW

    _, page_id, photo_id = await _seed_review_data()
    revision = uuid4().hex
    outline = [{"x": x, "y": y} for x, y in ((.25, .40), (.60, .40), (.60, .80), (.25, .80))]
    offer = {"id": 2, "reason": NESTED_EDGE_CONTOUR, "polygon": outline, "parent_cluster_id": 1,
             "source_mask": {"polygon": outline}, "shared_paper_edge_traces": [outline[:2], outline[2:]]}
    async with get_session_factory()() as session:
        page = await session.get(Page, page_id)
        page.processing_metadata = {"photo_count_revision": revision, "steps": {
            "segmentation": {"fusion": {"deferred_print_proposals": [offer]}},
        }}
        parent = await session.get(ExtractedPhoto, photo_id)
        parent.enhancement_applied = {**parent.enhancement_applied, "segmentation_review_reasons": [NESTED_EDGE_CONTOUR_REVIEW],
            "review_reasons": ["PRINT_BOUNDARY_UNSUPPORTED", NESTED_EDGE_CONTOUR_REVIEW],
            "review_provenance": {"original_prediction": {"segmentation_mask": {"fusion": {"cluster_id": 1}}}}}
        await session.commit()

    payload = {"task_type": "segmentation", "task_id": str(page_id), "answer": "correct", "idempotency_key": uuid4().hex,
               "metadata": {"photo_count_revision": revision, "deferred_proposal_ids": [2]}}
    saved = await client.post("/api/v1/review/events", json=payload)
    assert saved.status_code == 201, saved.text
    replay = await client.post("/api/v1/review/events", json=payload)
    assert replay.status_code == 201 and replay.json()["duplicate"] is True
    stale = await client.post("/api/v1/review/events", json={**payload, "idempotency_key": uuid4().hex})
    assert stale.status_code == 409
    async with get_session_factory()() as session:
        parent = await session.get(ExtractedPhoto, photo_id)
        assert parent.enhancement_applied["segmentation_review_reasons"] == []
        assert parent.enhancement_applied["review_reasons"] == ["PRINT_BOUNDARY_UNSUPPORTED"]
        event = await session.get(ReviewEvent, UUID(saved.json()["id"]))
        assert event.event_metadata["snapshot"]["deferred_print_proposals"] == [offer]
        page = await session.get(Page, page_id)
        assert "deferred_print_proposals" not in page.processing_metadata["steps"]["segmentation"]["fusion"]


async def test_tiled_print_requires_own_paper_confirmation_and_keeps_crops_held(client, monkeypatch):
    from app.config import get_settings
    from app.database import get_session_factory
    from app.models import ExtractedPhoto, Page
    from app.pipeline.tile_print_review import TILED_PRINT_PROPOSAL

    monkeypatch.setenv("ENABLE_PHOTO_ORIENTATION_CORRECTION", "false")
    get_settings.cache_clear()
    album_id, page_id, photo_id = await _seed_review_data()
    def rect(x1, y1, x2, y2):
        return [{"x": x, "y": y} for x, y in ((x1, y1), (x2, y1), (x2, y2), (x1, y2))]
    own = rect(.12, .20, .40, .62)
    outline = rect(.58, .23, .89, .65)
    offer = {"id": 1, "reason": TILED_PRINT_PROPOSAL, "polygon": outline}
    revision = str(uuid4())
    async with get_session_factory()() as session:
        page = await session.get(Page, page_id)
        page.processing_metadata = {"photo_count_revision": revision, "steps": {
            "segmentation": {"fusion": {"deferred_print_proposals": [offer]}},
        }}
        parent = await session.get(ExtractedPhoto, photo_id)
        parent.segmentation_mask = {"polygon": own}
        parent.enhancement_applied = {**parent.enhancement_applied, "correction": {
            "crop_modes": {"full_print": {"warp_quad": rect(.12, .20, .72, .62)}},
        }}
        await session.commit()

    url = f"/api/v1/pages/{page_id}/manual-photos"
    base = {"request_id": str(uuid4()), "warp_quad": outline}
    linked = {**base, "deferred_proposal_id": offer["id"], "photo_count_revision": revision}
    for request in (base, linked):
        missing = await client.post(url, json=request)
        assert (missing.status_code, missing.json()["detail"]) == (409, "standalone_print_confirmation_required")
    # This smaller outline still qualifies when linked to the offer, even though its IoU is below 0.90.
    inset = rect(.61, .26, .86, .62)
    size = {"width": 96, "height": 100}
    unlinked = await client.post(url, json={**base, "warp_quad": inset, "print_size_mm": size})
    assert (unlinked.status_code, unlinked.json()["detail"]) == (409, "pending_tiled_print_confirmation_required")
    replaced = await client.patch(f"/api/v1/photos/{photo_id}/bounding-box", json={
        "bounding_box": {"x1": .61, "y1": .26, "x2": .86, "y2": .62},
        "segmentation_mask": {"source": "manual_quad", "polygon": inset, "warp_quad": inset, "print_size_mm": size},
    })
    assert (replaced.status_code, replaced.json()["detail"]) == (409, "pending_tiled_print_confirmation_required")
    fitted = await client.patch(f"/api/v1/photos/{photo_id}/bounding-box", json={
        "bounding_box": {"x1": .61, "y1": .26, "x2": .86, "y2": .62},
        "segmentation_mask": {"polygon": [inset[0], {"x": .70, "y": .26}, *inset[1:]], "visible_polygon": own},
    })
    assert (fitted.status_code, fitted.json()["detail"]) == (409, "pending_tiled_print_confirmation_required")
    unchanged, = (await client.get(f"/api/v1/albums/{album_id}/photos")).json()
    assert unchanged["id"] == str(photo_id) and unchanged["segmentation_mask"]["polygon"] == own
    confirmed = {**linked, "confirm_standalone_print": True}
    overlapping = await client.post(url, json=confirmed)
    assert (overlapping.status_code, overlapping.json()["detail"]) == (409, "tiled_print_claim_conflict")

    async with get_session_factory()() as session:
        parent = await session.get(ExtractedPhoto, photo_id)
        parent.enhancement_applied = {**parent.enhancement_applied, "correction": {}}
        await session.commit()
    confirmed = {**confirmed, "warp_quad": inset, "print_size_mm": size}
    saved = await client.post(url, json=confirmed)
    assert saved.status_code == 201, saved.text
    added = saved.json()
    assert added["crop_download_urls"] == {"full_print": None, "image_only": None}
    photo = f"/api/v1/photos/{added['id']}/image"
    for variant in ("full_print", "image_only", "source_corrected_trimmed"):
        assert (await client.get(photo, params={"variant": variant})).status_code == 404
    assert (await client.get(photo, params={"variant": "source_corrected_untrimmed"})).status_code == 200
    replay = await client.post(url, json=confirmed)
    assert replay.status_code == 200 and replay.json()["id"] == added["id"]
    assert len((await client.get(f"/api/v1/albums/{album_id}/photos")).json()) == 2
    async with get_session_factory()() as session:
        page = await session.get(Page, page_id)
        assert page.processing_metadata["steps"]["segmentation"]["fusion"]["deferred_print_proposals"] == []


async def test_capture_fabric_choices_remain_reversible_and_cannot_restore_twice(client, monkeypatch):
    from app.config import get_settings
    from app.database import get_session_factory
    from app.models import Page
    from app.pipeline.capture_fabric_review import CAPTURE_FABRIC_PROPOSAL

    monkeypatch.setenv("ENABLE_PHOTO_ORIENTATION_CORRECTION", "false")
    get_settings.cache_clear()
    album_id, page_id, _ = await _seed_review_data()

    def proposal(offer_id, left, right):
        outline = [{"x": x, "y": y} for x, y in ((left, .03), (right, .03), (right, .17), (left, .17))]
        return {"id": offer_id, "reason": CAPTURE_FABRIC_PROPOSAL, "polygon": outline}

    first, alternative = proposal(0, .08, .79), proposal(1, .11, .76)
    revision = str(uuid4())
    async with get_session_factory()() as session:
        page = await session.get(Page, page_id)
        page.processing_metadata = {"photo_count_revision": revision, "steps": {
            "segmentation": {"fusion": {"deferred_print_proposals": [first, alternative]}},
        }}
        await session.commit()

    url = f"/api/v1/pages/{page_id}/manual-photos"
    payload = {"request_id": str(uuid4()), "deferred_proposal_id": first["id"],
               "photo_count_revision": revision, "warp_quad": first["polygon"]}
    refused = await client.post(url, json=payload)
    assert (refused.status_code, refused.json()["detail"]) == (409, "standalone_print_confirmation_required")
    accepted = await client.post(url, json={**payload, "confirm_standalone_print": True})
    assert accepted.status_code == 201, accepted.text
    assert accepted.json()["crop_download_urls"] == {"full_print": None, "image_only": None}
    async with get_session_factory()() as session:
        page = await session.get(Page, page_id)
        current = page.processing_metadata
        assert current["steps"]["segmentation"]["fusion"]["deferred_print_proposals"] == [alternative]
    retry = {**payload, "request_id": str(uuid4()), "deferred_proposal_id": alternative["id"],
             "warp_quad": alternative["polygon"], "confirm_standalone_print": True}
    stale = await client.post(url, json=retry)
    assert (stale.status_code, stale.json()["detail"]) == (409, "deferred_print_proposal_stale")
    overlap = await client.post(url, json={**retry, "photo_count_revision": current["photo_count_revision"]})
    assert (overlap.status_code, overlap.json()["detail"]) == (409, "capture_fabric_claim_conflict")
    photos = (await client.get(f"/api/v1/albums/{album_id}/photos")).json()
    assert len(photos) == 2


async def test_edge_trace_reads_the_exif_oriented_source_without_editing_review(client, monkeypatch):
    from PIL import Image
    from sqlalchemy import func, select

    from app.config import get_settings
    from app.database import get_session_factory
    from app.models import AuditLog

    album_id, page_id, photo_id = await _seed_review_data()
    source = get_settings().storage_path / "uploads" / "review-page.jpg"
    image = Image.new("RGB", (480, 320), "#dddddd")
    image.paste("#505050", (0, 175, 480, 320))
    exif = image.getexif()
    exif[274] = 6
    image.save(source, exif=exif, quality=96)
    # The dark lower half of the stored image becomes the left half.
    coords = dict(start_x=144 / 319, start_y=50 / 479, middle_x=144 / 319,
                  middle_y=240 / 479, end_x=144 / 319, end_y=430 / 479)
    url = f"/api/v1/pages/{page_id}/edge-trace"
    photo_url = f"/api/v1/photos/{photo_id}"
    photos_url = f"/api/v1/albums/{album_id}/photos"
    before = (await client.get(photo_url)).json()
    async with get_session_factory()() as session:
        audit_count = await session.scalar(select(func.count()).select_from(AuditLog))

    measured = await client.get(url, params=coords)
    assert measured.status_code == 200, measured.text
    assert measured.headers["cache-control"] == "private, no-store"
    trace, = measured.json()["traces"]
    assert all(abs(point["x"] * 319 - 144.5) < 1 for point in trace)
    assert trace[0]["y"] < .15 and trace[-1]["y"] > .85

    flat = await client.get(url, params=dict(coords, start_x=.75, middle_x=.75, end_x=.75))
    assert flat.status_code == 200
    assert flat.json() == {"traces": []}
    too_short = await client.get(url, params=dict(coords, end_y=60 / 479))
    assert too_short.status_code == 422
    assert too_short.json()["detail"] == "edge_trace_points_too_close"
    assert (await client.get(url, params=dict(coords, middle_x="nan"))).status_code == 422
    assert (await client.get(photo_url)).json() == before
    assert [photo["id"] for photo in (await client.get(photos_url)).json()] == [str(photo_id)]
    async with get_session_factory()() as session:
        assert await session.scalar(select(func.count()).select_from(AuditLog)) == audit_count

    monkeypatch.setattr(get_settings(), "max_image_pixels", 100_000)
    oversized = await client.get(url, params=coords)
    assert oversized.status_code == 413
    assert oversized.json()["detail"] == "edge_trace_image_too_large"


async def test_rejected_crop_is_hidden_from_album_browsing_and_kept_for_geometry_review(submit_geometry_review, client, monkeypatch):
    from app.config import get_settings
    from app.database import get_session_factory
    from app.models import ExtractedPhoto, OCRResult

    monkeypatch.setenv("ENABLE_PROTECTED_REGION_GUARD", "false")
    monkeypatch.setenv("ENABLE_PHOTO_ORIENTATION_CORRECTION", "false")
    get_settings.cache_clear()
    album_id, page_id, photo_id = await _seed_review_data()
    async with get_session_factory()() as session:
        photo = await session.get(ExtractedPhoto, photo_id)
        metadata = photo.enhancement_applied
        photo.enhancement_applied = {**metadata, "variants": {
            **metadata["variants"], "source_corrected_trimmed": {"path": photo.original_storage_path},
        }}
        session.add(OCRResult(page_id=page_id, photo_id=photo_id, text_content="Searchable caption", ocr_engine="test"))
        await session.commit()

    album_url = f"/api/v1/albums/{album_id}"
    photo_url = f"/api/v1/photos/{photo_id}"
    assert (await client.get(f"{photo_url}/image")).status_code == 200
    assert (await client.get(f"{album_url}/photos")).json()[0]["id"] == str(photo_id)
    event = await submit_geometry_review("/api/v1/review/events", json={
        "task_type": "geometry", "task_id": str(photo_id), "answer": "wrong_object", "idempotency_key": uuid4().hex,
    })
    assert event.status_code == 201

    for summary in ((await client.get(album_url)).json(), (await client.get("/api/v1/albums")).json()[0]):
        assert summary["photos_extracted"] == 0
        assert summary["review_needed_count"] == 1  # The page's count review stays open.
    assert (await client.get(f"{album_url}/photos")).json() == []
    archived, = (await client.get(f"{album_url}/photos", params={"include_rejected": True})).json()
    assert archived["id"] == str(photo_id)
    assert archived["reviewed_non_print"] is True
    assert archived["enhancement_applied"]["review_rejected_detection"]
    assert (await client.get(f"{photo_url}/image")).status_code == 404
    for variant in ("original", "source_corrected_trimmed", "enhanced", "full_print", "image_only"):
        assert (await client.get(f"{photo_url}/image", params={"variant": variant})).status_code == 404
    evidence = await client.get(f"{photo_url}/image", params={"variant": "source_corrected"})
    assert evidence.status_code == 200 and evidence.headers["cache-control"] == "private, no-store"
    review_params = {"album_id": str(album_id), "type": "geometry"}
    item, = (await client.get("/api/v1/review/queue", params=review_params)).json()
    assert item["id"] == str(photo_id)
    active_params = {**review_params, "include_rejected": False, "limit": 1}
    active = await client.get("/api/v1/review/queue", params=active_params)
    assert active.status_code == 200 and active.json() == []
    match, = (await client.get("/api/v1/search", params={"q": "Searchable"})).json()
    assert match["page_id"] == str(page_id) and match["photo_id"] is None

    quad = archived["segmentation_mask"]["warp_quad"]
    restored = await client.patch(f"{photo_url}/bounding-box", json={
        "bounding_box": archived["bounding_box"],
        "segmentation_mask": {"source": "manual_quad", "warp_quad": quad, "polygon": quad},
    })
    assert restored.status_code == 200, restored.text
    assert restored.json()["reviewed_non_print"] is False
    assert (await client.get(f"{album_url}/photos")).json()[0]["id"] == str(photo_id)
    assert (await client.get(album_url)).json()["photos_extracted"] == 1
    restored_task, = (await client.get("/api/v1/review/queue", params=active_params)).json()
    assert restored_task["id"] == str(photo_id)

    # The rejection only hides the geometry that was actually reviewed.
    async with get_session_factory()() as session:
        photo = await session.get(ExtractedPhoto, photo_id)
        photo.enhancement_applied = {**photo.enhancement_applied,
            "review_rejected_detection": archived["enhancement_applied"]["review_rejected_detection"],
        }
        await session.commit()
    assert (await client.get(f"{album_url}/photos")).json()[0]["reviewed_non_print"] is False
    assert (await client.get(album_url)).json()["photos_extracted"] == 1


@pytest.mark.parametrize("surface", ["bowed_paper", "straight_paper", "printed_arc"])
async def test_manual_export_checks_the_server_owned_paper_outline(client, monkeypatch, surface):
    from app.config import get_settings
    from app.database import get_session_factory
    from app.models import ExtractedPhoto
    from app.pipeline.manual_paper_boundary import MANUAL_PAPER_BOUNDARY_REVIEW

    monkeypatch.setenv("ENABLE_PHOTO_ORIENTATION_CORRECTION", "false")
    monkeypatch.setenv("ENABLE_PROTECTED_REGION_GUARD", "false")
    get_settings.cache_clear()
    height, width = 1050, 1300
    outer = np.float32([[225, 165], [1079, 165], [1079, 935], [225, 935]])
    fraction = np.linspace(0, 1, 145, dtype=np.float32)[1:-1]
    sag = 0 if surface == "straight_paper" else 4
    outline = np.vstack((outer, np.column_stack((225 - 4 * sag * fraction * (1 - fraction), 935 - 770 * fraction))))
    # The false outline follows ink inside a print whose outer edge is faint and straight.
    image = np.full((height, width, 3), 224, np.uint8)
    if surface == "printed_arc":
        image[165:936, 215:1080] = 215
    pixels = np.zeros((height, width), np.uint8)
    cv2.fillPoly(pixels, [np.int32(np.rint(outline))], 1)
    rows, columns = np.nonzero(pixels)
    ink = (89 + 30 * np.cos(columns / 75) * np.cos(rows / 40)).astype(np.uint8)
    image[rows, columns] = np.column_stack((ink, ink + 14, ink + 28))
    _, _, photo_id = await _seed_review_data(image)

    def vertices(points):
        return [{"x": float(x / (width - 1)), "y": float(y / (height - 1))} for x, y in points]

    quad = vertices(outer)
    box = {"x1": quad[0]["x"], "y1": quad[0]["y"], "x2": quad[2]["x"], "y2": quad[2]["y"]}
    async with get_session_factory()() as session:
        photo = await session.get(ExtractedPhoto, photo_id)
        photo.bounding_box = box
        photo.segmentation_mask = {"polygon": quad, "warp_quad": quad, "visible_polygon": vertices(outline)}
        await session.commit()

    photo_url = f"/api/v1/photos/{photo_id}"
    edited_outer = outer.copy()
    if surface == "printed_arc":
        edited_outer[[0, 3], 0] = 215
    edited_quad = vertices(edited_outer)
    body = {
        "bounding_box": {**box, "x1": edited_quad[0]["x"]},
        "segmentation_mask": {
            "source": "manual_quad", "polygon": edited_quad, "warp_quad": edited_quad,
            "image_quad": vertices([[248, 188], [1056, 188], [1056, 912], [248, 912]]),
            "print_size_mm": {"width": 111, "height": 100},
        },
    }
    held = surface == "bowed_paper"
    for _ in range(2):
        saved = await client.patch(f"{photo_url}/bounding-box", json=body)
        assert saved.status_code == 200, saved.text
        correction = saved.json()["enhancement_applied"]["correction"]
        assert correction["manual_paper_boundary"]["requires_review"] is held
        for mode in ("full_print", "image_only", "source_corrected_trimmed"):
            assert (await client.get(f"{photo_url}/image?variant={mode}")).status_code == (404 if held else 200)
        for mode in correction["crop_modes"].values():
            assert (MANUAL_PAPER_BOUNDARY_REVIEW in mode["reason_codes"]) is held
    if held:
        assert correction["crop_modes"]["full_print"]["suggested_edge_traces"]

    # Simulate the saved data from an older worker that manufactured ready exports.
    async with get_session_factory()() as session:
        photo = await session.get(ExtractedPhoto, photo_id)
        metadata = dict(photo.enhancement_applied)
        correction = dict(metadata["correction"])
        correction.pop("manual_paper_boundary")
        correction["crop_modes"] = {mode: {**value, "status": "ready"}
                                    for mode, value in correction["crop_modes"].items()}
        metadata["correction"] = correction
        metadata["variants"] = {mode: {"path": photo.original_storage_path} for mode in correction["crop_modes"]}
        photo.enhancement_applied = metadata
        await session.commit()
    assert (await client.get(f"{photo_url}/image?variant=full_print")).status_code == (
        200 if surface == "straight_paper" else 404
    )
    replay = await client.post(f"{photo_url}/reprocess")
    assert replay.status_code == 200, replay.text
    assert replay.json()["enhancement_applied"]["correction"]["manual_paper_boundary"]["requires_review"] is held


async def test_manual_edit_keeps_the_disputed_straight_side_in_review_until_both_spans_align(client, monkeypatch):
    from app.config import get_settings
    from app.database import get_session_factory
    from app.models import ExtractedPhoto
    from app.pipeline.print_edges import POSSIBLE_BACKED_PAPER_EDGE, PRINT_EDGE_CONFIRMATION

    monkeypatch.setenv("ENABLE_PRECISE_PHOTO_CROPS", "true")
    monkeypatch.setenv("ENABLE_PHOTO_ORIENTATION_CORRECTION", "false")
    monkeypatch.setenv("ENABLE_PROTECTED_REGION_GUARD", "false")
    get_settings.cache_clear()
    height, width = 570, 670
    page = np.full((height, width, 3), 242, np.uint8)
    page[74:413, 330:551] = (47, 68, 81)
    page[123:225, 490:551] = 232
    _, _, photo_id = await _seed_review_data(page)

    def vertices(points):
        return [{"x": x / (width - 1), "y": y / (height - 1)} for x, y in points]

    original = vertices([(330, 74), (551, 74), (560, 412), (330, 412)])
    corrected = vertices([(330, 74), (550, 74), (550, 412), (330, 412)])
    traces = [vertices([(550, 84), (550, 117)]), vertices([(550, 235), (550, 404)])]

    def box(quad):
        return {"x1": min(point["x"] for point in quad), "x2": max(point["x"] for point in quad),
                "y1": min(point["y"] for point in quad), "y2": max(point["y"] for point in quad)}

    async with get_session_factory()() as session:
        photo = await session.get(ExtractedPhoto, photo_id)
        photo.bounding_box = box(original)
        photo.segmentation_mask = {"source": "yolo_seg_mask", "polygon": original}
        photo.enhancement_applied = {"correction": {"crop_modes": {"full_print": {
            "status": "review_needed", "warp_quad": original, "suggested_edge_source": POSSIBLE_BACKED_PAPER_EDGE,
            "suggested_edge_traces": traces,
        }}}}
        await session.commit()

    url = f"/api/v1/photos/{photo_id}"

    async def save(quad):
        response = await client.patch(f"{url}/bounding-box", json={
            "bounding_box": box(quad), "segmentation_mask": {
                "source": "manual_quad", "polygon": quad, "warp_quad": quad,
                "image_quad": vertices([(340, 86), (537, 86), (537, 398), (340, 398)]),
                "print_size_mm": {"width": 110, "height": 169},
            },
        })
        assert response.status_code == 200, response.text
        return response.json()["enhancement_applied"]["correction"]

    async def verify_exports(expected):
        for variant in ("full_print", "image_only", "source_corrected_trimmed"):
            assert (await client.get(f"{url}/image?variant={variant}")).status_code == expected
        assert (await client.get(f"{url}/image?variant=source_corrected_untrimmed")).status_code == 200

    partly_edited = [*corrected[:2], *original[2:]]
    for quad in (original, partly_edited):
        saved = await save(quad)
        assert saved["manual_paper_boundary"]["backed_edge_verified"] is False
        modes = saved["crop_modes"]
        assert all(mode["status"] == "review_needed" for mode in modes.values())
        assert PRINT_EDGE_CONFIRMATION in modes["full_print"]["reason_codes"]
        assert modes["full_print"]["suggested_edge_traces"] == traces
        await verify_exports(404)

    async with get_session_factory()() as session:
        photo = await session.get(ExtractedPhoto, photo_id)
        metadata = dict(photo.enhancement_applied)
        stale = dict(metadata["correction"])
        stale.pop("manual_paper_boundary")
        stale["crop_modes"] = {mode: {**value, "status": "ready"} for mode, value in stale["crop_modes"].items()}
        metadata["correction"] = stale
        metadata["variants"] = {
            **metadata["variants"], **{mode: {"path": photo.original_storage_path} for mode in stale["crop_modes"]},
        }
        photo.enhancement_applied = metadata
        await session.commit()
    await verify_exports(404)
    replay = await client.post(f"{url}/reprocess")
    assert replay.status_code == 200, replay.text
    assert replay.json()["enhancement_applied"]["correction"]["manual_paper_boundary"]["backed_edge_verified"] is False

    saved = await save(corrected)
    assert saved["manual_paper_boundary"]["backed_edge_verified"] is True
    assert all(mode["status"] == "ready" for mode in saved["crop_modes"].values())
    assert "suggested_edge_traces" not in saved["crop_modes"]["full_print"]
    await verify_exports(200)

    reverted = await save(original)
    assert reverted["manual_paper_boundary"]["backed_edge_verified"] is False
    await verify_exports(404)


async def test_manual_crop_cannot_remove_a_whole_protected_source_region(client, monkeypatch):
    from app.config import get_settings
    from app.database import get_session_factory
    from app.models import ExtractedPhoto, Page

    monkeypatch.setenv("ENABLE_PRECISE_PHOTO_CROPS", "true")
    monkeypatch.setenv("ENABLE_PROTECTED_REGION_GUARD", "true")
    monkeypatch.setenv("ENABLE_PHOTO_ORIENTATION_CORRECTION", "false")
    get_settings.cache_clear()
    image = np.full((480, 640, 3), 232, np.uint8)
    image[85:395, 235:593] = (59, 90, 131)
    cv2.putText(image, "A", (257, 252), cv2.FONT_HERSHEY_SIMPLEX, .8, (241, 241, 241), 2)
    _, page_id, photo_id = await _seed_review_data(image)
    url = f"/api/v1/photos/{photo_id}"

    def shape(left, top, right, bottom):
        return [{"x": x / 639, "y": y / 479} for x, y in [(left, top), (right, top), (right, bottom), (left, bottom)]]

    async def save(outer, inner):
        return await client.patch(f"{url}/bounding-box", json={"bounding_box": {
            "x1": outer[0]["x"], "x2": outer[2]["x"], "y1": outer[0]["y"], "y2": outer[2]["y"],
        }, "segmentation_mask": {
            "source": "manual_quad", "polygon": outer, "warp_quad": outer, "image_quad": inner,
            "print_size_mm": {"width": 160, "height": 140},
        }})

    outer, motif = shape(235, 85, 592, 394), shape(245, 97, 580, 382)
    overextended = shape(25, 85, 592, 394)
    text = {"type": "text", "source": "manual", "pinned": True,
            "box": {"x1": 251, "y1": 225, "x2": 282, "y2": 258}}
    outside = {"type": "face", "source": "opencv_haarcascade_frontalface_default",
               "box": {"x1": 40, "y1": 150, "x2": 105, "y2": 215}}
    async with get_session_factory()() as session:
        page = await session.get(Page, page_id)
        page.processing_metadata = {"steps": {"protected_regions": {"enabled": True, "regions": [outside, text]}}}
        photo = await session.get(ExtractedPhoto, photo_id)
        photo.segmentation_mask = {
            "source": "classical_boundary", "polygon": outer, "visible_polygon": outer, "warp_quad": overextended,
        }
        photo.bounding_box = {"x1": outer[0]["x"], "x2": outer[2]["x"], "y1": outer[0]["y"], "y2": outer[2]["y"]}
        await session.commit()

    saved = await save(outer, motif)
    assert saved.status_code == 200, saved.text
    async with get_session_factory()() as session:
        photo = await session.get(ExtractedPhoto, photo_id)
        photo.segmentation_mask = {**photo.segmentation_mask, "visible_polygon": shape(270, 85, 592, 394)}
        await session.commit()
    original = saved.json()
    original["segmentation_mask"]["visible_polygon"] = shape(270, 85, 592, 394)
    variants = ("full_print", "image_only")

    async def exports():
        return [await client.get(f"{url}/image", params={"variant": variant}) for variant in variants]

    downloads = await exports()
    assert [response.status_code for response in downloads] == [200, 200]
    for lost_mode, new_outer, new_inner in (
        ("full_print", shape(313, 85, 592, 394), shape(325, 97, 580, 382)),
        ("image_only", outer, shape(315, 97, 580, 382)),
    ):
        rejected = await save(new_outer, new_inner)
        assert rejected.status_code == 422, rejected.text
        assert rejected.json()["detail"] == {
            "code": "protected_region_removed", "mode": lost_mode,
            "regions": [{"x1": 251 / 639, "y1": 225 / 479, "x2": 282 / 639, "y2": 258 / 479}],
        }
        current = (await client.get(url)).json()
        assert current["segmentation_mask"] == original["segmentation_mask"]
        assert current["enhancement_applied"]["review_provenance"] == original["enhancement_applied"]["review_provenance"]
        repeated = await exports()
        assert [response.status_code for response in repeated] == [200, 200]
        assert [response.content for response in repeated] == [response.content for response in downloads]
    assert (await save(outer, shape(246, 97, 580, 382))).status_code == 200

    # A stored motif from an older release is checked again on rendering.
    async with get_session_factory()() as session:
        photo = await session.get(ExtractedPhoto, photo_id)
        photo.segmentation_mask = {**photo.segmentation_mask, "image_quad": shape(315, 97, 580, 382)}
        await session.commit()
    replay = await client.post(f"{url}/reprocess")
    assert replay.status_code == 200, replay.text
    assert "BORDER_PROTECTED_CONTENT_VETO" in replay.json()["enhancement_applied"]["correction"]["crop_modes"]["image_only"]["reason_codes"]
    assert [response.status_code for response in await exports()] == [200, 404]


async def test_album_page_mark_is_bound_to_the_reviewed_crop_and_source(client, monkeypatch, tmp_path):
    from app.config import get_settings
    from app.database import get_session_factory
    from app.models import ExtractedPhoto, Page
    from app.pipeline.manual_paper_extent import MANUAL_PAPER_EXTENT_SOURCE
    from app.pipeline.print_outline_suggestion import POSSIBLE_OVEREXTENDED_PRINT_EDGE
    from app.pipeline.protected_regions import protected_region_digest
    from app.pipeline.rounded_corner_review import ROUNDED_CORNER_SOURCE

    monkeypatch.setenv("ENABLE_PRECISE_PHOTO_CROPS", "true")
    monkeypatch.setenv("ENABLE_PROTECTED_REGION_GUARD", "true")
    monkeypatch.setenv("ENABLE_PHOTO_ORIENTATION_CORRECTION", "false")
    get_settings.cache_clear()
    image = np.full((480, 640, 3), 232, np.uint8)
    image[85:395, 270:593] = (59, 90, 131)
    album_id, page_id, photo_id = await _seed_review_data(image)
    mark = {"type": "face", "source": "opencv_haarcascade_frontalface_default",
            "box": {"x1": 221, "y1": 209, "x2": 294, "y2": 267}}
    trusted = {"type": "text", "source": "manual", "box": {"x1": 240, "y1": 280, "x2": 255, "y2": 300}}
    review_id = str(uuid4())

    def shape(left, top, right, bottom):
        return [{"x": x / 639, "y": y / 479} for x, y in [(left, top), (right, top), (right, bottom), (left, bottom)]]

    original = shape(235, 85, 592, 394)
    edited = shape(270, 85, 592, 394)

    def box(quad):
        return {"x1": quad[0]["x"], "x2": quad[2]["x"], "y1": quad[0]["y"], "y2": quad[2]["y"]}

    async def write_protection(regions, *, renew_offer=False):
        async with get_session_factory()() as session:
            page = await session.get(Page, page_id)
            page.processing_metadata = {"steps": {"protected_regions": {"enabled": True, "regions": regions}}}
            if renew_offer:
                photo = await session.get(ExtractedPhoto, photo_id)
                photo.bounding_box = box(original)
                photo.segmentation_mask = {"source": "yolo_seg_mask", "polygon": original}
                photo.enhancement_applied = {"correction": {
                    "crop_modes": {"full_print": {"status": "review_needed", "warp_quad": original,
                        "outline_preview": {"source": POSSIBLE_OVEREXTENDED_PRINT_EDGE}}},
                    "album_page_mark_review": {"id": review_id, "page_sha256": page.file_hash_sha256,
                        "protected_regions_sha256": protected_region_digest(regions), "region": mark},
                }}
            await session.commit()

    url = f"/api/v1/photos/{photo_id}"

    async def save(quad, selected=None):
        payload = {"bounding_box": box(quad), "segmentation_mask": {
            "source": "manual_quad", "polygon": quad, "warp_quad": quad, "image_quad": shape(281, 97, 580, 382),
            "print_size_mm": {"width": 161, "height": 154.5},
        }}
        if selected:
            payload["album_page_mark_review_id"] = selected
        return await client.patch(f"{url}/bounding-box", json=payload)

    await write_protection([mark, trusted], renew_offer=True)
    response = await save(edited, review_id)
    assert response.status_code == 422 and response.json()["detail"] == "protected_region_cut"
    await write_protection([mark], renew_offer=True)
    assert (await save(edited)).status_code == 422
    assert (await save(edited, str(uuid4()))).status_code == 409
    response = await save(original, review_id)
    assert response.status_code == 422 and response.json()["detail"] == "album_page_mark_inside_crop"
    response = await save(edited, review_id)
    assert response.status_code == 200, response.text
    result = response.json()["enhancement_applied"]
    assert result["review_provenance"]["reviewed_album_mark"]["warp_quad"] == edited
    assert result["correction"]["protected_regions"]["reviewed_album_mark_id"] == review_id

    async def download_codes():
        return [(await client.get(f"{url}/image", params={"variant": variant})).status_code
                for variant in ("full_print", "image_only", "source_corrected_trimmed")]

    assert await download_codes() == [200, 200, 200]
    assert (await client.post(f"{url}/reprocess")).status_code == 200
    assert await download_codes() == [200, 200, 200]
    assert (await save(edited, review_id)).status_code == 409
    assert (await save(original)).status_code == 422
    assert await download_codes() == [200, 200, 200]

    preview = tmp_path / "held-corner.png"
    assert cv2.imwrite(str(preview), np.zeros((12, 12, 4), np.uint8))
    async with get_session_factory()() as session:
        photo = await session.get(ExtractedPhoto, photo_id)
        metadata = photo.enhancement_applied
        correction = metadata["correction"]
        modes = correction["crop_modes"]
        full = {**modes["full_print"], "status": "review_needed", "suggested_edge_source": ROUNDED_CORNER_SOURCE,
                "outline_preview": {"source": ROUNDED_CORNER_SOURCE},
                "source_strip_review": {"source": MANUAL_PAPER_EXTENT_SOURCE}}
        photo.enhancement_applied = {**metadata, "correction": {**correction, "full_print_preview_path": str(preview),
                                                             "crop_modes": {**modes, "full_print": full}}}
        await session.commit()

    variant = {"variant": "full_print_preview"}
    assert (await client.get(f"{url}/image", params=variant)).status_code == 200
    async with get_session_factory()() as session:
        photo = await session.get(ExtractedPhoto, photo_id)
        metadata = photo.enhancement_applied
        correction = metadata["correction"]
        protection = {key: value for key, value in correction["protected_regions"].items()
                      if key != "reviewed_album_mark_id"}
        photo.enhancement_applied = {**metadata, "correction": {**correction, "protected_regions": protection}}
        await session.commit()

    assert (await client.get(f"{url}/image", params=variant)).status_code == 404
    cropped = (await client.get(url)).json()["enhancement_applied"]["correction"]["crop_modes"]["full_print"]
    assert "source_strip_review" not in cropped and "outline_preview" not in cropped
    assert (await client.post(f"{url}/reprocess")).status_code == 200
    assert await download_codes() == [200, 200, 200]

    await write_protection([mark, trusted])
    assert await download_codes() == [404, 404, 404]
    detail = (await client.get(url)).json()
    assert detail["crop_download_urls"] == {"full_print": None, "image_only": None}
    assert detail["urls"]["full_print"] is None
    album, = (await client.get(f"/api/v1/albums/{album_id}/photos")).json()
    assert album["crop_download_urls"] == detail["crop_download_urls"]
    task, = (await client.get("/api/v1/review/queue", params={"type": "geometry", "photo_id": str(photo_id)})).json()
    assert task["status"] == "review_needed" and task["payload"]["variant_urls"]["full_print"] is None
    assert (await client.post(f"{url}/reprocess")).status_code == 200
    assert await download_codes() == [404, 404, 404]


@pytest.mark.parametrize("operation", ["add", "edit"])
@pytest.mark.parametrize(("y1", "y2", "padding_only"), [(235, 400, False), (260, 400, True), (490, 570, True)])
async def test_manual_draft_can_explain_a_protected_cut_without_changing_the_page(
    client, monkeypatch, operation, y1, y2, padding_only,
):
    from app.config import get_settings
    from app.database import get_session_factory
    from app.models import AuditLog, ExtractedPhoto, Page
    from sqlalchemy import select

    monkeypatch.setenv("ENABLE_PRECISE_PHOTO_CROPS", "true")
    monkeypatch.setenv("ENABLE_PROTECTED_REGION_GUARD", "true")
    get_settings.cache_clear()
    width, height = 800, 600
    album_id, page_id, photo_id = await _seed_review_data(np.full((height, width, 3), 150, np.uint8))
    region = {"source": "opencv_haarcascade_frontalface_default", "type": "face", "margin_ratio": .18,
              "box": {"x1": 280, "y1": y1, "x2": 400, "y2": y2}}
    page_metadata = {"photo_count_revision": str(uuid4()), "steps": {"protected_regions": {"regions": [region]}}}
    async with get_session_factory()() as session:
        page = await session.get(Page, page_id)
        page.processing_metadata = page_metadata
        if operation == "edit":
            # Exercise a newly encountered mark. Removing content from a previous crop has a separate error.
            photo = await session.get(ExtractedPhoto, photo_id)
            previous = [{"x": x, "y": y} for x, y in ((.66, .2), (.82, .2), (.82, .9), (.66, .9))]
            photo.bounding_box = {"x1": .66, "y1": .2, "x2": .82, "y2": .9}
            photo.segmentation_mask = {**photo.segmentation_mask, "polygon": previous, "warp_quad": previous}
        await session.commit()

    quad = [
        {"x": x / (width - 1), "y": y / (height - 1)} for x, y in ((100, 250), (700, 250), (700, 480), (100, 480))
    ]
    if operation == "edit":
        send, url = client.patch, f"/api/v1/photos/{photo_id}/bounding-box"
        payload = {"bounding_box": {"x1": quad[0]["x"], "y1": quad[0]["y"], "x2": quad[2]["x"], "y2": quad[2]["y"]},
                   "segmentation_mask": {"source": "manual_quad", "polygon": quad, "warp_quad": quad}}
    else:
        send, url = client.post, f"/api/v1/pages/{page_id}/manual-photos"
        payload = {"request_id": str(uuid4()), "warp_quad": quad}
    photos_url = f"/api/v1/albums/{album_id}/photos"
    before = (await client.get(photos_url)).json()
    assert len(before) == 1

    legacy = await send(url, json=payload)
    assert (legacy.status_code, legacy.json()["detail"]) == (422, "protected_region_cut")
    detailed = await send(url, json=payload, params={"explain_protected_cut": "true"})
    assert detailed.status_code == 422
    assert detailed.json()["detail"] == {"code": "protected_region_cut", "regions": [{
        "x1": 280 / 799, "y1": y1 / 599, "x2": 400 / 799, "y2": y2 / 599, "padding_only": padding_only,
    }]}
    assert (await client.get(photos_url)).json() == before
    async with get_session_factory()() as session:
        assert (await session.get(Page, page_id)).processing_metadata == page_metadata
        assert await session.scalar(select(AuditLog.id).where(AuditLog.entity_id == photo_id)) is None


@pytest.mark.parametrize("case", ["disjoint", "raw_cut", "raw_contained", "near_raw", "verified", "different_outline"])
async def test_only_exact_contested_outline_outside_advisory_core_gets_ephemeral_preview(client, monkeypatch, case):
    from base64 import b64decode

    from app.config import get_settings
    from app.database import get_session_factory
    from app.models import ExtractedPhoto, Page
    from app.pipeline.crossing_prints import CONTESTED_TILE_OUTLINE

    monkeypatch.setenv("ENABLE_PRECISE_PHOTO_CROPS", "true")
    monkeypatch.setenv("ENABLE_PROTECTED_REGION_GUARD", "true")
    get_settings.cache_clear()
    width, height = 800, 600
    album_id, page_id, photo_id = await _seed_review_data(np.full((height, width, 3), 150, np.uint8))

    def rect(left, top, right, bottom):
        return [{"x": x / (width - 1), "y": y / (height - 1)}
                for x, y in ((left, top), (right, top), (right, bottom), (left, bottom))]

    old = rect(710, 60, 785, 560)
    guide = rect(100, 250, 700, 480)
    y1, y2 = {"raw_cut": (235, 400), "raw_contained": (260, 400), "near_raw": (481, 570)}.get(case, (490, 570))
    region = {"type": "face", "source": "opencv_haarcascade_frontalface_default", "margin_ratio": .18,
              "verified": case == "verified", "box": {"x1": 280, "y1": y1, "x2": 400, "y2": y2}}
    async with get_session_factory()() as session:
        page = await session.get(Page, page_id)
        page.processing_metadata = {"photo_count_revision": str(uuid4()), "steps": {"protected_regions": {"regions": [region]}}}
        photo = await session.get(ExtractedPhoto, photo_id)
        photo.segmentation_mask = {"source": "yolo_seg_mask", "polygon": old, "warp_quad": old,
                                   "crossing_print_suggestion": guide, "crossing_print_suggestion_source": CONTESTED_TILE_OUTLINE}
        photo.enhancement_applied = {"correction": {"crop_modes": {
            mode: {"status": "review_needed"} for mode in ("full_print", "image_only")
        }}}
        await session.commit()

    url = f"/api/v1/photos/{photo_id}"
    queue = "/api/v1/review/queue"
    params = {"album_id": str(album_id), "type": "segmentation"}
    initial = (await client.get(url)).json()
    count = (await client.get(queue, params=params)).json()
    entered = rect(100, 250, 695, 480) if case == "different_outline" else guide
    edit = await client.patch(f"{url}/bounding-box", params={"explain_protected_cut": "true"}, json={
        "bounding_box": dict(x1=entered[0]["x"], y1=entered[0]["y"], x2=entered[2]["x"], y2=entered[2]["y"]),
        "segmentation_mask": {"source": "manual_quad", "polygon": entered, "warp_quad": entered},
    })
    assert edit.status_code == 422, edit.text[:300]
    feedback = edit.json()["detail"]
    assert feedback["code"] == "protected_region_cut"
    if case == "disjoint":
        assert edit.headers["cache-control"] == "private, no-store"
        preview = feedback["protected_margin_preview"]
        assert preview["estimated_aspect"] is True
        pixels = cv2.imdecode(np.frombuffer(b64decode(preview["png_base64"], validate=True), dtype=np.uint8), cv2.IMREAD_COLOR)
        assert pixels.shape == (231, 601, 3)
        assert np.all(pixels == 150)
    else:
        assert "protected_margin_preview" not in feedback
    assert (await client.get(url)).json() == initial
    assert (await client.get(queue, params=params)).json() == count
    for variant in ("full_print", "image_only", "source_corrected_trimmed", "protected_margin_review"):
        assert (await client.get(f"{url}/image", params={"variant": variant})).status_code == 404


async def test_manual_print_margin_review_keeps_core_and_binds_confirmation(submit_geometry_review, client, monkeypatch):
    from app.config import get_settings
    from app.database import get_session_factory
    from app.models import ExtractedPhoto, Page
    from app.pipeline.off_print_advisory import manual_margin_advisory
    from app.pipeline.protected_regions import validate_protected_regions

    monkeypatch.setenv("ENABLE_PRECISE_PHOTO_CROPS", "true")
    monkeypatch.setenv("ENABLE_PROTECTED_REGION_GUARD", "true")
    monkeypatch.setenv("ENABLE_PHOTO_ORIENTATION_CORRECTION", "false")
    get_settings.cache_clear()
    height, width = 670, 780
    lab = np.full((height, width, 3), (185, 129, 146), np.uint8)
    lab[..., 0] = np.rint(185 + 33 * np.cos((np.arange(height) - 398)[:, None] * (2 * np.pi / 12)))
    lab[95:390, 100:603] = (113, 138, 129)
    lab[406:600, 103:606] = (137, 133, 126)
    raster = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    album_id, page_id, other_id = await _seed_review_data(raster)

    def shape(left, top, right, bottom):
        return [{"x": x / (width - 1), "y": y / (height - 1)}
                for x, y in [(left, top), (right, top), (right, bottom), (left, bottom)]]

    mark = {"type": "face", "source": "opencv_haarcascade_frontalface_default", "margin_ratio": .18,
            "box": {"x1": 293, "y1": 407, "x2": 403, "y2": 517}}
    source_outline = np.float32([[100, 95], [602, 95], [602, 394], [100, 394]])
    violations = validate_protected_regions(source_outline, [mark], width=width, height=height)["violations"]
    flat_gap = raster.copy()
    flat_gap[390:406, 220:470] = raster[398, 52]
    assert not manual_margin_advisory(flat_gap, source_outline, [mark], violations, [])
    gap_claim = np.float32([[320, 390], [395, 390], [395, 398], [320, 398]])
    assert not manual_margin_advisory(raster, source_outline, [mark], violations, [gap_claim])

    async def write_regions(regions, *, new_count=False):
        async with get_session_factory()() as session:
            page = await session.get(Page, page_id)
            metadata = page.processing_metadata or {"original_size": {"width": width, "height": height}}
            page.processing_metadata = {**metadata, "steps": {"protected_regions": {"regions": regions}},
                                        **({"photo_count_revision": str(uuid4())} if new_count else {})}
            other = await session.get(ExtractedPhoto, other_id)
            other.segmentation_mask = {"source": "manual_quad", "polygon": shape(645, 95, 755, 250)}
            await session.commit()

    add_url = f"/api/v1/pages/{page_id}/manual-photos"
    aligned = shape(100, 95, 602, 390)
    wide = shape(100, 95, 602, 394)
    size = {"width": 170, "height": 100}
    payload = {"request_id": str(uuid4()), "warp_quad": wide, "print_size_mm": size}
    core_cut = {**mark, "box": {**mark["box"], "y1": 389, "y2": 499}}
    for hard_hit in ({**mark, "pinned": True}, core_cut, {**mark, "source": "manual"}):
        await write_regions([hard_hit])
        denied = await client.post(add_url, json=payload)
        assert denied.status_code == 422 and denied.json()["detail"] == "protected_region_cut"
    await write_regions([mark])
    created = await client.post(add_url, json=payload)
    assert created.status_code == 201, created.text
    photo = created.json()
    assert photo["segmentation_mask"]["warp_quad"] == wide
    assert (await client.post(add_url, json=payload)).json()["id"] == photo["id"]
    url = f"/api/v1/photos/{photo['id']}"
    modes = ("full_print", "image_only", "source_corrected_trimmed", "source_corrected_untrimmed")

    async def exports():
        return [(await client.get(f"{url}/image", params={"variant": mode})).status_code for mode in modes]

    def offer(photo):
        return photo["enhancement_applied"]["correction"]["album_page_mark_review"]

    async def save(quad, mark_id=None, indices=None):
        return await client.patch(f"{url}/bounding-box", json={
            "bounding_box": {"x1": quad[0]["x"], "y1": quad[0]["y"], "x2": quad[2]["x"], "y2": quad[2]["y"]},
            "segmentation_mask": {"source": "manual_quad", "warp_quad": quad, "polygon": quad, "print_size_mm": size},
            **({"album_page_mark_review_id": mark_id, "album_page_mark_review_indices": indices} if mark_id else {}),
        })

    assert await exports() == [404, 404, 404, 200]
    assert offer(photo)["regions"][0]["requires_edge_alignment"]
    unaligned = await save(wide, offer(photo)["id"], [0])
    assert unaligned.status_code == 422 and unaligned.json()["detail"] == "outside_manual_print_edge_unresolved"
    ordinary = await submit_geometry_review("/api/v1/review/events", json={
        "task_type": "geometry", "task_id": photo["id"], "answer": "correct", "idempotency_key": str(uuid4()),
    })
    assert ordinary.status_code == 409
    monkeypatch.setenv("ENABLE_PROTECTED_REGION_GUARD", "false")
    get_settings.cache_clear()
    assert (await client.post(f"{url}/reprocess")).status_code == 200
    assert await exports() == [404, 404, 404, 200]
    monkeypatch.setenv("ENABLE_PROTECTED_REGION_GUARD", "true")
    get_settings.cache_clear()
    await write_regions([])
    assert (await client.post(f"{url}/reprocess")).status_code == 200
    assert (await exports())[:3] == [404, 404, 404]
    other_edge_edit = shape(100, 90, 602, 394)
    blocked_edit = await save(other_edge_edit)
    assert blocked_edit.status_code == 422 and blocked_edit.json()["detail"] == "album_page_mark_review_stale"
    await write_regions([mark])
    saved = await save(aligned)
    assert saved.status_code == 200, saved.text
    current_offer = offer(saved.json())
    assert current_offer["context"] == "outside_manual_print"
    assert not current_offer["regions"][0]["requires_edge_alignment"]
    assert (await save(aligned, current_offer["id"], None)).status_code == 409
    approved = await save(aligned, current_offer["id"], [0])
    assert approved.status_code == 200, approved.text
    assert approved.json()["enhancement_applied"]["correction"]["protected_regions"]["regions_checked"] == 1
    assert (await exports())[0] == 200
    exported = await exports()
    assert (await client.post(f"{url}/reprocess")).status_code == 200
    assert await exports() == exported
    assert len((await client.get(f"/api/v1/albums/{album_id}/photos")).json()) == 2
    async with get_session_factory()() as session:
        page = await session.get(Page, page_id)
        assert page.processing_metadata["steps"]["protected_regions"]["regions"] == [mark]
    await write_regions([mark], new_count=True)
    assert (await exports())[:3] == [404, 404, 404]
    assert (await client.post(f"{url}/reprocess")).status_code == 200
    assert (await exports())[:3] == [404, 404, 404]


async def test_manual_corner_shape_remains_in_review_with_dimensions_and_late_claim(submit_geometry_review, client, monkeypatch, rounded_corner_page):
    from app.config import get_settings
    from app.database import get_session_factory
    from app.models import ExtractedPhoto
    from app.pipeline import perspective
    from app.pipeline.rounded_corner_review import ROUNDED_CORNER_REVIEW, ROUNDED_CORNER_SOURCE, ROUNDED_CORNER_WEDGES

    monkeypatch.setenv("ENABLE_PRECISE_PHOTO_CROPS", "true")
    monkeypatch.setenv("ENABLE_PHOTO_ORIENTATION_CORRECTION", "false")
    get_settings.cache_clear()
    raster, native, *_ = rounded_corner_page
    _, page_id, other_id = await _seed_review_data(raster)

    def to_normalized(points):
        return [{"x": float(x) / 484, "y": float(y) / 484} for x, y in points]

    async with get_session_factory()() as session:
        other = await session.get(ExtractedPhoto, other_id)
        other.segmentation_mask = {"polygon": to_normalized([(438, 30), (478, 30), (478, 95), (438, 95)])}
        await session.commit()

    size = {"width": 123, "height": 124}
    quad = to_normalized(native)
    page_url = f"/api/v1/pages/{page_id}/manual-photos"
    created = await client.post(page_url, json={"request_id": str(uuid4()), "warp_quad": quad})
    assert created.status_code == 201, created.text
    photo = created.json()
    url = f"/api/v1/photos/{photo['id']}"
    correction = photo["enhancement_applied"]["correction"]
    mode = correction["crop_modes"]["full_print"]
    assert mode["outline_preview"]["source"] == ROUNDED_CORNER_SOURCE
    assert mode["warp_quad"] == correction["warp_quad"]
    assert 2 <= len(mode["suggested_edge_traces"]) == len(photo["segmentation_mask"][ROUNDED_CORNER_WEDGES]) <= 4

    async def image(mode):
        return await client.get(f"{url}/image", params={"variant": mode})

    async def codes():
        return [(await image(name)).status_code for name in (
            "full_print", "image_only", "source_corrected_trimmed", "source_corrected_untrimmed",
        )]

    original_preview = await image("source_corrected_untrimmed")
    preview = await image("full_print_preview")
    assert preview.status_code == 200 and preview.headers["content-type"] == "image/png"
    rendered = cv2.imdecode(np.frombuffer(preview.content, np.uint8), cv2.IMREAD_UNCHANGED)
    assert rendered.shape[2] == 4 and np.any(rendered[:, :, 3] == 0) and np.any(rendered[:, :, 3] == 255)
    assert await codes() == [404, 404, 404, 200]
    assert (await image("image_only_preview")).status_code == 404
    ordinary = await submit_geometry_review("/api/v1/review/events", json={
        "task_type": "geometry", "task_id": photo["id"], "answer": "correct", "idempotency_key": str(uuid4()),
    })
    assert ordinary.status_code == 409
    repeated = await client.post(f"{url}/reprocess")
    assert repeated.status_code == 200
    assert (await image("source_corrected_untrimmed")).content == original_preview.content
    assert (await image("full_print_preview")).content == preview.content

    async def save(corners):
        return await client.patch(f"{url}/bounding-box", json={
            "bounding_box": {"x1": corners[0]["x"], "y1": corners[0]["y"], "x2": corners[2]["x"], "y2": corners[2]["y"]},
            "segmentation_mask": {"source": "manual_quad", "polygon": corners, "warp_quad": corners, "print_size_mm": size},
        })

    saved = await save(quad)
    assert saved.status_code == 200, saved.text
    modes = saved.json()["enhancement_applied"]["correction"]["crop_modes"]
    assert all(mode["status"] == "review_needed" and ROUNDED_CORNER_REVIEW in mode["reason_codes"] for mode in modes.values())
    assert await codes() == [404, 404, 404, 200]
    assert (await image("full_print_preview")).status_code == 200
    inset = to_normalized([(90, 90), (388, 90), (388, 390), (90, 390)])
    moved = await save(inset)
    assert moved.status_code == 200 and await codes() == [404, 404, 404, 200]
    assert (await image("full_print_preview")).status_code == 404
    assert (await save(quad)).status_code == 200
    with monkeypatch.context() as context:
        context.setattr(perspective, "measure_rounded_corners", lambda *_: None)
        missing = await client.post(f"{url}/reprocess")
        assert missing.status_code == 200 and await codes() == [404, 404, 404, 200]
        assert (await image("full_print_preview")).status_code == 404

    assert (await client.post(f"{url}/reprocess")).status_code == 200
    assert (await image("full_print_preview")).status_code == 200
    claim = to_normalized([(35, 35), (89, 35), (89, 90), (35, 90)])
    claimed = await client.post(page_url, json={"request_id": str(uuid4()), "warp_quad": claim})
    assert claimed.status_code == 201, claimed.text
    assert (await image("full_print_preview")).status_code == 404
    assert await codes() == [404, 404, 404, 200]


async def test_capture_outline_hold_survives_manual_edit_and_legacy_exports(client, monkeypatch, tmp_path):
    from copy import deepcopy

    from app.config import get_settings
    from app.database import get_session_factory
    from app.models import ExtractedPhoto
    from app.pipeline.manual_paper_boundary import MANUAL_CAPTURE_OUTLINE_REVIEW
    from app.services import pipeline

    monkeypatch.setenv("AUTO_ANALYZE_UPLOADS", "true")
    monkeypatch.setenv("ENABLE_PHOTO_ORIENTATION_CORRECTION", "false")
    monkeypatch.setenv("MAX_UPLOAD_SIZE", "2000000")
    monkeypatch.setenv("MAX_REQUEST_SIZE", "2100000")
    monkeypatch.setenv("YOLO_MODEL_PATH", str(tmp_path / "absent.onnx"))
    monkeypatch.setattr(pipeline, "run_paddle_ocr", lambda *_: [])
    get_settings.cache_clear()
    source = Path(__file__).resolve().parents[3] / "data" / "golden_fixtures" / "page_04.jpg"
    album = await client.post("/api/v1/albums", json={"name": "Original capture outline"})
    upload = await client.post(
        f"/api/v1/albums/{album.json()['id']}/pages/upload",
        files={"files": (source.name, source.read_bytes(), "image/jpeg")},
    )
    assert upload.status_code == 201, upload.text
    photos = (await client.get(f"/api/v1/albums/{album.json()['id']}/photos")).json()
    assert len(photos) == 2
    right, = [photo for photo in photos if photo["bounding_box"]["x2"] == 1]
    original = right["enhancement_applied"]["correction"]["crop_modes"]["full_print"]
    assert {"PRINT_BOUNDARY_UNSUPPORTED", "PRINT_AT_CAPTURE_EDGE"} <= set(original["reason_codes"])
    url = f"/api/v1/photos/{right['id']}"
    quad = [{"x": x / 2559, "y": y / 1919} for x, y in [(1617, 728), (2350, 736), (2298, 1798), (1596, 1814)]]
    edit = {
        "bounding_box": {"x1": 1596 / 2559, "y1": 728 / 1919, "x2": 2350 / 2559, "y2": 1814 / 1919},
        "segmentation_mask": {
            "source": "manual_quad", "polygon": quad, "warp_quad": quad,
            "print_size_mm": {"width": 68, "height": 100},
        },
    }

    async def assert_downloads_held():
        for variant in ("full_print", "image_only", "source_corrected_trimmed"):
            result = await client.get(f"{url}/image", params={"variant": variant})
            assert result.status_code == 404, variant

    for _ in range(2):
        saved = await client.patch(f"{url}/bounding-box", json=edit)
        assert saved.status_code == 200, saved.text
        metadata = saved.json()["enhancement_applied"]
        assert metadata["review_provenance"]["original_prediction"]["capture_outline_unverified"] is True
        correction = metadata["correction"]
        assert correction["manual_paper_boundary"]["capture_outline_verified"] is False
        assert all(MANUAL_CAPTURE_OUTLINE_REVIEW in mode["reason_codes"] for mode in correction["crop_modes"].values())
        await assert_downloads_held()

    # Restore the shape of a pre-fix ready row, including a live file in each
    # variant. Older reviews only persisted the original rendered quad.
    async with get_session_factory()() as session:
        photo = await session.get(ExtractedPhoto, UUID(right["id"]))
        metadata = deepcopy(photo.enhancement_applied)
        metadata["review_provenance"]["original_prediction"].pop("capture_outline_unverified")
        metadata["correction"].pop("manual_paper_boundary")
        for mode in metadata["correction"]["crop_modes"].values():
            mode.update(status="ready", reason_codes=[])
        metadata["variants"] = {name: {"path": str(source)} for name in ("full_print", "image_only", "source_corrected_trimmed")}
        photo.enhancement_applied = metadata
        await session.commit()
    await assert_downloads_held()

    replay = await client.post(f"{url}/reprocess")
    assert replay.status_code == 200, replay.text
    assert replay.json()["enhancement_applied"]["correction"]["manual_paper_boundary"]["requires_review"] is True
    await assert_downloads_held()

    pages_url = f"/api/v1/albums/{album.json()['id']}/pages"
    page, = (await client.get(pages_url)).json()
    page_steps = page["processing_metadata"]["steps"]["segmentation"]
    trace_runs = page_steps["missing_print_edge_traces"]
    queue = await client.get("/api/v1/review/queue", params={"type": "segmentation", "album_id": album.json()["id"]})
    count_review, = queue.json()
    assert count_review["payload"]["detection_count"] == 2
    assert count_review["payload"]["missing_print_edge_traces"] == trace_runs
    left_outline = [
        {"x": x / 2559, "y": y / 1919} for x, y in [(70, 749), (843, 754), (860, 1847), (140, 1843)]
    ]
    addition = await client.post(f"/api/v1/pages/{page['id']}/manual-photos", json={
        "request_id": str(uuid4()), "warp_quad": left_outline, "print_size_mm": {"width": 70, "height": 100},
    })
    assert addition.status_code == 201, addition.text
    photo = addition.json()
    metadata = photo["enhancement_applied"]
    assert metadata["review_provenance"]["paper_trace_review"]["traces"]
    assert metadata["correction"]["manual_paper_boundary"]["requires_review"] is True
    added_url = f"/api/v1/photos/{photo['id']}"
    assert (await client.post(f"{added_url}/reprocess")).status_code == 200
    for variant in ("full_print", "image_only", "source_corrected_trimmed"):
        assert (await client.get(f"{added_url}/image", params={"variant": variant})).status_code == 404
    queue = await client.get("/api/v1/review/queue", params={"type": "segmentation", "album_id": album.json()["id"]})
    count_review, = queue.json()
    assert count_review["payload"]["detection_count"] == 3
    assert not count_review["payload"]["missing_print_edge_traces"]


async def test_missing_neighbor_traces_preserve_a_curved_paper_edge_after_manual_addition(client, monkeypatch, tmp_path):
    from app.config import get_settings
    from app.services import pipeline

    monkeypatch.setenv("AUTO_ANALYZE_UPLOADS", "true")
    monkeypatch.setenv("ENABLE_PHOTO_ORIENTATION_CORRECTION", "false")
    monkeypatch.setenv("MAX_UPLOAD_SIZE", "2000000")
    monkeypatch.setenv("MAX_REQUEST_SIZE", "2100000")
    monkeypatch.setenv("YOLO_MODEL_PATH", str(tmp_path / "absent.onnx"))
    monkeypatch.setattr(pipeline, "run_paddle_ocr", lambda *_: [])
    get_settings.cache_clear()
    source = Path(__file__).resolve().parents[3] / "data" / "golden_fixtures" / "page_11.jpg"
    album_id = (await client.post("/api/v1/albums", json={"name": "Missing neighbor traces"})).json()["id"]
    album = f"/api/v1/albums/{album_id}"
    upload = await client.post(f"{album}/pages/upload", files={"files": (source.name, source.read_bytes(), "image/jpeg")})
    assert upload.status_code == 201, upload.text
    count_url = f"/api/v1/review/queue?album_id={album_id}&type=segmentation"
    task, = (await client.get(count_url)).json()
    assert task["payload"]["detection_count"] == 1
    existing, = task["payload"]["detections"]
    assert existing["segmentation_mask"]["detector_source"] == "line_pairs"
    top, outer, bottom = task["payload"]["missing_print_edge_traces"]
    assert all(len(trace) >= 8 and all(point["x"] > .60 for point in trace) for trace in (top, outer, bottom))
    assert min(point["x"] for point in outer) > .87
    assert max(point["y"] for point in top) < .41
    assert min(point["y"] for point in bottom) > .94

    corners = [{"x": x / 2559, "y": y / 1919} for x, y in [(1539, 765), (2311, 749), (2271, 1835), (1552, 1859)]]
    added = await client.post(f"/api/v1/pages/{task['page_id']}/manual-photos", json={
        "request_id": str(uuid4()), "warp_quad": corners, "print_size_mm": {"width": 70, "height": 100},
    })
    assert added.status_code == 201, added.text
    photo = added.json()
    correction = photo["enhancement_applied"]["correction"]
    curve = photo["enhancement_applied"]["review_provenance"]["paper_trace_review"]
    assert curve["sides"] == [1] and curve["traces"] == [outer]
    assert correction["manual_paper_boundary"]["requires_review"] is True
    current, = (await client.get(count_url)).json()
    assert current["payload"]["detection_count"] == 2
    assert current["payload"]["missing_print_edge_traces"] == []
    photo_url = f"/api/v1/photos/{photo['id']}"
    assert (await client.post(f"{photo_url}/reprocess")).status_code == 200
    for variant in ("full_print", "image_only", "source_corrected_trimmed"):
        assert (await client.get(f"{photo_url}/image?variant={variant}")).status_code == 404
    assert (await client.get(f"{photo_url}/image?variant=source_corrected_untrimmed")).status_code == 200


async def test_verified_recovery_of_inset_straight_print_exports_both_modes(client, monkeypatch):
    from app.config import get_settings
    from app.database import get_session_factory
    from app.models import ExtractedPhoto

    monkeypatch.setenv("ENABLE_PHOTO_ORIENTATION_CORRECTION", "false")
    monkeypatch.setenv("ENABLE_PROTECTED_REGION_GUARD", "false")
    get_settings.cache_clear()
    page = np.full((560, 720, 3), 231, dtype=np.uint8)
    page[93:475, 103:605] = 202
    page[123:445, 133:575] = [120, 151, 193]
    page[92:475, 668:675] = 105
    _, _, photo_id = await _seed_review_data(page)

    def vertices(left, top, right, bottom):
        return [{"x": x / 719, "y": y / 559} for x, y in [(left, top), (right, top), (right, bottom), (left, bottom)]]

    def box(points):
        return {"x1": points[0]["x"], "y1": points[0]["y"], "x2": points[2]["x"], "y2": points[2]["y"]}

    proposal = vertices(103, 93, 719, 474)
    async with get_session_factory()() as session:
        photo = await session.get(ExtractedPhoto, photo_id)
        photo.bounding_box = box(proposal)
        photo.segmentation_mask = {"polygon": proposal, "source": "classical_border_quad"}
        await session.commit()

    url = f"/api/v1/photos/{photo_id}"
    original = await client.post(f"{url}/reprocess")
    assert original.status_code == 200, original.text
    assert "PRINT_AT_CAPTURE_EDGE" in original.json()["enhancement_applied"]["correction"]["crop_modes"]["full_print"]["reason_codes"]
    outer, motif = vertices(103, 93, 604, 474), vertices(133, 123, 574, 444)
    saved = await client.patch(f"{url}/bounding-box", json={
        "bounding_box": box(outer),
        "segmentation_mask": {
            "source": "manual_quad", "polygon": outer, "warp_quad": outer, "image_quad": motif,
            "print_size_mm": {"width": 501, "height": 381},
        },
    })
    assert saved.status_code == 200, saved.text
    correction = saved.json()["enhancement_applied"]["correction"]
    assert correction["manual_paper_boundary"]["capture_outline_verified"] is True, correction
    for variant, expected in (("full_print", (382, 502)), ("image_only", (322, 442)), ("source_corrected_trimmed", (322, 442))):
        download = await client.get(f"{url}/image", params={"variant": variant})
        assert download.status_code == 200, variant
        image = cv2.imdecode(np.frombuffer(download.content, dtype=np.uint8), cv2.IMREAD_COLOR)
        assert image.shape[:2] == expected


async def test_measured_aspect_does_not_clear_curved_paper_review_on_golden_page_37(client, monkeypatch, tmp_path):
    from app.config import Settings, get_settings
    from app.database import get_session_factory
    from app.models import ExtractedPhoto, Page
    from app.pipeline.enhancement import prepare_for_cutout
    from app.pipeline.image_ops import load_image
    from app.pipeline.manual_paper_boundary import MANUAL_PAPER_BOUNDARY_REVIEW
    from app.pipeline.segmentation import segment_album_page

    monkeypatch.setenv("ENABLE_PHOTO_ORIENTATION_CORRECTION", "false")
    monkeypatch.setenv("ENABLE_PROTECTED_REGION_GUARD", "false")
    get_settings.cache_clear()
    source = Path(__file__).resolve().parents[3] / "data" / "golden_fixtures" / "page_37.jpg"
    image = load_image(source)
    _, page_id, photo_id = await _seed_review_data(image)
    settings = Settings(DATABASE_URL="sqlite://", REDIS_URL="redis://", YOLO_MODEL_PATH=tmp_path / "absent.onnx")
    proposal, = segment_album_page(prepare_for_cutout(image)[0], settings, source_image=image).detections
    async with get_session_factory()() as session:
        page = await session.get(Page, page_id)
        page.storage_path = str(source)
        page.file_hash_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
        photo = await session.get(ExtractedPhoto, photo_id)
        photo.bounding_box, photo.segmentation_mask = proposal.bounding_box, proposal.mask
        await session.commit()

    url = f"/api/v1/photos/{photo_id}"
    response = await client.post(f"{url}/reprocess")
    assert response.status_code == 200, response.text
    current = response.json()
    full = current["enhancement_applied"]["correction"]["crop_modes"]["full_print"]
    assert "PRINT_BOUNDARY_CURVED" in full["reason_codes"]
    quad = full["warp_quad"]
    for adjustment in (0, .5 / (image.shape[0] - 1)):
        edited = [{**point, "y": point["y"] - adjustment} for point in quad]
        saved = await client.patch(f"{url}/bounding-box", json={
            "bounding_box": current["bounding_box"],
            "segmentation_mask": {
                "source": "manual_quad", "polygon": edited, "warp_quad": edited,
                "print_size_mm": {"width": 100, "height": 140},
            },
        })
        assert saved.status_code == 200, saved.text
        correction = saved.json()["enhancement_applied"]["correction"]
        assert correction["crop_modes"]["full_print"]["suggested_edge_traces"]
        for name, mode in correction["crop_modes"].items():
            assert MANUAL_PAPER_BOUNDARY_REVIEW in mode["reason_codes"]
            assert (await client.get(f"{url}/image?variant={name}")).status_code == 404
    # Older saves retained the extended proposal but did not keep the curved
    # source diagnosis. Their ready flags must be rechecked before download.
    async with get_session_factory()() as session:
        photo = await session.get(ExtractedPhoto, photo_id)
        metadata = dict(photo.enhancement_applied)
        review = dict(metadata["review_provenance"])
        review.pop("paper_curve_review")
        correction = dict(metadata["correction"])
        correction.pop("manual_paper_boundary")
        correction["crop_modes"] = {name: {**mode, "status": "ready"} for name, mode in correction["crop_modes"].items()}
        photo.enhancement_applied = {**metadata, "review_provenance": review, "correction": correction, "variants": {
            name: {"path": photo.original_storage_path} for name in correction["crop_modes"]
        }}
        await session.commit()
    assert (await client.get(f"{url}/image?variant=full_print")).status_code == 404
    replay = await client.post(f"{url}/reprocess")
    assert replay.status_code == 200, replay.text
    assert replay.json()["enhancement_applied"]["correction"]["manual_paper_boundary"]["requires_review"]


async def test_bowed_source_review_survives_preview_invalidation_and_small_manual_edit(client, monkeypatch, tmp_path):
    from app.config import get_settings
    from app.database import get_session_factory
    from app.models import ExtractedPhoto, Page
    from app.pipeline.bowed_edge_review import POSSIBLE_BOWED_PRINT_EDGE
    from app.pipeline.manual_paper_boundary import MANUAL_PAPER_BOUNDARY_REVIEW

    monkeypatch.setenv("ENABLE_PRECISE_PHOTO_CROPS", "true")
    monkeypatch.setenv("ENABLE_CLASSICAL_SEGMENTATION_FALLBACK", "true")
    monkeypatch.setenv("ENABLE_PHOTO_ORIENTATION_CORRECTION", "false")
    monkeypatch.setenv("ENABLE_PROTECTED_REGION_GUARD", "false")
    get_settings.cache_clear()
    path = Path(__file__).resolve().parents[3] / "data" / "golden_fixtures" / "page_26.jpg"
    image = cv2.rotate(cv2.imread(str(path)), cv2.ROTATE_180)
    source = tmp_path / "golden26-180.jpeg"
    assert cv2.imwrite(str(source), image, [cv2.IMWRITE_JPEG_QUALITY, 75])
    _, page_id, photo_id = await _seed_review_data()
    height, width = image.shape[:2]

    def points(vertices):
        return [{"x": x / (width - 1), "y": y / (height - 1)} for x, y in vertices]

    async with get_session_factory()() as session:
        page = await session.get(Page, page_id)
        page.storage_path = str(source)
        page.file_hash_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
        photo = await session.get(ExtractedPhoto, photo_id)
        photo.bounding_box = {"x1": 1618 / width, "y1": 810 / height, "x2": 2256 / width, "y2": 1767 / height}
        photo.segmentation_mask = {"source": "classical_border_quad", "detector_source": "edge_contour",
                                   "polygon": points([(1618, 819), (2255, 810), (2256, 1744), (1631, 1767)])}
        await session.commit()

    url = f"/api/v1/photos/{photo_id}"
    rendered = await client.post(f"{url}/reprocess")
    assert rendered.status_code == 200, rendered.text
    original = (await client.get(url)).json()
    metadata = original["enhancement_applied"]
    full = metadata["correction"]["crop_modes"]["full_print"]
    assert full["outline_preview"]["source"] == POSSIBLE_BOWED_PRINT_EDGE
    reference = metadata["review_provenance"]["paper_curve_review"]
    assert reference["traces"] == full["suggested_edge_traces"]
    preview_url = original["urls"]["full_print_preview"]
    assert (await client.get(preview_url)).status_code == 200

    # A separate review claim intersects the strip's source search band only.
    other = await client.post(f"/api/v1/pages/{page_id}/manual-photos", json={
        "request_id": str(uuid4()), "warp_quad": points([(1595, 965), (1611, 965), (1611, 1072), (1595, 1072)]),
    })
    assert other.status_code == 201, other.text
    assert (await client.get(preview_url)).status_code == 404
    regenerated = await client.post(f"{url}/reprocess")
    assert regenerated.status_code == 200, regenerated.text
    evidence = regenerated.json()["enhancement_applied"]
    assert evidence["review_provenance"]["paper_curve_review"] == reference
    assert "outline_preview" not in evidence["correction"]["crop_modes"]["full_print"]

    edited = [dict(point) for point in full["warp_quad"]]
    edited[0]["x"] += .35 / (width - 1)
    saved = await client.patch(f"{url}/bounding-box", json={
        "bounding_box": original["bounding_box"],
        "segmentation_mask": {"source": "manual_quad", "polygon": edited, "warp_quad": edited,
                              "print_size_mm": {"width": 70, "height": 100}},
    })
    assert saved.status_code == 200, saved.text
    modes = saved.json()["enhancement_applied"]["correction"]["crop_modes"]
    assert all(MANUAL_PAPER_BOUNDARY_REVIEW in mode["reason_codes"] for mode in modes.values())
    for variant in ("full_print", "image_only", "source_corrected_trimmed"):
        assert (await client.get(f"{url}/image?variant={variant}")).status_code == 404


@pytest.mark.parametrize("guide", ["image_only", "outside_motif", "full_print"])
async def test_new_paper_claim_retires_intersecting_open_crop_guides(client, monkeypatch, guide):
    from app.config import get_settings
    from app.database import get_session_factory
    from app.models import ExtractedPhoto
    from app.pipeline.motif_boundary import OUTSIDE_CROP_SEAM_TRACES, PARTIAL_MOTIF_SEAMS
    from app.pipeline.partial_print_review import PARTIAL_PRINT_EXTENT_REVIEW, PARTIAL_PRINT_EXTENT_TRACES

    _, page_id, photo_id = await _seed_review_data()
    reasons = ["PRINT_BOUNDARY_AMBIGUOUS", "PRINT_ASPECT_UNCERTAIN"]
    held = {"status": "review_needed", "reason_codes": reasons}
    guides = [
        [{"x": .2, "y": .33}, {"x": .79, "y": .33}],
        [{"x": .79, "y": .35}, {"x": .79, "y": .75}],
        [{"x": .75, "y": .77}, {"x": .2, "y": .77}],
    ]
    modes = {"full_print": held, "image_only": held}
    mode = "image_only" if guide == "outside_motif" else guide
    field = "suggested_seam_traces" if mode == "image_only" else PARTIAL_PRINT_EXTENT_TRACES
    modes[mode] = {**held, field: guides}
    if mode == "image_only":
        modes[mode]["suggested_seam_source"] = PARTIAL_MOTIF_SEAMS
    if guide == "outside_motif":
        modes[mode][OUTSIDE_CROP_SEAM_TRACES] = [[{"x": .8, "y": .35}, {"x": .8, "y": .75}]]
    async with get_session_factory()() as session:
        photo = await session.get(ExtractedPhoto, photo_id)
        if mode == "full_print":
            photo.segmentation_mask = {
                **{key: value for key, value in photo.segmentation_mask.items() if key != "warp_quad"},
                PARTIAL_PRINT_EXTENT_TRACES: guides,
            }
        photo.enhancement_applied = {**photo.enhancement_applied, "correction": {
            "crop_modes": modes,
        }}
        await session.commit()

    photo_url = f"/api/v1/photos/{photo_id}"
    before = (await client.get(photo_url)).json()["enhancement_applied"]["correction"]["crop_modes"]
    assert before[mode][field] == guides
    # The new paper enters the relevant sampling band without crossing the measured line.
    left = .83 if guide == "outside_motif" else .8
    claim = [{"x": x, "y": y} for x, y in ((left, .25), (.97, .25), (.97, .55), (left, .55))]
    added = await client.post(f"/api/v1/pages/{page_id}/manual-photos", json={
        "request_id": str(uuid4()), "warp_quad": claim,
    })
    assert added.status_code == 201, added.text
    after = (await client.get(photo_url)).json()["enhancement_applied"]["correction"]["crop_modes"]
    expected_motif = {**held, field: guides, "suggested_seam_source": PARTIAL_MOTIF_SEAMS} if guide == "outside_motif" else held
    assert after == {"full_print": held, "image_only": expected_motif}
    for variant in ("full_print", "image_only", "image_only_preview"):
        assert (await client.get(f"{photo_url}/image", params={"variant": variant})).status_code == 404
    if mode == "full_print":
        monkeypatch.setenv("ENABLE_PRECISE_PHOTO_CROPS", "false")
        get_settings.cache_clear()
        reprocessed = await client.post(f"{photo_url}/reprocess")
        assert reprocessed.status_code == 200, reprocessed.text
        refreshed = reprocessed.json()["enhancement_applied"]["correction"]["crop_modes"]
        assert field not in refreshed[mode]
        assert all(result["status"] == "review_needed" and PARTIAL_PRINT_EXTENT_REVIEW in result["reason_codes"]
                   for result in refreshed.values())
        assert (await client.get(f"{photo_url}/image", params={"variant": "source_corrected_trimmed"})).status_code == 404


async def test_new_paper_claim_retires_a_merged_pair_keeps_crop_hold(client, monkeypatch):
    from app.config import get_settings
    from app.database import get_session_factory
    from app.models import ExtractedPhoto
    from app.pipeline.merged_print_review import MERGED_PRINT_PAIR, MERGED_PRINT_PAIR_REVIEW

    album_id, page_id, photo_id = await _seed_review_data()

    def quad(x1, y1, x2, y2):
        return [{"x": x, "y": y} for x, y in ((x1, y1), (x2, y1), (x2, y2), (x1, y2))]

    pair = {"outlines": [quad(.19, .24, .73, .48), quad(.16, .53, .72, .81)],
            "separator": [{"x": .12, "y": .51}, {"x": .82, "y": .51}]}
    async with get_session_factory()() as session:
        photo = await session.get(ExtractedPhoto, photo_id)
        mask = {key: value for key, value in photo.segmentation_mask.items() if key != "warp_quad"}
        photo.segmentation_mask = {**mask, MERGED_PRINT_PAIR: pair}
        await session.commit()
    url = f"/api/v1/photos/{photo_id}"
    assert (await client.get(url)).json()["segmentation_mask"][MERGED_PRINT_PAIR] == pair
    duplicate = await client.post(f"/api/v1/pages/{page_id}/manual-photos", json={
        "request_id": str(uuid4()), "warp_quad": pair["outlines"][0],
    })
    assert duplicate.status_code == 409 and duplicate.json()["detail"] == "merged_print_pair_claim_conflict"
    inset = quad(.21, .25, .71, .47)
    duplicate = await client.post(f"/api/v1/pages/{page_id}/manual-photos", json={
        "request_id": str(uuid4()), "warp_quad": inset, "print_size_mm": {"width": 90, "height": 100},
    })
    assert (duplicate.status_code, duplicate.json()["detail"]) == (409, "merged_print_pair_claim_conflict")
    added = await client.post(f"/api/v1/pages/{page_id}/manual-photos", json={
        "request_id": str(uuid4()), "warp_quad": quad(.6, .56, .89, .83),
    })
    assert added.status_code == 201, added.text
    assert (await client.get(url)).json()["segmentation_mask"][MERGED_PRINT_PAIR] is None
    assert len((await client.get(f"/api/v1/albums/{album_id}/photos")).json()) == 2
    broad = quad(.13, .18, .79, .87)
    stale = await client.patch(f"{url}/bounding-box", json={
        "bounding_box": {"x1": .13, "y1": .18, "x2": .79, "y2": .87}, "confirm_merged_print_as_one": True,
        "segmentation_mask": {"polygon": broad, "warp_quad": broad},
    })
    assert (stale.status_code, stale.json()["detail"]) == (409, "merged_print_pair_resolution_required")
    monkeypatch.setenv("ENABLE_PRECISE_PHOTO_CROPS", "false")
    get_settings.cache_clear()
    refreshed = await client.post(f"{url}/reprocess")
    assert refreshed.status_code == 200, refreshed.text
    modes = refreshed.json()["enhancement_applied"]["correction"]["crop_modes"]
    assert all(crop["status"] == "review_needed" and MERGED_PRINT_PAIR_REVIEW in crop["reason_codes"] for crop in modes.values())
    for variant in ("full_print", "image_only", "source_corrected_trimmed"):
        assert (await client.get(f"{url}/image", params={"variant": variant})).status_code == 404


async def test_possible_print_regions_are_read_only_and_keep_held_crops_after_a_neighbor_claim(client, monkeypatch):
    from app.config import get_settings
    from app.database import get_session_factory
    from app.models import ExtractedPhoto
    from app.pipeline.possible_print_regions import POSSIBLE_PRINT_REGIONS, POSSIBLE_PRINT_REGIONS_REVIEW

    def quad(x1, y1, x2, y2):
        return [{"x": x, "y": y} for x, y in ((x1, y1), (x2, y1), (x2, y2), (x1, y2))]

    monkeypatch.setenv("ENABLE_PRECISE_PHOTO_CROPS", "true")
    monkeypatch.setenv("ENABLE_PHOTO_ORIENTATION_CORRECTION", "false")
    get_settings.cache_clear()
    album_id, page_id, photo_id = await _seed_review_data()
    outlines = [quad(.15, .22, .75, .51), quad(.17, .54, .74, .83)]
    async with get_session_factory()() as session:
        photo = await session.get(ExtractedPhoto, photo_id)
        outer = photo.segmentation_mask["warp_quad"]
        box = photo.bounding_box
        photo.segmentation_mask = {"source": "yolo_seg_mask", "detector_source": "yolo_onnx", "polygon": outer,
                                   POSSIBLE_PRINT_REGIONS: {"outlines": outlines}}
        await session.commit()

    url = f"/api/v1/photos/{photo_id}"
    rendered = await client.post(f"{url}/reprocess")
    assert rendered.status_code == 200, rendered.text
    assert rendered.json()["segmentation_mask"][POSSIBLE_PRINT_REGIONS] == {"outlines": outlines}
    assert len((await client.get(f"/api/v1/albums/{album_id}/photos")).json()) == 1
    paper = {"source": "manual_quad", "polygon": outer, "warp_quad": outer}
    update = {"bounding_box": box, "segmentation_mask": paper}
    for request in ({"bounding_box": box}, update,
                    {**update, "confirm_print_outline": True, "segmentation_mask": {**paper, "image_quad": outlines[0]}}):
        refused = await client.patch(f"{url}/bounding-box", json=request)
        assert (refused.status_code, refused.json()["detail"]) == (409, "contrast_print_outline_review_required")
    tampered = await client.patch(f"{url}/bounding-box", json={
        **update, "segmentation_mask": {**paper, POSSIBLE_PRINT_REGIONS: {"outlines": outlines}},
    })
    assert tampered.status_code == 422

    added = await client.post(f"/api/v1/pages/{page_id}/manual-photos", json={
        "request_id": str(uuid4()), "warp_quad": quad(.62, .56, .90, .85),
    })
    assert added.status_code == 201, added.text
    assert (await client.get(url)).json()["segmentation_mask"][POSSIBLE_PRINT_REGIONS] is None
    monkeypatch.setenv("ENABLE_PRECISE_PHOTO_CROPS", "false")
    get_settings.cache_clear()
    repeated = await client.post(f"{url}/reprocess")
    assert repeated.status_code == 200, repeated.text
    modes = repeated.json()["enhancement_applied"]["correction"]["crop_modes"]
    assert all(crop["status"] == "review_needed" and POSSIBLE_PRINT_REGIONS_REVIEW in crop["reason_codes"]
               for crop in modes.values())
    assert (await client.patch(f"{url}/bounding-box", json=update)).status_code == 409
    confirmed = await client.patch(f"{url}/bounding-box", json={**update, "confirm_print_outline": True})
    assert confirmed.status_code == 200, confirmed.text
    assert POSSIBLE_PRINT_REGIONS not in confirmed.json()["segmentation_mask"]
    assert (await client.get(f"{url}/image?variant=source_corrected_untrimmed")).status_code == 200
    for variant in ("full_print", "image_only", "source_corrected_trimmed"):
        assert (await client.get(f"{url}/image", params={"variant": variant})).status_code == 404


@pytest.mark.parametrize("same_paper", [True, False])
async def test_merged_print_owner_edit_requires_a_current_paper_decision(client, same_paper):
    from app.database import get_session_factory
    from app.models import ExtractedPhoto, Page
    from app.pipeline.merged_print_review import MERGED_PRINT_PAIR

    _, page_id, owner_id = await _seed_review_data()

    def rect(x1, y1, x2, y2):
        return [{"x": x, "y": y} for x, y in ((x1, y1), (x2, y1), (x2, y2), (x1, y2))]

    def payload(points):
        box = {f"{axis}{i}": fn(point[axis] for point in points)
               for axis in "xy" for i, fn in ((1, min), (2, max))}
        return {"bounding_box": box, "segmentation_mask": {"polygon": points, "warp_quad": points}}

    whole = rect(.14, .17, .79, .86)
    first, second = rect(.19, .23, .73, .48), rect(.17, .53, .72, .81)
    pair = {"outlines": [first, second], "separator": [{"x": .14, "y": .51}, {"x": .79, "y": .51}]}
    revision = uuid4().hex
    async with get_session_factory()() as session:
        owner = await session.get(ExtractedPhoto, owner_id)
        owner.segmentation_mask = {"source": "yolo_seg_mask", "polygon": whole, MERGED_PRINT_PAIR: pair}
        page = await session.get(Page, page_id)
        page.processing_metadata = {"photo_count_revision": revision}
        await session.commit()
    url = f"/api/v1/photos/{owner_id}/bounding-box"
    broad = payload(whole)
    broad["segmentation_mask"]["print_size_mm"] = {"width": 123, "height": 100}
    clipped = {**payload(rect(.14, .24, .79, .86)), "confirm_merged_print_as_one": True}
    clipped_child = payload(rect(.20, .23, .73, .48))
    clipped_child["segmentation_mask"]["print_size_mm"] = {"width": 133, "height": 100}
    for request in (broad, clipped, clipped_child, {**payload(first), "confirm_merged_print_as_one": True}):
        refused = await client.patch(url, json=request)
        assert (refused.status_code, refused.json()["detail"]) == (409, "merged_print_pair_resolution_required")
    unrelated = await client.post(f"/api/v1/pages/{page_id}/manual-photos", json={
        "request_id": str(uuid4()), "warp_quad": rect(.83, .20, .97, .37),
    })
    assert unrelated.status_code == 201, unrelated.text
    reassign = await client.patch(f"/api/v1/photos/{unrelated.json()['id']}/bounding-box", json=payload(first))
    assert (reassign.status_code, reassign.json()["detail"]) == (409, "merged_print_pair_claim_conflict")
    async with get_session_factory()() as session:
        page = await session.get(Page, page_id)
        owner = await session.get(ExtractedPhoto, owner_id)
        assert owner.segmentation_mask[MERGED_PRINT_PAIR] == pair
        before = page.processing_metadata["photo_count_revision"]
    decided = await client.patch(url, json={**payload(whole if same_paper else first), "confirm_merged_print_as_one": same_paper})
    assert decided.status_code == 200, decided.text
    result = decided.json()
    assert result["id"] == str(owner_id) and MERGED_PRINT_PAIR not in result["segmentation_mask"]
    rev = result["enhancement_applied"]["review_provenance"]["correction_revisions"][-1]
    assert rev["merged_print_resolution"] == ("one_shared_print" if same_paper else "one_child")
    async with get_session_factory()() as session:
        page = await session.get(Page, page_id)
        assert page.processing_metadata["photo_count_revision"] != before


@pytest.mark.parametrize("comparison", ["displaced_edges", "album_layer", "local_edge"])
async def test_source_claim_retires_measured_comparison(client, monkeypatch, tmp_path, comparison):
    from app.config import get_settings
    from app.database import get_session_factory
    from app.models import ExtractedPhoto, Page
    from app.pipeline.bowed_edge_review import (
        POSSIBLE_ALBUM_LAYER_EDGE, POSSIBLE_DISPLACED_PRINT_EDGES, POSSIBLE_LOCAL_PRINT_EDGE,
    )

    monkeypatch.setenv("ENABLE_PRECISE_PHOTO_CROPS", "true")
    monkeypatch.setenv("ENABLE_CLASSICAL_SEGMENTATION_FALLBACK", "true")
    monkeypatch.setenv("ENABLE_PHOTO_ORIENTATION_CORRECTION", "false")
    monkeypatch.setenv("ENABLE_PROTECTED_REGION_GUARD", "false")
    get_settings.cache_clear()
    original = Path(__file__).resolve().parents[3] / "data" / "golden_fixtures" / "page_04.jpg"
    source = tmp_path / "golden04.jpg"
    if comparison == "album_layer":
        image = cv2.rotate(cv2.imread(str(original)), cv2.ROTATE_180)
        assert cv2.imwrite(str(source), image, [cv2.IMWRITE_JPEG_QUALITY, 55])
    elif comparison == "local_edge":
        image = cv2.rotate(cv2.imread(str(original)), cv2.ROTATE_90_CLOCKWISE)
        assert cv2.imwrite(str(source), image, [cv2.IMWRITE_JPEG_QUALITY, 75])
    else:
        source.write_bytes(original.read_bytes())
    height, width = cv2.imread(str(source)).shape[:2]
    _, page_id, photo_id = await _seed_review_data()
    seed = [(0.0324345455, 0.3887441456), (0.0578350909, 0.9775924683),
            (0.3348964453, 0.9718603492), (0.3372411132, 0.3856174946)]
    bounds = {"x1": .0324, "y1": .3854, "x2": .3375, "y2": .9776}
    claim = [(603, 1847), (673, 1847), (673, 1869), (603, 1869)]
    expected_source = POSSIBLE_DISPLACED_PRINT_EDGES
    if comparison == "album_layer":
        seed = [(x / (width - 1), y / (height - 1)) for x, y in (
            (1685, 33), (1702, 512), (1685, 1187), (1710, 1187),
            (1707, 1153), (1810, 1181), (2483, 1179), (2421, 33),
        )]
        bounds = {"x1": .6582, "y1": .0172, "x2": .9703, "y2": .6188}
        claim = [(1550, 23), (1602, 23), (1602, 53), (1550, 53)]
        expected_source = POSSIBLE_ALBUM_LAYER_EDGE
    elif comparison == "local_edge":
        seed = [(x / (width - 1), y / (height - 1)) for x, y in (
            (1174, 89), (50, 139), (50, 853), (964, 870), (895, 850), (1161, 841),
        )]
        bounds = {"x1": .02604, "y1": .03476, "x2": .61198, "y2": .34024}
        claim = [(120, 116), (240, 116), (240, 133), (120, 133)]
        expected_source = POSSIBLE_LOCAL_PRINT_EDGE
    async with get_session_factory()() as session:
        page = await session.get(Page, page_id)
        page.storage_path = str(source)
        page.file_hash_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
        photo = await session.get(ExtractedPhoto, photo_id)
        photo.bounding_box = bounds
        photo.segmentation_mask = {"source": "yolo_seg_mask", "detector_source": "yolo_onnx",
                                   "polygon": [{"x": x, "y": y} for x, y in seed]}
        await session.commit()

    url = f"/api/v1/photos/{photo_id}"
    rendered = await client.post(f"{url}/reprocess")
    assert rendered.status_code == 200, rendered.text
    photo = (await client.get(url)).json()
    metadata = photo["enhancement_applied"]
    full = metadata["correction"]["crop_modes"]["full_print"]
    assert full["outline_preview"]["source"] == expected_source
    assert not (metadata.get("review_provenance") or {}).get("paper_curve_review")
    preview = photo["urls"]["full_print_preview"]
    assert (await client.get(preview)).status_code == 200

    # The claim crosses a measured band outside the saved print.
    added = await client.post(f"/api/v1/pages/{page_id}/manual-photos", json={
        "request_id": str(uuid4()), "warp_quad": [{"x": x / (width - 1), "y": y / (height - 1)} for x, y in claim],
    })
    assert added.status_code == 201, added.text
    assert (await client.get(preview)).status_code == 404
    replay = await client.post(f"{url}/reprocess")
    assert replay.status_code == 200, replay.text
    modes = replay.json()["enhancement_applied"]["correction"]["crop_modes"]
    assert all(mode["status"] == "review_needed" for mode in modes.values())
    assert all(key not in modes["full_print"] for key in ("outline_preview", "suggested_edge_source", "suggested_edge_traces"))
    for variant in ("full_print", "image_only", "source_corrected_trimmed"):
        assert (await client.get(f"{url}/image?variant={variant}")).status_code == 404


async def test_legacy_photo_serves_trim_and_keeps_untrimmed_variant(client, monkeypatch) -> None:
    from app.config import get_settings

    monkeypatch.setenv("ENABLE_PRECISE_PHOTO_CROPS", "false")
    get_settings.cache_clear()
    image = np.full((120, 160, 3), 240, dtype=np.uint8)
    image[9:-9, 10:-10] = (50, 85, 150)
    _, _, photo_id = await _seed_review_data(image)
    photo_url = f"/api/v1/photos/{photo_id}"
    correction = await client.patch(
        f"{photo_url}/bounding-box",
        json={"bounding_box": {"x1": 0.0, "y1": 0.0, "x2": 1.0, "y2": 1.0}},
    )
    assert correction.status_code == 200, correction.text
    metadata = correction.json()["enhancement_applied"]
    assert metadata["correction"]["border_trim"]["applied"]
    assert metadata["variants"]["source_corrected_trimmed"]["path"] == correction.json()["original_storage_path"]

    image_url = correction.json()["image_url"]
    displayed = await client.get(image_url)
    archived = await client.get(f"{photo_url}/image?variant=source_corrected_untrimmed")
    assert displayed.status_code == archived.status_code == 200
    crop = cv2.imdecode(np.frombuffer(displayed.content, np.uint8), cv2.IMREAD_COLOR)
    untrimmed = cv2.imdecode(np.frombuffer(archived.content, np.uint8), cv2.IMREAD_COLOR)
    assert crop.shape[0] < untrimmed.shape[0] and crop.shape[1] < untrimmed.shape[1]
    source_shape = metadata["variants"]["source_corrected"]["metrics"]["shape"]
    assert crop.shape[:2] == (source_shape["height"], source_shape["width"])

    monkeypatch.setenv("ENABLE_BORDER_TRIM", "false")
    get_settings.cache_clear()
    retry = await client.post(f"{photo_url}/reprocess")
    assert retry.status_code == 200, retry.text
    assert "source_corrected_trimmed" not in retry.json()["enhancement_applied"]["variants"]
    new_url = retry.json()["image_url"]
    assert new_url != image_url
    variants = (await client.get(f"{photo_url}/variants")).json()["urls"]
    assert new_url.split("?")[1] in variants["original"]
    displayed = await client.get(new_url)
    assert cv2.imdecode(np.frombuffer(displayed.content, np.uint8), cv2.IMREAD_COLOR).shape == untrimmed.shape


async def test_resaved_model_corners_stay_within_the_reviewers_selection(client, monkeypatch) -> None:
    from app.config import get_settings
    from app.pipeline import perspective

    monkeypatch.setenv("ENABLE_PRECISE_PHOTO_CROPS", "false")
    get_settings.cache_clear()
    _, _, photo_id = await _seed_review_data()
    url = f"/api/v1/photos/{photo_id}"
    reference_boxes = []
    find_border = perspective.find_physical_border

    def check_reference(analysis, *, reference_box, protected_regions):
        reference_boxes.append((reference_box, analysis.shape[:2]))
        return find_border(analysis, reference_box=reference_box, protected_regions=protected_regions)

    monkeypatch.setattr(perspective, "find_physical_border", check_reference)
    polygon = [{"x": x, "y": y} for x, y in [(0.1, 0.2), (0.8, 0.2), (0.8, 0.9), (0.1, 0.9)]]
    saved = await client.patch(
        f"{url}/bounding-box",
        json={
            "bounding_box": {"x1": 0.1, "y1": 0.2, "x2": 0.8, "y2": 0.9},
            "segmentation_mask": {"source": "yolo_seg_mask", "polygon": polygon},
        },
    )
    assert saved.status_code == 200, saved.text
    assert (await client.post(f"{url}/reprocess")).status_code == 200
    assert len(reference_boxes) == 2
    assert all(box == (0, 0, w, h) for box, (h, w) in reference_boxes)


async def test_mobile_queues_expose_overlays_and_blind_pair(client) -> None:
    album_id, page_id, photo_id = await _seed_review_data()
    segmentation = (await client.get(f"/api/v1/review/queue?album_id={album_id}&type=segmentation")).json()
    geometry = (await client.get(f"/api/v1/review/queue?album_id={album_id}&type=geometry")).json()
    enhancement = (await client.get(f"/api/v1/review/queue?album_id={album_id}&type=enhancement")).json()
    assert segmentation[0]["id"] == str(page_id)
    assert segmentation[0]["payload"]["detection_count"] == 1
    assert geometry[0]["payload"]["photo_id"] == str(photo_id)
    assert len(geometry[0]["payload"]["visible_polygon"]) == 6
    assert len(geometry[0]["payload"]["warp_quad"]) == 4
    assert geometry[0]["payload"]["confidence_components"]["perspective_confidence"] == 0.84
    assert geometry[0]["payload"]["variants"]["enhanced"]["metrics"]["source_similarity"] == 0.91
    assert enhancement[0]["payload"]["candidate_name"] == "enhanced"
    assert "candidate_url" in enhancement[0]["payload"]


async def test_enhancement_review_persists_semantic_variant_selection(client) -> None:
    _, _, photo_id = await _seed_review_data()
    response = await client.post(
        "/api/v1/review/events",
        json={
            "task_type": "enhancement",
            "task_id": str(photo_id),
            "answer": "left",
            "idempotency_key": uuid4().hex,
            "metadata": {"blind_choice": "candidate", "candidate_name": "enhanced"},
        },
    )
    assert response.status_code == 201

    from app.database import get_session_factory
    from app.models import ExtractedPhoto
    from app.pipeline.variants import released_variant_from_photo

    async with get_session_factory()() as session:
        photo = await session.get(ExtractedPhoto, photo_id)
        decision = photo.enhancement_applied["review_decisions"]["enhancement"][-1]
        assert decision["blind_choice"] == "candidate"
        assert photo.enhancement_applied["selected_variant"] == "enhanced"
        assert released_variant_from_photo(photo) == "enhanced"


async def test_enhancement_review_rejects_invalid_semantic_metadata(client) -> None:
    _, _, photo_id = await _seed_review_data()
    missing_choice = await client.post(
        "/api/v1/review/events",
        json={
            "task_type": "enhancement",
            "task_id": str(photo_id),
            "answer": "left",
            "idempotency_key": uuid4().hex,
            "metadata": {"candidate_name": "enhanced"},
        },
    )
    source_as_candidate = await client.post(
        "/api/v1/review/events",
        json={
            "task_type": "enhancement",
            "task_id": str(photo_id),
            "answer": "right",
            "idempotency_key": uuid4().hex,
            "metadata": {"blind_choice": "candidate", "candidate_name": "original"},
        },
    )
    assert missing_choice.status_code == 422
    assert source_as_candidate.status_code == 422


async def test_enhancement_idempotency_key_covers_semantic_metadata(client) -> None:
    _, _, photo_id = await _seed_review_data()
    key = uuid4().hex
    payload = {
        "task_type": "enhancement",
        "task_id": str(photo_id),
        "answer": "left",
        "idempotency_key": key,
        "metadata": {"blind_choice": "candidate", "candidate_name": "enhanced"},
    }
    accepted = await client.post("/api/v1/review/events", json=payload)
    changed_replay = await client.post(
        "/api/v1/review/events",
        json=payload | {"metadata": {"blind_choice": "original", "candidate_name": "enhanced"}},
    )
    assert accepted.status_code == 201
    assert changed_replay.status_code == 409


async def test_mobile_review_event_is_idempotent_and_removes_task(client) -> None:
    album_id, page_id, _ = await _seed_review_data()
    key = uuid4().hex
    payload = {
        "task_type": "segmentation",
        "task_id": str(page_id),
        "answer": "correct",
        "idempotency_key": key,
        "metadata": {"ui": "test"},
    }
    first = await client.post("/api/v1/review/events", json=payload)
    second = await client.post("/api/v1/review/events", json=payload)
    assert first.status_code == 201
    assert second.status_code == 201
    assert second.json()["duplicate"] is True
    remaining = (await client.get(f"/api/v1/review/queue?album_id={album_id}&type=segmentation")).json()
    assert all(item["id"] != str(page_id) for item in remaining)


@pytest.mark.parametrize("reason", ["IMAGE_BOUNDARY_CURVED", "WIDE_MOTIF_CONFIRMATION_REQUIRED"])
async def test_geometry_review_reopens_a_motif_missed_by_an_old_approval(client, reason) -> None:
    from app.database import get_session_factory
    from app.models import ExtractedPhoto, PhotoStatus, ReviewEvent

    album_id, _, photo_id = await _seed_review_data()
    payload = {
        "task_type": "geometry",
        "task_id": str(photo_id),
        "answer": "correct",
        "idempotency_key": uuid4().hex,
        "metadata": {"ui": "mobile-v2"},
    }
    async with get_session_factory()() as session:
        photo = await session.get(ExtractedPhoto, photo_id)
        photo.segmentation_confidence = None
        photo.status = PhotoStatus.completed
        correction = {"crop_modes": {
            "full_print": {"status": "ready"},
            "image_only": {"status": "review_needed", "reason_codes": [reason]},
        }}
        photo.enhancement_applied = {
            "variants": {"full_print": {"path": photo.original_storage_path}},
            "correction": correction,
        }
        session.add(ReviewEvent(
            task_type="geometry", task_id=photo_id, answer="correct", idempotency_key=payload["idempotency_key"],
            event_metadata={"client": payload["metadata"], "snapshot": {"photos": [{
                "photo_id": str(photo_id),
                "current_geometry": {"bounding_box": photo.bounding_box, "segmentation_mask": photo.segmentation_mask},
                "current_correction": correction,
            }]}},
        ))
        await session.commit()

    url = f"/api/v1/photos/{photo_id}"
    queue = {"album_id": str(album_id), "type": "geometry", "include_rejected": False}
    event_url = "/api/v1/review/events"
    queue_url = "/api/v1/review/queue"
    pending, = (await client.get(queue_url, params=queue)).json()
    assert pending["id"] == str(photo_id) and pending["status"] == "review_needed"
    assert pending["payload"]["motif_review_pending"] is True
    assert (await client.get(url)).json()["motif_review_pending"] is True
    assert (await client.get(f"{url}/image?variant=full_print")).status_code == 200
    for mode in ("image_only", "source_corrected_trimmed"):
        assert (await client.get(f"{url}/image?variant={mode}")).status_code == 404
    blocked = await client.post(event_url, json=payload | {"idempotency_key": uuid4().hex, "metadata": {
        **payload["metadata"], "geometry_review_revision": pending["payload"]["geometry_review_revision"],
    }})
    assert blocked.status_code == 409 and blocked.json()["detail"] == "motif_geometry_review_required"
    repeated = await client.post(event_url, json=payload)
    assert repeated.status_code == 201 and repeated.json()["duplicate"] is True
    assert (await client.get("/api/v1/review/progress")).json()["by_type"] == {"geometry": 1}

    # A newer raster from the same mask does not make the historic approval current.
    async with get_session_factory()() as session:
        photo = await session.get(ExtractedPhoto, photo_id)
        metadata = photo.enhancement_applied
        photo.enhancement_applied = {
            "variants": {**metadata["variants"], "image_only": {"path": photo.original_storage_path}},
            "correction": {"crop_modes": {
                **metadata["correction"]["crop_modes"], "image_only": {"status": "ready"},
            }},
        }
        await session.commit()
    newer, = (await client.get(queue_url, params=queue)).json()
    assert newer["payload"]["motif_review_pending"] is False
    assert (await client.get(f"{url}/image?variant=image_only")).status_code == 200
    approved = await client.post(event_url, json=payload | {"idempotency_key": uuid4().hex, "metadata": {
        **payload["metadata"], "geometry_review_revision": newer["payload"]["geometry_review_revision"],
    }})
    assert approved.status_code == 201, approved.text
    assert (await client.get(queue_url, params=queue)).json() == []


async def test_disabled_motif_classification_does_not_keep_geometry_review_open(submit_geometry_review, client) -> None:
    from app.database import get_session_factory
    from app.models import ExtractedPhoto

    album_id, _, photo_id = await _seed_review_data()
    async with get_session_factory()() as session:
        photo = await session.get(ExtractedPhoto, photo_id)
        photo.segmentation_confidence = None
        photo.enhancement_applied = {
            "variants": {"full_print": {"path": photo.original_storage_path}},
            "correction": {"crop_modes": {
                "full_print": {"status": "ready"},
                "image_only": {"status": "review_needed", "reason_codes": ["IMAGE_BORDER_CLASSIFICATION_DISABLED"]},
            }},
        }
        await session.commit()
    accepted = await submit_geometry_review("/api/v1/review/events", json={
        "task_type": "geometry", "task_id": str(photo_id), "answer": "correct", "idempotency_key": uuid4().hex,
    })
    assert accepted.status_code == 201, accepted.text
    assert (await client.get("/api/v1/review/queue", params={"album_id": str(album_id), "type": "geometry"})).json() == []


async def test_mobile_review_rejects_answer_from_wrong_task_contract(client) -> None:
    _, page_id, _ = await _seed_review_data()
    response = await client.post(
        "/api/v1/review/events",
        json={
            "task_type": "segmentation",
            "task_id": str(page_id),
            "answer": "left",
            "idempotency_key": uuid4().hex,
        },
    )
    assert response.status_code == 422


async def test_unsure_geometry_stays_queued_and_event_snapshots_prediction(submit_geometry_review, client) -> None:
    album_id, _, photo_id = await _seed_review_data()
    response = await submit_geometry_review(
        "/api/v1/review/events",
        json={
            "task_type": "geometry",
            "task_id": str(photo_id),
            "answer": "unsure",
            "reason_code": "ambiguous_border",
            "idempotency_key": uuid4().hex,
            "metadata": {"ui": "test", "difficulty_tags": ["low_contrast"]},
        },
    )
    assert response.status_code == 201
    queued = (await client.get(f"/api/v1/review/queue?album_id={album_id}&type=geometry")).json()
    assert any(item["id"] == str(photo_id) for item in queued)

    from app.database import get_session_factory
    from app.models import ExtractedPhoto, ReviewEvent
    from sqlalchemy import select

    async with get_session_factory()() as session:
        event = await session.scalar(select(ReviewEvent).where(ReviewEvent.id == response.json()["id"]))
        photo = await session.get(ExtractedPhoto, photo_id)
        snapshot = event.event_metadata["snapshot"]
        assert snapshot["page"]["sha256"]
        assert snapshot["photos"][0]["original_prediction"]["bounding_box"] == photo.bounding_box
        assert snapshot["photos"][0]["current_correction"]["perspective_confidence"] == 0.84
        assert event.event_metadata["client"]["difficulty_tags"] == ["low_contrast"]
        assert photo.status.value == "review_needed"
        assert "GEOMETRY_REVIEW_UNSURE" in photo.enhancement_applied["review_reasons"]


async def test_review_event_validates_entity_type_and_idempotency_payload(submit_geometry_review, client) -> None:
    _, page_id, photo_id = await _seed_review_data()
    wrong_entity = await client.post(
        "/api/v1/review/events",
        json={
            "task_type": "geometry",
            "task_id": str(page_id),
            "answer": "correct",
            "idempotency_key": uuid4().hex,
        },
    )
    assert wrong_entity.status_code == 404

    key = uuid4().hex
    accepted = await submit_geometry_review(
        "/api/v1/review/events",
        json={
            "task_type": "geometry",
            "task_id": str(photo_id),
            "answer": "correct",
            "idempotency_key": key,
        },
    )
    conflict = await submit_geometry_review(
        "/api/v1/review/events",
        json={
            "task_type": "geometry",
            "task_id": str(photo_id),
            "answer": "unsure",
            "idempotency_key": key,
        },
    )
    assert accepted.status_code == 201
    assert conflict.status_code == 409


async def test_bounding_correction_preserves_dense_boundary_and_audit_provenance(client, monkeypatch) -> None:
    _, _, photo_id = await _seed_review_data()

    async def correction_stub(session, target_photo_id, settings, **kwargs):
        return {"review_required": True}

    monkeypatch.setattr("app.routers.photos.perspective_correction_task_logic", correction_stub)
    visible = [
        {"x": 0.12, "y": 0.22},
        {"x": 0.42, "y": 0.19},
        {"x": 0.76, "y": 0.23},
        {"x": 0.79, "y": 0.58},
        {"x": 0.75, "y": 0.86},
        {"x": 0.13, "y": 0.88},
    ]
    warp = [visible[0], visible[2], visible[4], visible[5]]
    response = await client.patch(
        f"/api/v1/photos/{photo_id}/bounding-box",
        json={
            "bounding_box": {"x1": 0.12, "y1": 0.19, "x2": 0.79, "y2": 0.88},
            "segmentation_mask": {
                "polygon": warp,
                "visible_polygon": visible,
                "warp_quad": warp,
                "source": "manual_quad",
            },
        },
    )
    assert response.status_code == 200
    assert response.json()["segmentation_mask"]["visible_polygon"] == visible

    from app.database import get_session_factory
    from app.models import AuditLog, ExtractedPhoto
    from sqlalchemy import select

    async with get_session_factory()() as session:
        photo = await session.get(ExtractedPhoto, photo_id)
        audit = await session.scalar(
            select(AuditLog).where(AuditLog.entity_id == photo_id, AuditLog.action == "bounding_box_updated")
        )
        provenance = photo.enhancement_applied["review_provenance"]
        assert provenance["original_prediction"]["bounding_box"]["x1"] == 0.1
        assert provenance["correction_revisions"][-1]["after"]["segmentation_mask"]["visible_polygon"] == visible
        assert audit.details["before"]["bounding_box"]["x1"] == 0.1
        assert audit.details["after"]["bounding_box"]["x1"] == 0.12
        assert audit.details["prediction_provenance"]["page_sha256"]


async def test_precise_crop_variants_follow_explicit_corners_and_preserve_last_good_export(client, monkeypatch) -> None:
    from app.config import get_settings

    monkeypatch.setenv("ENABLE_BORDER_TRIM", "false")
    monkeypatch.setenv("ENABLE_PHOTO_ORIENTATION_CORRECTION", "false")
    get_settings.cache_clear()
    image = np.full((120, 160, 3), 30, dtype=np.uint8)
    image[12:108, 15:145] = 240
    image[20:101, 23:138] = (50, 90, 150)
    _, _, photo_id = await _seed_review_data(image)
    url = f"/api/v1/photos/{photo_id}"
    outer = [{"x": x / 159, "y": y / 119} for x, y in [(15, 12), (144, 12), (144, 107), (15, 107)]]
    inner = [{"x": x / 159, "y": y / 119} for x, y in [(23, 20), (137, 20), (137, 100), (23, 100)]]
    body = {
        "bounding_box": {"x1": 0.0, "y1": 0.0, "x2": 1.0, "y2": 1.0},
        "segmentation_mask": {"polygon": inner, "visible_polygon": outer, "warp_quad": outer, "source": "manual_quad"},
    }
    response = await client.patch(f"{url}/bounding-box", json=body)
    assert response.status_code == 200, response.text
    unavailable = (await client.get(url)).json()
    assert unavailable["urls"]["full_print"]
    assert not unavailable["urls"]["image_only"]
    assert (await client.get(f"{url}/image?variant=image_only")).status_code == 404

    body["segmentation_mask"]["image_quad"] = inner
    response = await client.patch(f"{url}/bounding-box", json=body)
    assert response.status_code == 200, response.text
    payload = (await client.get(url)).json()
    for mode, size in (("full_print", (96, 130)), ("image_only", (81, 115))):
        artifact = await client.get(payload["urls"][mode])
        assert artifact.status_code == 200
        crop = cv2.imdecode(np.frombuffer(artifact.content, np.uint8), cv2.IMREAD_COLOR)
        assert crop.shape[:2] == size
        assert payload["enhancement_applied"]["correction"]["crop_modes"][mode]["status"] == "ready"
        if mode == "full_print":
            assert crop[2, 60].min() >= 225
        else:
            np.testing.assert_allclose(crop[4, 60], (50, 90, 150), atol=5)

    body["segmentation_mask"]["image_quad"] = [
        {"x": 0.0, "y": 0.0}, *inner[1:]
    ]
    failure = await client.patch(f"{url}/bounding-box", json=body)
    assert failure.status_code == 422, failure.text
    persisted = (await client.get(url)).json()
    assert persisted["segmentation_mask"]["image_quad"] == inner
    assert persisted["urls"]["full_print"] == payload["urls"]["full_print"]
    assert persisted["urls"]["image_only"] == payload["urls"]["image_only"]


async def test_saved_crop_orientation_survives_geometry_edits_and_reprocess(client, monkeypatch) -> None:
    from app.config import get_settings
    from app.pipeline import perspective

    monkeypatch.setenv("ENABLE_PHOTO_ORIENTATION_CORRECTION", "true")
    get_settings.cache_clear()
    monkeypatch.setattr(perspective, "correct_photo_orientation", lambda image, **_: (
        cv2.rotate(image, cv2.ROTATE_180), {"rotation_degrees": 180, "method": "test_model"},
    ))
    page = np.full((120, 160, 3), 30, dtype=np.uint8)
    page[12:108, 15:145] = 240
    page[20:101, 23:138] = (50, 90, 150)
    page[26:47, 29:53] = (180, 90, 40)
    _, _, photo_id = await _seed_review_data(page)
    url = f"/api/v1/photos/{photo_id}"
    outer = [{"x": x / 159, "y": y / 119} for x, y in [(15, 12), (144, 12), (144, 107), (15, 107)]]
    inner = [{"x": x / 159, "y": y / 119} for x, y in [(23, 20), (137, 20), (137, 100), (23, 100)]]
    box = {"x1": 15 / 160, "y1": 12 / 120, "x2": 145 / 160, "y2": 108 / 120}
    mask = {"polygon": outer, "warp_quad": outer, "image_quad": inner, "source": "manual_quad"}

    async def save(**fields):
        return await client.patch(f"{url}/bounding-box", json={
            "bounding_box": box, "segmentation_mask": {**mask, **fields},
        })

    rotated = await save(orientation_degrees=90)
    geometry_edit = await save(source="manual_mobile_quad")
    reprocessed = await client.post(f"{url}/reprocess")
    for response in (rotated, geometry_edit, reprocessed):
        assert response.status_code == 200, response.text
        photo = response.json()
        assert photo["segmentation_mask"]["orientation_degrees"] == 90
        metadata = photo["enhancement_applied"]["correction"]
        assert metadata["orientation"]["method"] == "manual"
        assert metadata["orientation"]["rotation_degrees"] == 90

    for mode, size in (("full_print", (130, 96)), ("image_only", (115, 81))):
        artifact = await client.get(reprocessed.json()["crop_download_urls"][mode])
        assert artifact.status_code == 200
        crop = cv2.imdecode(np.frombuffer(artifact.content, np.uint8), cv2.IMREAD_COLOR)
        assert crop.shape[:2] == size
        np.testing.assert_allclose(crop[17, -17], (180, 90, 40), atol=8)

    for invalid in (45, True, "90", -90, 360):
        assert (await save(orientation_degrees=invalid)).status_code == 422
    preserved = await save(orientation_degrees=0)
    assert preserved.status_code == 200, preserved.text
    assert preserved.json()["enhancement_applied"]["correction"]["orientation"]["rotation_degrees"] == 0

    automatic = await save(orientation_degrees=None)
    assert automatic.status_code == 200, automatic.text
    assert "orientation_degrees" not in automatic.json()["segmentation_mask"]
    assert automatic.json()["enhancement_applied"]["correction"]["orientation"]["rotation_degrees"] == 180

    monkeypatch.setattr(perspective, "correct_photo_orientation", lambda image, **_: (
        image, {"rotation_degrees": 0, "reason_code": "ORIENTATION_UNCERTAIN"},
    ))
    uncertain = await client.post(f"{url}/reprocess")
    assert uncertain.status_code == 200, uncertain.text
    assert "ORIENTATION_UNCERTAIN" in uncertain.json()["enhancement_applied"]["review_reasons"]
    confirmed = await save(orientation_degrees=270)
    assert confirmed.status_code == 200, confirmed.text
    assert "ORIENTATION_UNCERTAIN" not in confirmed.json()["enhancement_applied"]["review_reasons"]


async def test_saving_coordinates_cannot_release_a_photo_hidden_by_an_album_window(client, monkeypatch) -> None:
    from app.config import get_settings
    from app.database import get_session_factory
    from app.models import ExtractedPhoto
    from app.pipeline.album_windows import ALBUM_WINDOW_OCCLUDED

    monkeypatch.setenv("ENABLE_PRECISE_PHOTO_CROPS", "true")
    monkeypatch.setenv("ENABLE_PHOTO_ORIENTATION_CORRECTION", "false")
    get_settings.cache_clear()
    image = np.full((120, 160, 3), 215, np.uint8)
    image[20:100, 30:125] = (75, 100, 145)
    _, _, photo_id = await _seed_review_data(image)
    async with get_session_factory()() as session:
        photo = await session.get(ExtractedPhoto, photo_id)
        photo.segmentation_mask = {**photo.segmentation_mask, "album_window_occluded": True}
        await session.commit()

    url = f"/api/v1/photos/{photo_id}"
    saved = await client.patch(f"{url}/bounding-box", json={
        "bounding_box": {"x1": 30 / 160, "y1": 20 / 120, "x2": 125 / 160, "y2": 100 / 120},
    })
    reapplied = await client.post(f"{url}/reprocess")

    for response in (saved, reapplied):
        assert response.status_code == 200, response.text
        payload = response.json()
        assert payload["segmentation_mask"]["album_window_occluded"] is True
        modes = payload["enhancement_applied"]["correction"]["crop_modes"]
        for name in ("full_print", "image_only"):
            assert modes[name]["status"] == "review_needed" and ALBUM_WINDOW_OCCLUDED in modes[name]["reason_codes"]
            assert payload["crop_download_urls"][name] is None
    for name in ("full_print", "image_only"):
        assert (await client.get(f"{url}/image?variant={name}")).status_code == 404


@pytest.mark.parametrize("with_motif", [False, True], ids=["partial-extension", "measured-motif"])
async def test_competing_outline_comparison_tracks_current_page_candidates(submit_geometry_review, client, with_motif):
    from app.database import get_session_factory
    from app.models import ExtractedPhoto, PhotoStatus
    from app.pipeline.shared_motif_review import SHARED_MOTIF_REVIEW, motif_review_id

    album_id, page_id, wider_id = await _seed_review_data()

    def outline(left, top, right, bottom):
        return [{"x": x, "y": y} for x, y in ((left, top), (right, top), (right, bottom), (left, bottom))]

    wider = outline(.15, .35, .85, .83)
    narrower = outline(.145, .36, .85, .72)
    partial = outline(.15, .745, .85, .875)
    guide = outline(.19, .40, .80, .68)
    evidence = {"id": motif_review_id(guide, [wider, narrower]), "outline": guide}
    modes = {"correction": {"crop_modes": {mode: {
        "status": "review_needed", "reason_codes": ["PRINT_BOUNDARY_UNSUPPORTED"],
    } for mode in ("full_print", "image_only")}}}

    def mask(polygon):
        return {"source": "yolo_seg_mask", "detector_source": "yolo_onnx", "polygon": polygon,
                **({SHARED_MOTIF_REVIEW: evidence} if with_motif and polygon in (wider, narrower) else {})}

    def box(polygon):
        return {f"{axis}{index}": bound(point[axis] for point in polygon)
                for axis in "xy" for index, bound in ((1, min), (2, max))}

    async with get_session_factory()() as session:
        photo = await session.get(ExtractedPhoto, wider_id)
        photo.segmentation_mask = mask(wider)
        photo.bounding_box = box(wider)
        photo.enhancement_applied = modes
        close, extension = [
            ExtractedPhoto(album_id=album_id, page_id=page_id, segmentation_mask=mask(polygon),
                           bounding_box=box(polygon), segmentation_confidence=.48, geometry_valid=valid,
                           status=PhotoStatus.review_needed, storage_path=photo.storage_path, enhancement_applied=modes)
            for polygon, valid in ((narrower, True), (partial, with_motif))
        ]
        session.add_all([close, extension])
        await session.flush()
        narrower_id, extension_id = close.id, extension.id
        await session.commit()

    page_queue = f"/api/v1/review/queue?album_id={album_id}&type=segmentation"
    geometry_queue = f"/api/v1/review/queue?type=geometry&photo_id={narrower_id}"
    album_photos = f"/api/v1/albums/{album_id}/photos"
    expected = {"wider_photo_id": str(wider_id), "narrower_photo_id": str(narrower_id),
                "wider": wider, "narrower": narrower}
    if with_motif:
        expected["motif_guide"] = guide
    count, = (await client.get(page_queue)).json()
    assert count["payload"]["detection_count"] == 3
    member_ids = {str(wider_id), str(narrower_id)}
    for entry in count["payload"]["detections"]:
        assert entry.get("competing_photo_outline") == (expected if entry["photo_id"] in member_ids else None)
    for entry in (await client.get(album_photos)).json():
        assert entry["competing_photo_outline"] == (expected if entry["id"] in member_ids else None)
        assert "competing_photo_outline" not in entry["segmentation_mask"]
    geometry, = (await client.get(geometry_queue)).json()
    assert geometry["payload"]["competing_photo_outline"] == expected
    assert geometry["payload"]["segmentation_mask"] == mask(narrower)
    for mode in ("full_print", "image_only"):
        assert (await client.get(f"/api/v1/photos/{narrower_id}/image?variant={mode}")).status_code == 404

    if with_motif:
        # An independently claimed overlapping print can appear after the guide has been stored.
        async with get_session_factory()() as session:
            other = await session.get(ExtractedPhoto, extension_id)
            other.segmentation_confidence = None
            other.segmentation_mask = {"source": "manual_quad", "polygon": partial, "warp_quad": guide}
            await session.commit()
        for entry in (await client.get(album_photos)).json():
            assert entry["competing_photo_outline"] is None
        async with get_session_factory()() as session:
            other = await session.get(ExtractedPhoto, extension_id)
            other.segmentation_mask = {"source": "manual_quad", "polygon": partial}
            await session.commit()

    removed = await submit_geometry_review("/api/v1/review/events", json={
        "task_type": "geometry", "task_id": str(wider_id if with_motif else extension_id),
        "answer": "wrong_object", "idempotency_key": uuid4().hex,
    })
    assert removed.status_code == 201, removed.text
    count, = (await client.get(page_queue)).json()
    assert count["payload"]["detection_count"] == 2
    assert all("competing_photo_outline" not in entry for entry in count["payload"]["detections"])
    geometry, = (await client.get(geometry_queue)).json()
    assert "competing_photo_outline" not in geometry["payload"]
    assert all(entry["competing_photo_outline"] is None for entry in (await client.get(album_photos)).json())


async def test_visible_scene_guide_stays_held_and_retires_with_a_new_claim(client, monkeypatch, visible_scene_edge_page):
    from app.config import get_settings
    from app.database import get_session_factory
    from app.models import ExtractedPhoto
    from app.pipeline.visible_scene_edge import VISIBLE_SCENE_EDGE

    monkeypatch.setenv("ENABLE_PRECISE_PHOTO_CROPS", "true")
    monkeypatch.setenv("ENABLE_CLASSICAL_SEGMENTATION_FALLBACK", "true")
    monkeypatch.setenv("ENABLE_PROTECTED_REGION_GUARD", "false")
    monkeypatch.setenv("ENABLE_PHOTO_ORIENTATION_CORRECTION", "false")
    get_settings.cache_clear()
    image, quad = visible_scene_edge_page
    album_id, page_id, photo_id = await _seed_review_data(image)

    def points(vertices):
        return [{"x": float(x) / (image.shape[1] - 1), "y": float(y) / (image.shape[0] - 1)}
                for x, y in vertices]

    proposal = points(quad)
    async with get_session_factory()() as session:
        photo = await session.get(ExtractedPhoto, photo_id)
        photo.segmentation_mask = {"source": "yolo_seg_mask", "detector_source": "yolo_onnx", "polygon": proposal}
        photo.bounding_box = {f"{axis}{index}": bound(point[axis] for point in proposal)
                              for axis in "xy" for index, bound in ((1, min), (2, max))}
        await session.commit()

    url = f"/api/v1/photos/{photo_id}"
    response = await client.post(f"{url}/reprocess")
    assert response.status_code == 200, response.text
    modes = response.json()["enhancement_applied"]["correction"]["crop_modes"]
    full = modes["full_print"]
    assert full["suggested_edge_source"] == VISIBLE_SCENE_EDGE
    assert "suggested_warp_quad" not in full and "outline_preview" not in full
    assert all(mode["status"] == "review_needed" for mode in modes.values())
    queue = await client.get("/api/v1/review/queue", params={"album_id": str(album_id), "type": "geometry"})
    item, = queue.json()
    assert item["payload"]["crop_modes"]["full_print"]["suggested_edge_traces"] == full["suggested_edge_traces"]

    claim = points([(210, 380), (381, 380), (381, 470), (210, 470)])
    added = await client.post(f"/api/v1/pages/{page_id}/manual-photos", json={"request_id": str(uuid4()), "warp_quad": claim})
    assert added.status_code == 201, added.text
    after = (await client.get(url)).json()["enhancement_applied"]["correction"]["crop_modes"]["full_print"]
    assert "suggested_edge_source" not in after and "suggested_edge_traces" not in after
    assert (await client.post(f"{url}/reprocess")).status_code == 200
    final = (await client.get(url)).json()["enhancement_applied"]["correction"]["crop_modes"]["full_print"]
    assert "suggested_edge_traces" not in final
    for variant in ("full_print", "image_only", "source_corrected_trimmed", "full_print_preview", "image_only_preview"):
        assert (await client.get(f"{url}/image", params={"variant": variant})).status_code == 404
    assert (await client.get(f"{url}/image", params={"variant": "source_corrected_untrimmed"})).status_code == 200


async def test_ribbed_outline_comparison_is_held_and_retires_with_an_adjacent_claim(client, monkeypatch, rounded_corner_page):
    from app.config import get_settings
    from app.database import get_session_factory
    from app.models import ExtractedPhoto
    from app.pipeline.ribbed_outline_review import MEASURED_RIBBED_PRINT_OUTLINE

    monkeypatch.setenv("ENABLE_PRECISE_PHOTO_CROPS", "true")
    monkeypatch.setenv("ENABLE_PHOTO_ORIENTATION_CORRECTION", "false")
    get_settings.cache_clear()
    _, paper, _, rectangle, backing = rounded_corner_page
    lab = backing.copy()
    lab[rectangle] = (121, 151, 119)
    lab[411, 75:404, 0] = 95  # A second plausible straight rail in the album.
    raster = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    _, page_id, photo_id = await _seed_review_data(raster)

    def points(native):
        return [{"x": float(x) / 484, "y": float(y) / 484} for x, y in native]

    quad = points(paper + np.float32([[2, 2], [-2, 2], [-2, -2], [2, -2]]))
    box = {f"{axis}{index}": bound(point[axis] for point in quad)
           for axis in "xy" for index, bound in ((1, min), (2, max))}
    async with get_session_factory()() as session:
        photo = await session.get(ExtractedPhoto, photo_id)
        photo.segmentation_mask = {"source": "classical_min_area_rect", "detector_source": "contrast_region", "polygon": quad}
        photo.bounding_box = box
        photo.enhancement_applied = {"correction": {"crop_modes": {
            name: {"status": "review_needed", "warp_quad": quad, "reason_codes": ["PRINT_BOUNDARY_UNSUPPORTED"]}
            for name in ("full_print", "image_only")
        }}}
        await session.commit()

    url = f"/api/v1/photos/{photo_id}"
    saved = await client.patch(f"{url}/bounding-box", json={
        "bounding_box": box, "confirm_print_outline": True,
        "segmentation_mask": {"source": "manual_quad", "polygon": quad, "warp_quad": quad},
    })
    assert saved.status_code == 200, saved.text
    data = saved.json()
    assert data["segmentation_mask"]["warp_quad"] == quad
    modes = data["enhancement_applied"]["correction"]["crop_modes"]
    assert all(mode["status"] == "review_needed" for mode in modes.values())
    full = modes["full_print"]
    assert full["outline_preview"]["source"] == MEASURED_RIBBED_PRINT_OUTLINE
    assert len(full["outline_preview"]["source_outline"]) == 4
    assert len(full["suggested_edge_traces"]) == 2 and "suggested_warp_quad" not in full

    async def crop(name):
        return await client.get(f"{url}/image", params={"variant": name})

    preview = await crop("full_print_preview")
    assert preview.status_code == 200 and preview.headers["content-type"] == "image/png"
    assert (await client.post(f"{url}/reprocess")).status_code == 200
    assert (await crop("full_print_preview")).content == preview.content

    claim = points([(413, 180), (454, 180), (454, 300), (413, 300)])
    added = await client.post(f"/api/v1/pages/{page_id}/manual-photos", json={"request_id": str(uuid4()), "warp_quad": claim})
    assert added.status_code == 201, added.text
    assert (await crop("full_print_preview")).status_code == 404
    details = (await client.get(url)).json()["enhancement_applied"]["correction"]["crop_modes"]["full_print"]
    assert details["suggested_edge_traces"] == full["suggested_edge_traces"]
    assert (await client.post(f"{url}/reprocess")).status_code == 200
    for name in ("full_print_preview", "full_print", "image_only", "source_corrected_trimmed"):
        assert (await crop(name)).status_code == 404
    assert (await crop("source_corrected_untrimmed")).status_code == 200


async def test_measured_motif_comparison_requires_a_distinct_confirmed_paper_outline(client, monkeypatch):
    from app.config import get_settings
    from app.database import get_session_factory
    from app.models import ExtractedPhoto
    from app.pipeline.motif_comparison import MEASURED_MOTIF_COMPARISON, MOTIF_COMPARISON_PAPER_EDIT

    monkeypatch.setenv("ENABLE_PRECISE_PHOTO_CROPS", "true")
    monkeypatch.setenv("ENABLE_PROTECTED_REGION_GUARD", "false")
    monkeypatch.setenv("ENABLE_PHOTO_ORIENTATION_CORRECTION", "false")
    get_settings.cache_clear()
    _, page_id, photo_id = await _seed_review_data()

    def quad(left, top, right, bottom):
        return [{"x": x, "y": y} for x, y in ((left, top), (right, top), (right, bottom), (left, bottom))]

    original = quad(.1, .2, .8, .9)
    motif = quad(.23, .33, .7, .78)
    edited = quad(.18, .26, .74, .86)
    async with get_session_factory()() as session:
        photo = await session.get(ExtractedPhoto, photo_id)
        photo.segmentation_mask = {"source": "yolo_seg_mask", "detector_source": "yolo_onnx", "polygon": original}
        modes = {name: {"status": "review_needed", "reason_codes": ["PRINT_BOUNDARY_AMBIGUOUS"]}
                 for name in ("full_print", "image_only")}
        modes["full_print"].update({"warp_quad": original, MOTIF_COMPARISON_PAPER_EDIT: True,
                                    "motif_paper_edge_traces": [[edited[1], edited[2]]]})
        modes["image_only"].update({
            "suggested_warp_quad": motif, "suggested_seam_source": MEASURED_MOTIF_COMPARISON,
            "suggested_seam_traces": [[motif[1], motif[2]]],
            "outline_preview": {"source": MEASURED_MOTIF_COMPARISON, "source_outline": motif},
        })
        photo.enhancement_applied = {**photo.enhancement_applied, "correction": {"crop_modes": modes}}
        await session.commit()

    url = f"/api/v1/photos/{photo_id}"
    initial = (await client.get(url)).json()
    for corners, confirm, inner, reason in (
        (original, False, None, "print_outline_review_required"),
        (original, True, None, "print_outline_review_required"),
        (motif, True, None, "print_outline_review_required"),
        (edited, True, motif, "print_outline_review_required"),
        (edited, False, None, "contrast_print_outline_review_required"),
    ):
        update = {"source": "manual_quad", "warp_quad": corners, "polygon": corners}
        if inner is not None:
            update["image_quad"] = inner
        rejected = await client.patch(f"{url}/bounding-box", json={"bounding_box": initial["bounding_box"],
                                     "segmentation_mask": update, "confirm_print_outline": confirm})
        assert rejected.status_code == 409 and rejected.json()["detail"] == reason, rejected.text
    assert (await client.get(url)).json() == initial

    added = await client.post(f"/api/v1/pages/{page_id}/manual-photos", json={
        "request_id": str(uuid4()), "warp_quad": quad(.60, .47, .95, .72),
    })
    assert added.status_code == 201, added.text
    current = (await client.get(url)).json()["enhancement_applied"]["correction"]["crop_modes"]
    assert current["full_print"][MOTIF_COMPARISON_PAPER_EDIT] is True
    assert "motif_paper_edge_traces" not in current["full_print"]
    assert not {"outline_preview", "suggested_warp_quad", "suggested_seam_source", "suggested_seam_traces"} & current["image_only"].keys()
    for variant in ("image_only_preview", "full_print", "image_only", "source_corrected_trimmed"):
        assert (await client.get(f"{url}/image", params={"variant": variant})).status_code == 404


async def test_faint_outward_review_retains_confirmation_after_a_neighbor_claim_and_retry(client, monkeypatch):
    from app.config import get_settings
    from app.database import get_session_factory
    from app.models import ExtractedPhoto
    from app.pipeline.faint_paper_review import ADJACENT_FAINT_PAPER_STEPS, FAINT_PAPER_REVIEW_REQUIRES_CONFIRMATION

    monkeypatch.setenv("ENABLE_PRECISE_PHOTO_CROPS", "true")
    monkeypatch.setenv("ENABLE_PROTECTED_REGION_GUARD", "false")
    monkeypatch.setenv("ENABLE_PHOTO_ORIENTATION_CORRECTION", "false")
    get_settings.cache_clear()
    _, page_id, photo_id = await _seed_review_data()
    async with get_session_factory()() as session:
        photo = await session.get(ExtractedPhoto, photo_id)
        quad = photo.segmentation_mask["warp_quad"]
        box = photo.bounding_box
        photo.segmentation_mask = {"source": "yolo_seg_mask", "detector_source": "yolo_onnx", "polygon": quad}
        modes = {name: {"status": "review_needed", "reason_codes": ["PRINT_CORNERS_UNCERTAIN"]}
                 for name in ("full_print", "image_only")}
        modes["full_print"].update({
            "warp_quad": quad, FAINT_PAPER_REVIEW_REQUIRES_CONFIRMATION: True,
            "suggested_edge_source": ADJACENT_FAINT_PAPER_STEPS,
            "suggested_edge_traces": [[{"x": .20, "y": .18}, {"x": .65, "y": .18}],
                                      [{"x": .08, "y": .23}, {"x": .08, "y": .62}]],
        })
        photo.enhancement_applied = {**photo.enhancement_applied, "correction": {"crop_modes": modes}}
        await session.commit()

    url = f"/api/v1/photos/{photo_id}"
    request = {"bounding_box": box, "segmentation_mask": {"source": "manual_quad", "polygon": quad, "warp_quad": quad}}
    unconfirmed = await client.patch(f"{url}/bounding-box", json=request)
    assert (unconfirmed.status_code, unconfirmed.json()["detail"]) == (409, "contrast_print_outline_review_required")

    neighbor = [{"x": x, "y": y} for x, y in ((.36, .01), (.61, .01), (.61, .13), (.36, .13))]
    added = await client.post(f"/api/v1/pages/{page_id}/manual-photos", json={"request_id": str(uuid4()), "warp_quad": neighbor})
    assert added.status_code == 201, added.text
    for reprocess in (False, True):
        if reprocess:
            assert (await client.post(f"{url}/reprocess")).status_code == 200
        full = (await client.get(url)).json()["enhancement_applied"]["correction"]["crop_modes"]["full_print"]
        assert full[FAINT_PAPER_REVIEW_REQUIRES_CONFIRMATION] is True
        assert "suggested_edge_traces" not in full
        assert (await client.patch(f"{url}/bounding-box", json=request)).status_code == 409

    confirmed = await client.patch(f"{url}/bounding-box", json={**request, "confirm_print_outline": True})
    assert confirmed.status_code == 200, confirmed.text
    for name in ("full_print", "image_only", "source_corrected_trimmed"):
        assert (await client.get(f"{url}/image", params={"variant": name})).status_code == 404


@pytest.mark.parametrize(("detector_source", "edge_guide"), [
    ("contrast_region", None), ("contrast_region", "contact"), ("yolo_onnx", "contact"), ("yolo_onnx", "opposing"),
])
async def test_held_contrast_outline_requires_separate_paper_confirmation(submit_geometry_review, client, monkeypatch, detector_source, edge_guide):
    from copy import deepcopy

    from app.config import get_settings
    from app.database import get_session_factory
    from app.models import ExtractedPhoto, PhotoStatus
    from app.pipeline.contact_print_review import CONTACT_PRINT_EDGE_CHECK
    from app.pipeline.manual_paper_boundary import MANUAL_CONTRAST_OUTLINE_CHECK, MANUAL_CONTRAST_OUTLINE_REVISION
    from app.pipeline.print_edges import OPPOSING_OUTER_EDGES_CHECK, PRINT_OUTER_EDGE_CONFLICT_REVIEW

    monkeypatch.setenv("ENABLE_PRECISE_PHOTO_CROPS", "true")
    monkeypatch.setenv("ENABLE_PHOTO_ORIENTATION_CORRECTION", "false")
    get_settings.cache_clear()
    image = np.full((120, 160, 3), 215, np.uint8)
    image[24:108, 16:128] = (75, 100, 145)
    _, _, photo_id = await _seed_review_data(image)
    async with get_session_factory()() as session:
        photo = await session.get(ExtractedPhoto, photo_id)
        box = photo.bounding_box
        quad = photo.segmentation_mask["warp_quad"]
        photo.segmentation_mask = {**photo.segmentation_mask, "detector_source": detector_source, "polygon": quad}
        reasons = {
            ("contrast_region", None): ["PRINT_BOUNDARY_UNSUPPORTED"],
            ("contrast_region", "contact"): ["PRINT_BOUNDARY_AMBIGUOUS"],
            ("yolo_onnx", "contact"): ["PRINT_OUTER_RIM_UNCERTAIN", "PRINT_ASPECT_UNCERTAIN"],
            ("yolo_onnx", "opposing"): [PRINT_OUTER_EDGE_CONFLICT_REVIEW],
        }[detector_source, edge_guide]
        checks = {"contact": {CONTACT_PRINT_EDGE_CHECK: True}, "opposing": {OPPOSING_OUTER_EDGES_CHECK: True}}
        photo.enhancement_applied = {
            **photo.enhancement_applied,
            "correction": {"crop_modes": {
                name: {"status": "review_needed", "warp_quad": quad, "reason_codes": reasons,
                       **(checks.get(edge_guide, {}) if name == "full_print" else {})}
                for name in ("full_print", "image_only")
            }},
        }
        await session.commit()

    url = f"/api/v1/photos/{photo_id}"
    old_visibility = [{"x": p["x"] * .9 + .04, "y": p["y"] * .9 + .04} for p in quad]
    mask = {"source": "manual_quad", "polygon": quad, "visible_polygon": old_visibility, "warp_quad": quad,
            "print_size_mm": {"width": 100, "height": 75}}
    for update in (
        {"segmentation_mask": mask},
        {"segmentation_mask": {**mask, "orientation_degrees": 90}},
        {"confirm_print_outline": True},
        {"confirm_print_outline": True, "segmentation_mask": {**mask, "image_quad": quad}},
    ):
        blocked = await client.patch(f"{url}/bounding-box", json={"bounding_box": box, **update})
        assert blocked.status_code == 409 and blocked.json()["detail"] == "contrast_print_outline_review_required"

    unchanged = (await client.get(url)).json()
    assert unchanged["segmentation_confidence"] == .72
    assert unchanged["segmentation_mask"]["warp_quad"] == quad
    for name in ("full_print", "image_only", "source_corrected_trimmed"):
        assert (await client.get(f"{url}/image", params={"variant": name})).status_code == 404

    saved = await client.patch(f"{url}/bounding-box", json={
        "bounding_box": box, "segmentation_mask": mask, "confirm_print_outline": True,
    })
    assert saved.status_code == 200, saved.text
    details = saved.json()
    assert details["segmentation_confidence"] is None
    np.testing.assert_allclose(
        [[point["x"], point["y"]] for point in details["segmentation_mask"]["visible_polygon"]],
        [[point["x"], point["y"]] for point in quad], atol=1e-7,
    )
    revisions = details["enhancement_applied"]["review_provenance"]["correction_revisions"]
    assert len(revisions) == 1 and revisions[0]["contrast_print_outline_confirmed"] is True
    assert details["enhancement_applied"]["correction"][MANUAL_CONTRAST_OUTLINE_CHECK] == MANUAL_CONTRAST_OUTLINE_REVISION

    # A stored crop generated after confirmation but before native rechecking
    # cannot use legacy ready variants while it waits for reprocessing.
    async with get_session_factory()() as session:
        photo = await session.get(ExtractedPhoto, photo_id)
        metadata = deepcopy(photo.enhancement_applied)
        metadata["correction"].pop(MANUAL_CONTRAST_OUTLINE_CHECK)
        for mode in metadata["correction"]["crop_modes"].values():
            mode.update(status="ready", reason_codes=[])
        metadata["variants"].update({name: {"path": photo.original_storage_path} for name in (
            "full_print", "image_only", "source_corrected_trimmed",
        )})
        photo.status = PhotoStatus.completed
        photo.enhancement_applied = metadata
        await session.commit()
    for name in ("full_print", "image_only", "source_corrected_trimmed"):
        assert (await client.get(f"{url}/image", params={"variant": name})).status_code == 404
    rejected = await submit_geometry_review("/api/v1/review/events", json={
        "task_type": "geometry", "task_id": str(photo_id), "answer": "correct", "idempotency_key": str(uuid4()),
    })
    assert rejected.status_code == 409
    queue, = (await client.get("/api/v1/review/queue", params={"type": "geometry", "photo_id": str(photo_id)})).json()
    assert queue["status"] == "review_needed" and queue["payload"]["variant_urls"]["full_print"] is None
    replay = await client.post(f"{url}/reprocess")
    assert replay.status_code == 200, replay.text
    assert replay.json()["enhancement_applied"]["correction"][MANUAL_CONTRAST_OUTLINE_CHECK] == MANUAL_CONTRAST_OUTLINE_REVISION


async def _seed_adjacent_manual_addition(monkeypatch):
    from app.config import get_settings
    from app.database import get_session_factory
    from app.models import ExtractedPhoto

    monkeypatch.setenv("ENABLE_PHOTO_ORIENTATION_CORRECTION", "false")
    monkeypatch.setenv("ENABLE_PROTECTED_REGION_GUARD", "false")
    monkeypatch.setenv("ENABLE_BORDER_TRIM", "false")
    get_settings.cache_clear()
    image = np.full((170, 290, 3), 35, dtype=np.uint8)
    image[24:135, 18:128] = (61, 113, 154)
    image[27:145, 154:265] = (221, 230, 232)
    image[34:137, 162:257] = (115, 64, 82)
    album, page, anchor_id = await _seed_review_data(image)

    def vertices(left, top, right, bottom):
        return [{"x": x / 289, "y": y / 169} for x, y in [(left, top), (right, top), (right, bottom), (left, bottom)]]

    first = vertices(18, 24, 127, 134)
    suggestion = vertices(155, 27, 264, 144)
    async with get_session_factory()() as session:
        anchor = await session.get(ExtractedPhoto, anchor_id)
        anchor.bounding_box = {"x1": first[0]["x"], "y1": first[0]["y"], "x2": first[2]["x"], "y2": first[2]["y"]}
        anchor.segmentation_mask = {
            "source": "classical_border_quad", "polygon": first, "adjacent_print_suggestion": suggestion,
        }
        await session.commit()
    request = {
        "request_id": str(uuid4()), "warp_quad": vertices(154, 27, 264, 144),
        "adjacent_suggestion": {"photo_id": str(anchor_id), "polygon": suggestion},
    }
    return album, page, anchor_id, request, vertices(162, 34, 256, 136)


async def test_adjacent_save_consumes_only_the_saved_suggestion_and_keeps_exact_replays(client, monkeypatch) -> None:
    from app.routers import photos

    album, page, anchor, body, inner = await _seed_adjacent_manual_addition(monkeypatch)
    create = f"/api/v1/pages/{page}/manual-photos"
    listing = f"/api/v1/albums/{album}/photos"
    anchor_url = f"/api/v1/photos/{anchor}"
    body["print_size_mm"] = {"width": 110, "height": 117}
    outdated = {**body, "adjacent_suggestion": {**body["adjacent_suggestion"], "polygon": body["warp_quad"]}}
    assert (await client.post(create, json=outdated)).status_code == 409

    async def reject_render(*args, **kwargs):
        raise ValueError("simulated invalid crop")

    with monkeypatch.context() as faults:
        faults.setattr(photos, "perspective_correction_task_logic", reject_render)
        failed = await client.post(create, json=body)
        assert failed.status_code == 422, failed.text
    assert (await client.get(anchor_url)).json()["segmentation_mask"]["adjacent_print_suggestion"] == body["adjacent_suggestion"]["polygon"]
    assert len((await client.get(listing)).json()) == 1

    saved = await client.post(create, json=body)
    assert saved.status_code == 201, saved.text
    photo = saved.json()
    assert photo["segmentation_mask"]["print_size_mm"] == body["print_size_mm"]
    assert photo["aspect_ratio"] == pytest.approx(110 / 117)
    assert photo["enhancement_applied"]["correction"]["aspect_ratio_estimation"]["method"] == "known_print_aspect"
    assert "adjacent_print_suggestion" not in (await client.get(anchor_url)).json()["segmentation_mask"]
    for changed in (
        {**body, "request_id": str(uuid4())},
        {**body, "print_size_mm": {"width": 108, "height": 117}},
        {key: value for key, value in body.items() if key != "adjacent_suggestion"},
    ):
        assert (await client.post(create, json=changed)).status_code == 409
    replay = await client.post(create, json=body)
    assert replay.status_code == 200 and replay.json()["id"] == photo["id"]
    assert len((await client.get(listing)).json()) == 2
    assert (await client.get(photo["crop_download_urls"]["full_print"])).status_code == 200

    corrected = await client.patch(f"/api/v1/photos/{photo['id']}/bounding-box", json={
        "bounding_box": photo["bounding_box"],
        "segmentation_mask": {
            "polygon": body["warp_quad"], "warp_quad": body["warp_quad"], "image_quad": inner, "source": "manual_quad",
        },
    })
    assert corrected.status_code == 200, corrected.text
    for mode in ("full_print", "image_only"):
        assert (await client.get(corrected.json()["crop_download_urls"][mode])).status_code == 200


@pytest.mark.parametrize("neighbor_claim", [False, True])
async def test_contested_tile_cannot_be_added_separately_or_saved_over_a_new_claim(client, neighbor_claim) -> None:
    from app.database import get_session_factory
    from app.models import ExtractedPhoto
    from app.pipeline.crossing_prints import CONTESTED_TILE_OUTLINE

    album, page, photo_id = await _seed_review_data()

    def rectangle(x1, y1, x2, y2):
        return [{"x": x, "y": y} for x, y in ((x1, y1), (x2, y1), (x2, y2), (x1, y2))]

    alternative = rectangle(.2, .25, .5, .65)
    async with get_session_factory()() as session:
        photo = await session.get(ExtractedPhoto, photo_id)
        photo.segmentation_mask = {**photo.segmentation_mask, "crossing_print_suggestion": alternative,
                                   "crossing_print_suggestion_source": CONTESTED_TILE_OUTLINE}
        await session.commit()
    photo_url = f"/api/v1/photos/{photo_id}"
    add = f"/api/v1/pages/{page}/manual-photos"
    nearby = rectangle(.205, .25, .505, .65)
    duplicate = await client.post(add, json={"request_id": str(uuid4()), "warp_quad": nearby})
    assert duplicate.status_code == 409, duplicate.text
    assert duplicate.json()["detail"] == "contested_tile_correction_required"
    assert len((await client.get(f"/api/v1/albums/{album}/photos")).json()) == 1

    if neighbor_claim:
        late = await client.post(add, json={"request_id": str(uuid4()), "warp_quad": rectangle(.43, .5, .65, .8)})
        assert late.status_code == 201, late.text
    save = await client.patch(f"{photo_url}/bounding-box", json={
        "bounding_box": {"x1": .2, "y1": .25, "x2": .5, "y2": .65},
        "segmentation_mask": {"source": "manual_quad", "warp_quad": alternative, "polygon": alternative},
    })
    if neighbor_claim:
        assert save.status_code == 409, save.text
        assert save.json()["detail"] == "contested_tile_claim_conflict"
        state = (await client.get(photo_url)).json()["segmentation_mask"]
        assert state["crossing_print_suggestion"] == alternative
    else:
        assert save.status_code == 200, save.text
        assert save.json()["id"] == str(photo_id)
        assert "crossing_print_suggestion" not in save.json()["segmentation_mask"]
        duplicate = await client.post(add, json={"request_id": str(uuid4()), "warp_quad": nearby})
        assert duplicate.status_code == 409, duplicate.text
        assert duplicate.json()["detail"] == "contested_tile_correction_required"
        assert len((await client.get(f"/api/v1/albums/{album}/photos")).json()) == 1


@pytest.mark.parametrize("outcome", [
    "released_for_review", "occupied", "rejected_save", "different_outline", "refined_claim", "guide_claimed",
])
async def test_contested_tile_followup_requires_a_committed_matching_correction(client, monkeypatch, outcome):
    from app.config import get_settings
    from app.database import get_session_factory
    from app.models import ExtractedPhoto, Page
    from app.pipeline.crossing_prints import CONTESTED_TILE_OUTLINE
    from app.pipeline.neighboring_print_review import FOLLOWUP_PRINT_EDGE_GUIDE
    from app.pipeline.tile_print_review import CONTESTED_TILE_FOLLOWUP, TILED_PRINT_PROPOSAL
    from app.routers import photos

    monkeypatch.setenv("ENABLE_PRECISE_PHOTO_CROPS", "true")
    get_settings.cache_clear()
    album, page_id, photo_id = await _seed_review_data()

    def rect(x1, y1, x2, y2):
        return [{"x": x, "y": y} for x, y in ((x1, y1), (x2, y1), (x2, y2), (x1, y2))]

    alternative = rect(.15, .45, .60, .82)
    followup = rect(.14, .09, .59, .42)
    revision = str(uuid4())
    pending = [{"id": 2, "reason": TILED_PRINT_PROPOSAL, "polygon": rect(.16 if outcome == "occupied" else .70, .20, .92, .34)}]
    scope = rect(.75, .23, .85, .30) if outcome == "guide_claimed" else rect(.92, .48, .98, .65)
    guide = {"traces": [scope[:2]], "witnesses": [scope]}
    async with get_session_factory()() as session:
        page = await session.get(Page, page_id)
        page.processing_metadata = {"steps": {"segmentation": {"fusion": {"deferred_print_proposals": pending}}},
                                    "photo_count_revision": revision}
        photo = await session.get(ExtractedPhoto, photo_id)
        photo.segmentation_mask = {"polygon": rect(.12, .07, .67, .84), "source": "classical_min_area_rect",
                                   "crossing_print_suggestion": alternative,
                                   "crossing_print_suggestion_source": CONTESTED_TILE_OUTLINE,
                                   CONTESTED_TILE_FOLLOWUP: followup, FOLLOWUP_PRINT_EDGE_GUIDE: guide}
        await session.commit()

    queue = f"/api/v1/review/queue?album_id={album}&type=segmentation"

    async def payload():
        response = await client.get(queue)
        assert response.status_code == 200, response.text
        (item,) = response.json()
        return item["payload"]

    before = await payload()
    assert before["deferred_print_proposals"] == pending
    assert FOLLOWUP_PRINT_EDGE_GUIDE not in before
    if outcome == "rejected_save":
        async def reject(*args, **kwargs):
            raise ValueError("test_render_rejection")

        monkeypatch.setattr(photos, "perspective_correction_task_logic", reject)
    elif outcome == "refined_claim":
        async def refine(session, photo_id, *args, **kwargs):
            current = await session.get(ExtractedPhoto, photo_id)
            current.segmentation_mask = {**current.segmentation_mask, "visible_polygon": rect(.13, .17, .60, .82)}
            return {"review_required": True}

        monkeypatch.setattr(photos, "perspective_correction_task_logic", refine)
    corners = alternative if outcome != "different_outline" else rect(.17, .57, .57, .81)
    url = f"/api/v1/photos/{photo_id}/bounding-box"
    body = {"bounding_box": {"x1": corners[0]["x"], "y1": corners[0]["y"], "x2": corners[2]["x"], "y2": corners[2]["y"]},
            "segmentation_mask": {"source": "manual_quad", "warp_quad": corners, "polygon": corners}}
    saved = await client.patch(url, json=body)
    assert saved.status_code == (422 if outcome == "rejected_save" else 200), saved.text
    after = await payload()
    assert after["detection_count"] == before["detection_count"] == 1
    if outcome not in {"released_for_review", "guide_claimed"}:
        assert after["photo_count_revision"] == revision
        assert after["deferred_print_proposals"] == pending
        assert FOLLOWUP_PRINT_EDGE_GUIDE not in after
        if outcome == "rejected_save":
            async with get_session_factory()() as session:
                photo = await session.get(ExtractedPhoto, photo_id)
                assert photo.segmentation_mask[CONTESTED_TILE_FOLLOWUP] == followup
        return

    assert after["photo_count_revision"] != revision
    assert after["deferred_print_proposals"] == [
        *pending, {"id": 3, "reason": TILED_PRINT_PROPOSAL, "source": CONTESTED_TILE_FOLLOWUP, "polygon": followup},
    ]
    assert CONTESTED_TILE_FOLLOWUP not in saved.json()["segmentation_mask"]
    if outcome == "guide_claimed":
        assert FOLLOWUP_PRINT_EDGE_GUIDE not in after
    else:
        assert after[FOLLOWUP_PRINT_EDGE_GUIDE] == guide
    retry = await client.patch(url, json=body)
    assert retry.status_code == 200, retry.text
    assert (await payload())["deferred_print_proposals"] == after["deferred_print_proposals"]


async def test_measured_print_extent_holds_both_modes_after_manual_dimensions_and_stale_reads(client, monkeypatch):
    from copy import deepcopy

    from app.config import get_settings
    from app.database import get_session_factory
    from app.models import ExtractedPhoto, PhotoStatus, ReviewEvent
    from app.pipeline.crossing_prints import CONTESTED_TILE_OUTLINE
    from app.pipeline.manual_paper_boundary import (
        MEASURED_PRINT_OUTLINE_CUTOFF, MEASURED_PRINT_OUTLINE_RECHECK, MEASURED_PRINT_OUTLINE_SOURCE,
    )

    monkeypatch.setenv("ENABLE_PRECISE_PHOTO_CROPS", "true")
    monkeypatch.setenv("ENABLE_PHOTO_ORIENTATION_CORRECTION", "false")
    get_settings.cache_clear()
    image = np.full((641, 801, 3), (106, 163, 194), np.uint8)
    image[100:541, 150:651] = (55, 68, 106)
    image[100:541, 597:633] = (238, 246, 235)  # Strong inside line, with more print beyond it.
    album_id, _, photo_id = await _seed_review_data(image)

    def rectangle(left, top, right, bottom):
        return [{"x": x / 800, "y": y / 640}
                for x, y in ((left, top), (right, top), (right, bottom), (left, bottom))]

    guide = rectangle(150, 100, 650, 540)
    wrong = rectangle(150, 100, 597, 540)
    parent = rectangle(150, 76, 653, 580)
    async with get_session_factory()() as session:
        photo = await session.get(ExtractedPhoto, photo_id)
        photo.segmentation_mask = {"polygon": parent, "source": "yolo_seg_mask", "detector_source": "yolo_onnx",
                                   "crossing_print_suggestion_source": CONTESTED_TILE_OUTLINE,
                                   "crossing_print_suggestion": guide}
        await session.commit()
    url = f"/api/v1/photos/{photo_id}"
    names = ("full_print", "image_only", "source_corrected_trimmed")

    async def codes():
        return [(await client.get(f"{url}/image", params={"variant": name})).status_code for name in names]

    async def save(quad, width=100):
        return await client.patch(f"{url}/bounding-box", json={
            "bounding_box": {"x1": quad[0]["x"], "x2": quad[2]["x"], "y1": quad[0]["y"], "y2": quad[2]["y"]},
            "segmentation_mask": {"source": "manual_quad", "warp_quad": quad, "polygon": quad,
                                  "print_size_mm": {"width": width, "height": 88}},
        })

    def held(response):
        assert response.status_code == 200, response.text
        photo = response.json()
        correction = photo["enhancement_applied"]["correction"]
        assert all(mode["status"] == "review_needed" and MEASURED_PRINT_OUTLINE_CUTOFF in mode["reason_codes"]
                   for mode in correction["crop_modes"].values())
        whole = correction["crop_modes"]["full_print"]
        assert whole["suggested_edge_source"] == MEASURED_PRINT_OUTLINE_SOURCE
        assert any(all(point["x"] > .80 for point in trace) for trace in whole["suggested_edge_traces"])
        assert photo["crop_download_urls"] == {"full_print": None, "image_only": None}
        return correction

    held(await save(wrong))
    held(await save(wrong, width=112))
    assert await codes() == [404] * 3
    assert (await client.get(f"{url}/image", params={"variant": "source_corrected_untrimmed"})).status_code == 200

    async with get_session_factory()() as session:
        photo = await session.get(ExtractedPhoto, photo_id)
        metadata = deepcopy(photo.enhancement_applied)
        correction = metadata["correction"]
        for name in names:
            metadata["variants"][name] = {"path": correction["untrimmed_path"]}
        for mode in correction["crop_modes"].values():
            mode.update(status="ready", reason_codes=[])
        correction["manual_paper_boundary"] = {"revision": 2, "requires_review": False}
        photo.enhancement_applied = metadata
        photo.status = PhotoStatus.completed
        session.add(ReviewEvent(task_type="geometry", task_id=photo_id, answer="correct", idempotency_key=uuid4().hex,
                                event_metadata={"snapshot": {
            "photos": [{"photo_id": str(photo_id), "current_geometry": {"bounding_box": photo.bounding_box,
                        "segmentation_mask": photo.segmentation_mask}}],
        }}))
        await session.commit()
    assert await codes() == [404] * 3
    stale = (await client.get(url)).json()
    assert stale["status"] == "review_needed"
    assert all(mode["status"] == "review_needed" and MEASURED_PRINT_OUTLINE_RECHECK in mode["reason_codes"]
               for mode in stale["enhancement_applied"]["correction"]["crop_modes"].values())
    review = await client.get("/api/v1/review/queue", params={"type": "geometry", "photo_id": str(photo_id)})
    entry, = review.json()
    assert entry["status"] == "review_needed"
    assert MEASURED_PRINT_OUTLINE_RECHECK in entry["payload"]["crop_modes"]["full_print"]["reason_codes"]
    count = await client.get("/api/v1/review/queue", params={"type": "segmentation", "album_id": str(album_id)})
    page, = count.json()
    item, = page["payload"]["detections"]
    assert MEASURED_PRINT_OUTLINE_RECHECK in item["crop_modes"]["full_print"]["reason_codes"]
    held(await client.post(f"{url}/reprocess"))
    assert await codes() == [404] * 3

    restored = await save(guide)
    assert restored.status_code == 200, restored.text
    correction = restored.json()["enhancement_applied"]["correction"]
    assert correction["manual_paper_boundary"]["measured_outline_check"]["requires_review"] is False
    assert (await codes())[0] == 200


async def test_crossing_outline_requires_outer_correction_before_a_motif_edit(client, monkeypatch) -> None:
    from app.config import get_settings
    from app.database import get_session_factory
    from app.models import ExtractedPhoto
    from app.pipeline.crossing_prints import REVIEWED_CROSSING_OUTLINE

    album, page, anchor_id, _, inner = await _seed_adjacent_manual_addition(monkeypatch)
    url = f"/api/v1/photos/{anchor_id}"
    async with get_session_factory()() as session:
        anchor = await session.get(ExtractedPhoto, anchor_id)
        original = anchor.segmentation_mask["polygon"]
        outer = anchor.segmentation_mask["adjacent_print_suggestion"]
        anchor.segmentation_mask = {"polygon": original, "crossing_print_suggestion": outer}
        old_box = anchor.bounding_box
        await session.commit()

    subpixel = [{**point, "x": point["x"] + 1 / 289} for point in original]
    for mask in (None, {"polygon": original, "warp_quad": original[1:] + original[:1]},
                 {"polygon": subpixel, "warp_quad": subpixel},
                 {"polygon": outer, "warp_quad": outer, "image_quad": inner}):
        body = {"bounding_box": old_box, "segmentation_mask": mask}
        response = await client.patch(f"{url}/bounding-box", json=body)
        assert response.status_code == 409, response.text
        assert response.json()["detail"] == "crossing_print_outline_required"
    still_open = (await client.get(url)).json()["segmentation_mask"]
    assert still_open == {"polygon": original, "crossing_print_suggestion": outer}

    body = {
        "bounding_box": {"x1": outer[0]["x"], "y1": outer[0]["y"], "x2": outer[2]["x"], "y2": outer[2]["y"]},
        "segmentation_mask": {
            "polygon": outer, "warp_quad": outer, "visible_polygon": original,
            "print_size_mm": {"width": 109, "height": 117}, "source": "manual_quad",
        },
    }
    reviewed = await client.patch(f"{url}/bounding-box", json=body)
    assert reviewed.status_code == 200, reviewed.text
    photo = reviewed.json()
    assert photo["id"] == str(anchor_id)
    assert photo["segmentation_mask"]["visible_polygon"] == outer
    assert "crossing_print_suggestion" not in photo["segmentation_mask"]
    assert len((await client.get(f"/api/v1/albums/{album}/photos")).json()) == 1
    assert (await client.get(photo["crop_download_urls"]["full_print"])).status_code == 200

    body["segmentation_mask"] = {**body["segmentation_mask"], "visible_polygon": outer, "image_quad": inner}
    motif = await client.patch(f"{url}/bounding-box", json=body)
    assert motif.status_code == 200, motif.text
    assert motif.json()["segmentation_mask"][REVIEWED_CROSSING_OUTLINE] is True
    for mode in ("full_print", "image_only"):
        assert (await client.get(motif.json()["crop_download_urls"][mode])).status_code == 200

    # A later ordinary review edit can overlap the corrected print. Both
    # records must return to review, including the existing full and motif JPEG.
    neighbor = await client.post(f"/api/v1/pages/{page}/manual-photos", json={
        "request_id": str(uuid4()), "warp_quad": original, "print_size_mm": {"width": 109, "height": 110},
    })
    assert neighbor.status_code == 201, neighbor.text
    neighbor_id = neighbor.json()["id"]
    async with get_session_factory()() as session:
        record = await session.get(ExtractedPhoto, UUID(neighbor_id))
        record.segmentation_mask = {key: value for key, value in record.segmentation_mask.items() if key != "manual_added"}
        await session.commit()
    crossing = [{"x": x / 289, "y": y / 169} for x, y in [(80, 42), (162, 42), (162, 120), (80, 120)]]
    overlap = await client.patch(f"/api/v1/photos/{neighbor_id}/bounding-box", json={
        "bounding_box": {"x1": crossing[0]["x"], "y1": crossing[0]["y"], "x2": crossing[2]["x"], "y2": crossing[2]["y"]},
        "segmentation_mask": {
            "polygon": crossing, "warp_quad": crossing, "visible_polygon": crossing,
            "print_size_mm": {"width": 82, "height": 78}, "source": "manual_quad",
        },
    })
    assert overlap.status_code == 200, overlap.text
    for photo_url in (url, f"/api/v1/photos/{neighbor_id}"):
        for mode in ("full_print", "image_only"):
            assert (await client.get(f"{photo_url}/image?variant={mode}")).status_code == 404
    monkeypatch.setenv("ENABLE_PRECISE_PHOTO_CROPS", "false")
    get_settings.cache_clear()
    replay = await client.post(f"{url}/reprocess")
    assert replay.status_code == 200, replay.text
    modes = replay.json()["enhancement_applied"]["correction"]["crop_modes"]
    assert all(mode["status"] == "review_needed" for mode in modes.values())
    assert "PRINT_POSSIBLY_OCCLUDED" in modes["full_print"]["reason_codes"]


@pytest.mark.parametrize("use_suggestion", [False, True])
async def test_print_outline_suggestion_needs_explicit_whole_print_review(client, monkeypatch, use_suggestion):
    from app.database import get_session_factory
    from app.models import ExtractedPhoto
    from app.pipeline.print_edges import PRINT_EDGE_CONFIRMATION

    album, _, anchor_id, _, _ = await _seed_adjacent_manual_addition(monkeypatch)

    def outline(left, top, right, bottom):
        return [{"x": x / 289, "y": y / 169} for x, y in [(left, top), (right, top), (right, bottom), (left, bottom)]]

    initial = outline(23, 28, 123, 131)
    suggestion = outline(18, 24, 127, 134)
    async with get_session_factory()() as session:
        anchor = await session.get(ExtractedPhoto, anchor_id)
        anchor.segmentation_mask = {"polygon": initial}
        old_box = anchor.bounding_box
        mode = {"status": "review_needed", "reason_codes": [PRINT_EDGE_CONFIRMATION]}
        metadata = dict(anchor.enhancement_applied)
        metadata["correction"] = {**metadata["correction"], "crop_modes": {
            "full_print": {**mode, "warp_quad": initial, "suggested_warp_quad": suggestion},
            "image_only": mode,
        }}
        metadata["review_reasons"] = [PRINT_EDGE_CONFIRMATION]
        anchor.enhancement_applied = metadata
        await session.commit()

    url = f"/api/v1/photos/{anchor_id}"
    queue = (await client.get("/api/v1/review/queue", params={"album_id": album, "type": "geometry"})).json()
    item, = queue
    mode = item["payload"]["crop_modes"]["full_print"]
    assert mode["warp_quad"] == initial and mode["suggested_warp_quad"] == suggestion
    for mode_name in ("full_print", "image_only"):
        assert (await client.get(f"{url}/image", params={"variant": mode_name})).status_code == 404
    edit = {"polygon": initial, "warp_quad": initial, "image_quad": outline(30, 34, 110, 123)}
    for mask in (None, edit, {**edit, "orientation_degrees": 90}):
        blocked = await client.patch(f"{url}/bounding-box", json={"bounding_box": old_box, "segmentation_mask": mask})
        assert blocked.status_code == 409 and blocked.json()["detail"] == "print_outline_review_required"

    selected = suggestion if use_suggestion else initial
    corrected = await client.patch(f"{url}/bounding-box", json={
        "bounding_box": {"x1": selected[0]["x"], "y1": selected[0]["y"], "x2": selected[2]["x"], "y2": selected[2]["y"]},
        "segmentation_mask": {**edit, "polygon": selected, "warp_quad": selected, "source": "manual_quad",
                              "print_size_mm": {"width": 109, "height": 110}},
        "confirm_print_outline": not use_suggestion,
    })
    assert corrected.status_code == 200, corrected.text
    photo = corrected.json()
    assert photo["id"] == str(anchor_id)
    assert "suggested_warp_quad" not in photo["enhancement_applied"]["correction"]["crop_modes"]["full_print"]
    assert len((await client.get(f"/api/v1/albums/{album}/photos")).json()) == 1
    for mode_name in ("full_print", "image_only"):
        assert (await client.get(photo["crop_download_urls"][mode_name])).status_code == 200


async def test_manual_save_of_matching_outline_retires_the_open_suggestion(client, monkeypatch) -> None:
    from app.database import get_session_factory
    from app.models import ExtractedPhoto

    album, page, anchor, bound, _ = await _seed_adjacent_manual_addition(monkeypatch)
    route = f"/api/v1/pages/{page}/manual-photos"
    outlined = bound["adjacent_suggestion"]["polygon"]
    manual = {"request_id": str(uuid4()), "warp_quad": outlined}

    response = await client.post(route, json=manual)
    assert response.status_code == 201, response.text
    assert "adjacent_print_suggestion" not in (await client.get(f"/api/v1/photos/{anchor}")).json()["segmentation_mask"]
    same_footprint = {**manual, "request_id": str(uuid4()), "warp_quad": outlined[1:] + outlined[:1]}
    assert (await client.post(route, json=same_footprint)).status_code == 409
    assert (await client.post(route, json=bound)).status_code == 409

    # Older sessions could save exact corners without consuming the hint.
    async with get_session_factory()() as session:
        photo = await session.get(ExtractedPhoto, anchor)
        photo.segmentation_mask = {**photo.segmentation_mask, "adjacent_print_suggestion": outlined}
        await session.commit()
    assert (await client.post(route, json={**bound, "warp_quad": outlined})).status_code == 409
    assert "adjacent_print_suggestion" not in (await client.get(f"/api/v1/photos/{anchor}")).json()["segmentation_mask"]
    assert len((await client.get(f"/api/v1/albums/{album}/photos")).json()) == 2


@pytest.mark.parametrize("same_request", [True, False])
async def test_concurrent_adjacent_saves_exhibit_a_single_photo(client, monkeypatch, same_request) -> None:
    from app.routers import photos

    album, page, _, body, _ = await _seed_adjacent_manual_addition(monkeypatch)
    barrier = asyncio.Barrier(2)
    prediction = photos._prediction_provenance

    async def sync_before_insert(session, photo):
        result = await prediction(session, photo)
        await asyncio.wait_for(barrier.wait(), timeout=5)
        return result

    monkeypatch.setattr(photos, "_prediction_provenance", sync_before_insert)
    second = body if same_request else {**body, "request_id": str(uuid4())}
    route = f"/api/v1/pages/{page}/manual-photos"
    responses = await asyncio.gather(client.post(route, json=body), client.post(route, json=second))
    assert sorted(response.status_code for response in responses) == ([200, 201] if same_request else [201, 409])
    assert len((await client.get(f"/api/v1/albums/{album}/photos")).json()) == 2


async def test_add_missed_photo_reopens_count_review_and_supports_both_downloads(submit_geometry_review, client, monkeypatch) -> None:
    from app.config import get_settings
    from app.database import get_session_factory
    from app.models import ExtractedPhoto

    monkeypatch.setenv("ENABLE_BORDER_TRIM", "false")
    monkeypatch.setenv("ENABLE_PHOTO_ORIENTATION_CORRECTION", "false")
    get_settings.cache_clear()
    image = np.full((120, 160, 3), 35, dtype=np.uint8)
    image[12:108, 15:145] = 240
    image[20:101, 23:138] = (50, 90, 150)
    album_id, page_id, placeholder = await _seed_review_data(image)
    async with get_session_factory()() as session:
        await session.delete(await session.get(ExtractedPhoto, placeholder))
        await session.commit()

    page_url = f"/api/v1/pages/{page_id}"
    create_url = f"{page_url}/manual-photos"
    source = (await client.get(f"{page_url}/image")).content
    queue_url = f"/api/v1/review/queue?type=segmentation&album_id={album_id}"
    events_url = "/api/v1/review/events"
    first_decision = {
        "task_type": "segmentation", "task_id": str(page_id), "answer": "correct",
        "idempotency_key": uuid4().hex,
    }
    assert (await client.post(events_url, json=first_decision)).status_code == 201
    assert (await client.get(queue_url)).json() == []
    outer = [{"x": x / 159, "y": y / 119} for x, y in [(15, 12), (144, 12), (144, 107), (15, 107)]]
    inner = [{"x": x / 159, "y": y / 119} for x, y in [(23, 20), (137, 20), (137, 100), (23, 100)]]
    body = {"request_id": str(uuid4()), "warp_quad": outer}
    crossed = {**body, "warp_quad": [outer[index] for index in (0, 2, 1, 3)]}
    assert (await client.post(create_url, json=crossed)).status_code == 422

    result = await client.post(create_url, json=body)
    assert result.status_code == 201, result.text
    photo = result.json()
    photo_url = f"/api/v1/photos/{photo['id']}"
    assert photo["segmentation_confidence"] is None
    assert photo["segmentation_mask"]["source"] == "manual_added"
    assert photo["crop_download_urls"]["full_print"]
    assert not photo["crop_download_urls"]["image_only"]
    replay = await client.post(create_url, json=body)
    assert replay.status_code == 200 and replay.json()["id"] == photo["id"]
    assert (await client.post(create_url, json={**body, "warp_quad": inner})).status_code == 409
    assert (await client.get(f"/api/v1/albums/{album_id}")).json()["photos_extracted"] == 1

    item, = (await client.get(queue_url)).json()
    assert item["payload"]["detection_count"] == 1
    revision = item["payload"]["photo_count_revision"]
    assert revision == photo["id"]
    stale = {**first_decision, "idempotency_key": uuid4().hex}
    assert (await client.post(events_url, json=stale)).status_code == 409
    current = {**stale, "metadata": {"photo_count_revision": revision}}
    assert (await client.post(events_url, json=current)).status_code == 201
    assert (await client.get(queue_url)).json() == []

    corrected = await client.patch(f"{photo_url}/bounding-box", json={
        "bounding_box": photo["bounding_box"],
        "segmentation_mask": {"polygon": outer, "warp_quad": outer, "image_quad": inner, "source": "manual_quad"},
    })
    assert corrected.status_code == 200, corrected.text
    assert corrected.json()["segmentation_mask"]["manual_added"] is True
    reprocessed = await client.post(f"{photo_url}/reprocess")
    assert reprocessed.status_code == 200, reprocessed.text
    for mode, shape in (("full_print", (96, 130)), ("image_only", (81, 115))):
        download = await client.get(reprocessed.json()["crop_download_urls"][mode])
        assert download.status_code == 200
        crop = cv2.imdecode(np.frombuffer(download.content, np.uint8), cv2.IMREAD_COLOR)
        assert crop.shape[:2] == shape
    assert (await client.get(f"{page_url}/image")).content == source
    assert (await client.post(create_url, json=body)).status_code == 200

    rejected = await submit_geometry_review(events_url, json={
        "task_type": "geometry", "task_id": photo["id"], "answer": "wrong_object",
        "idempotency_key": uuid4().hex,
    })
    assert rejected.status_code == 201, rejected.text
    item, = (await client.get(queue_url)).json()
    assert item["payload"]["detection_count"] == 0
    removed_revision = item["payload"]["photo_count_revision"]
    assert removed_revision != revision
    after_retry = await client.post(f"{photo_url}/reprocess")
    assert after_retry.status_code == 200
    for mode in ("full_print", "image_only"):
        assert not after_retry.json()["crop_download_urls"][mode]
    assert (await client.post(events_url, json={
        **stale, "idempotency_key": uuid4().hex, "metadata": {"photo_count_revision": removed_revision},
    })).status_code == 201
    assert (await client.get(queue_url)).json() == []
    restored = await client.patch(f"{photo_url}/bounding-box", json={"bounding_box": photo["bounding_box"],
        "segmentation_mask": {"polygon": outer, "warp_quad": outer, "image_quad": inner},
    })
    assert restored.status_code == 200 and restored.json()["crop_download_urls"]["full_print"]
    item, = (await client.get(queue_url)).json()
    assert item["payload"]["detection_count"] == 1
    assert item["payload"]["photo_count_revision"] not in (revision, removed_revision)


async def test_new_overlapping_print_reopens_existing_downloads_for_review(client, monkeypatch) -> None:
    from app.config import get_settings

    monkeypatch.setenv("ENABLE_PHOTO_ORIENTATION_CORRECTION", "false")
    get_settings.cache_clear()
    page = np.full((190, 260, 3), 35, dtype=np.uint8)
    page[20:150, 25:163] = (67, 111, 144)
    page[75:166, 127:234] = (118, 90, 55)
    _, page_id, old_id = await _seed_review_data(page)
    old_url = f"/api/v1/photos/{old_id}"
    old_quad = [{"x": x / 259, "y": y / 189} for x, y in [(25, 20), (162, 20), (162, 149), (25, 149)]]
    saved = await client.patch(f"{old_url}/bounding-box", json={
        "bounding_box": {"x1": 25 / 259, "y1": 20 / 189, "x2": 162 / 259, "y2": 149 / 189},
        "segmentation_mask": {"polygon": old_quad, "warp_quad": old_quad},
    })
    assert saved.status_code == 200 and saved.json()["crop_download_urls"]["full_print"]
    new_quad = [{"x": x / 259, "y": y / 189} for x, y in [(127, 75), (233, 75), (233, 165), (127, 165)]]
    added = await client.post(f"/api/v1/pages/{page_id}/manual-photos", json={
        "request_id": str(uuid4()), "warp_quad": new_quad,
    })
    assert added.status_code == 201, added.text
    assert not added.json()["crop_download_urls"]["full_print"]
    assert (await client.get(f"{old_url}/image?variant=full_print")).status_code == 404
    reprocessed = await client.post(f"{old_url}/reprocess")
    assert reprocessed.status_code == 200 and not reprocessed.json()["crop_download_urls"]["full_print"]


@pytest.mark.parametrize("render_neighbor", [False, True])
async def test_ordinary_preview_cannot_export_paper_shared_with_an_active_neighbor(submit_geometry_review, client, monkeypatch, render_neighbor):
    from app.config import get_settings
    from app.database import get_session_factory
    from app.models import ExtractedPhoto, PhotoStatus

    monkeypatch.setenv("ENABLE_PRECISE_PHOTO_CROPS", "true")
    monkeypatch.setenv("ENABLE_PHOTO_ORIENTATION_CORRECTION", "false")
    monkeypatch.setenv("ENABLE_PROTECTED_REGION_GUARD", "false")
    get_settings.cache_clear()
    source = np.full((310, 480, 3), 35, dtype=np.uint8)
    source[40:260, 28:242] = (67, 111, 144)
    source[70:280, 245:435] = (118, 90, 55)
    album_id, page_id, photo_id = await _seed_review_data(source)

    def outline(left, top, right, bottom):
        return [{"x": x / 479, "y": y / 309} for x, y in ((left, top), (right, top), (right, bottom), (left, bottom))]

    def bounds(quad):
        return {"x1": quad[0]["x"], "y1": quad[0]["y"], "x2": quad[2]["x"], "y2": quad[2]["y"]}

    async def save(photo, outer, inner):
        return await client.patch(f"/api/v1/photos/{photo}/bounding-box", json={
            "bounding_box": bounds(outer), "segmentation_mask": {
                "source": "manual_quad", "polygon": outer, "visible_polygon": outer, "warp_quad": outer,
                "image_quad": inner, "print_size_mm": {"width": 120, "height": 140},
            },
        })

    async def exports(photo):
        return [(await client.get(f"/api/v1/photos/{photo}/image", params={"variant": mode})).status_code
                for mode in ("full_print", "image_only", "source_corrected_trimmed", "source_corrected_untrimmed")]

    left, left_motif = outline(28, 40, 241, 259), outline(39, 52, 228, 246)
    right, right_motif = outline(245, 70, 434, 279), outline(256, 82, 423, 267)
    async with get_session_factory()() as session:
        base = await session.get(ExtractedPhoto, photo_id)
        other = ExtractedPhoto(
            page_id=page_id, album_id=album_id, storage_path=base.storage_path,
            bounding_box=bounds(right), segmentation_mask={"source": "yolo_seg_mask", "polygon": right},
            segmentation_confidence=.89, geometry_valid=True, status=PhotoStatus.review_needed,
        )
        session.add(other)
        await session.flush()
        other_id = other.id
        await session.commit()
    assert (await save(photo_id, left, left_motif)).status_code == 200
    assert await exports(photo_id) == [200, 200, 200, 200]
    if render_neighbor:
        assert (await save(other_id, right, right_motif)).status_code == 200
        assert await exports(other_id) == [200, 200, 200, 200]

    contender = await save(photo_id, outline(28, 40, 291, 259), left_motif)
    assert contender.status_code == 200, contender.text
    assert await exports(photo_id) == [404, 404, 404, 200]
    modes = contender.json()["enhancement_applied"]["correction"]["crop_modes"]
    assert all(mode["status"] == "review_needed" and "PRINT_POSSIBLY_OCCLUDED" in mode["reason_codes"]
               for mode in modes.values())
    if render_neighbor:
        assert await exports(other_id) == [404, 404, 404, 200]
        narrowed_neighbor = await save(other_id, outline(293, 70, 434, 279), outline(305, 82, 423, 267))
        assert narrowed_neighbor.status_code == 200
        full = narrowed_neighbor.json()["enhancement_applied"]["correction"]["crop_modes"]["full_print"]
        assert "MANUAL_NEIGHBOR_CONTENT_LOSS" in full["reason_codes"]
        assert await exports(other_id) == [404, 404, 404, 200]
    for _ in range(2):
        assert (await client.post(f"/api/v1/photos/{photo_id}/reprocess")).status_code == 200
        assert await exports(photo_id) == [404, 404, 404, 200]
    approval = await submit_geometry_review("/api/v1/review/events", json={
        "task_type": "geometry", "task_id": str(photo_id), "answer": "correct", "idempotency_key": str(uuid4()),
    })
    assert approval.status_code == 409
    too_narrow = await save(photo_id, outline(28, 40, 184, 259), outline(39, 52, 173, 246))
    assert too_narrow.status_code == 200
    for result in (too_narrow, await client.post(f"/api/v1/photos/{photo_id}/reprocess")):
        assert result.status_code == 200
        crop = result.json()["enhancement_applied"]["correction"]["crop_modes"]["full_print"]
        assert "MANUAL_NEIGHBOR_CONTENT_LOSS" in crop["reason_codes"]
        assert crop["neighbor_content_loss_outlines"]
    assert await exports(photo_id) == [404, 404, 404, 200]
    assert (await save(photo_id, left, left_motif)).status_code == 200
    assert await exports(photo_id) == [200, 200, 200, 200]
    assert (await save(other_id, right, right_motif)).status_code == 200
    assert await exports(other_id) == [200, 200, 200, 200]
    if render_neighbor:
        async with get_session_factory()() as session:
            existing = await session.get(ExtractedPhoto, photo_id)
            existing.segmentation_mask = {**existing.segmentation_mask, "warp_quad": outline(28, 40, 291, 259)}
            await session.commit()
        assert (await client.post(f"/api/v1/photos/{photo_id}/reprocess")).status_code == 200
        assert await exports(photo_id) == await exports(other_id) == [404, 404, 404, 200]


@pytest.mark.parametrize("lost", ["visible_print", "album_backing", "marked_strip", "backing_colored_print"])
async def test_avoiding_an_overlap_preserves_visible_content_and_checks_the_original(client, monkeypatch, lost):
    from app.config import get_settings
    from app.database import get_session_factory
    from app.models import ExtractedPhoto

    monkeypatch.setenv("ENABLE_PRECISE_PHOTO_CROPS", "true")
    monkeypatch.setenv("ENABLE_PROTECTED_REGION_GUARD", "false")
    monkeypatch.setenv("ENABLE_PHOTO_ORIENTATION_CORRECTION", "false")
    get_settings.cache_clear()
    source = np.full((320, 480, 3), 30, dtype=np.uint8)
    source[40:270, 35:265] = (67, 111, 144)
    if lost == "backing_colored_print":
        source[40:270, 281:283] = 17  # A faint outer shadow beyond the claimed strip.
        source[72:87, 267:277] = 32  # Low-contrast content on the matching stock.
    peer_left = 205 if lost == "visible_print" else 272
    source[100:229, peer_left:380] = (118, 90, 55)
    if lost == "marked_strip":
        source[56:66, 264:275] = (70, 85, 235)
    album_id, page_id, photo_id = await _seed_review_data(source)

    def outline(left, top, right, bottom):
        return [{"x": x / 479, "y": y / 319} for x, y in ((left, top), (right, top), (right, bottom), (left, bottom))]

    def bounds(quad):
        return {"x1": quad[0]["x"], "y1": quad[0]["y"], "x2": quad[2]["x"], "y2": quad[2]["y"]}

    old = outline(35, 40, 264 if lost == "visible_print" else 297, 269)
    visible = old if lost == "visible_print" else outline(35, 40, 280, 269)
    neighbor = outline(peer_left, 100, 379, 228)
    inward = outline(35, 40, 202 if lost == "visible_print" else 264, 269)
    image = outline(47, 52, 191 if lost == "visible_print" else 251, 256)
    async with get_session_factory()() as session:
        photo = await session.get(ExtractedPhoto, photo_id)
        photo.bounding_box = bounds(old)
        photo.segmentation_mask = {"source": "yolo_seg_mask", "polygon": old, "visible_polygon": visible, "warp_quad": old}
        photo.enhancement_applied = {"correction": {"crop_modes": {"full_print": {
            "warp_quad": old, "status": "review_needed", "reason_codes": ["PRINT_POSSIBLY_OCCLUDED"],
        }}}}
        session.add(ExtractedPhoto(page_id=page_id, album_id=album_id, storage_path=photo.storage_path,
                                  bounding_box=bounds(neighbor), geometry_valid=True, segmentation_confidence=.89,
                                  segmentation_mask={"source": "yolo_seg_mask", "polygon": neighbor}))
        await session.commit()

    url = f"/api/v1/photos/{photo_id}"
    saved = await client.patch(f"{url}/bounding-box", json={"bounding_box": bounds(inward), "segmentation_mask": {
        "source": "manual_quad", "polygon": inward, "warp_quad": inward, "image_quad": image,
        "print_size_mm": {"width": 120, "height": 145},
    }})
    assert saved.status_code == 200, saved.text
    reprocessed = await client.post(f"{url}/reprocess")
    assert reprocessed.status_code == 200, reprocessed.text
    for response in (saved, reprocessed):
        correction = response.json()["enhancement_applied"]["correction"]
        modes = correction["crop_modes"]
        assert all(mode["status"] == "review_needed" for mode in modes.values()), modes
        full = modes["full_print"]
        assert "MANUAL_NEIGHBOR_CONTENT_LOSS" in full["reason_codes"]
        regions = full["neighbor_content_loss_outlines"]
        assert regions and all(region[0] == region[-1] for region in regions)
        assert all(point["x"] > inward[1]["x"] for region in regions for point in region)
        assert full["neighbor_review_source_size"] == {"width": 480, "height": 320}
        assert bool(full.get("neighbor_backing_color_match_pixels")) == (lost in {"album_backing", "backing_colored_print"})
    for variant in ("full_print", "image_only", "source_corrected_trimmed"):
        assert (await client.get(f"{url}/image", params={"variant": variant})).status_code == 404
    assert (await client.get(f"{url}/image?variant=source_corrected_untrimmed")).status_code == 200


@pytest.mark.parametrize("claim", ["paper_grid", "scalloped"])
async def test_measured_crop_stays_held_while_a_linked_proposal_overlaps(submit_geometry_review, client, monkeypatch, claim) -> None:
    from app.config import get_settings
    from app.database import get_session_factory
    from app.models import ExtractedPhoto, PhotoStatus
    from app.pipeline.manual_paper_boundary import PAPER_GRID_OVERLAP_REVIEW
    from app.pipeline.scalloped_prints import SCALLOPED_PRINT_CONFLICT

    monkeypatch.setenv("ENABLE_PHOTO_ORIENTATION_CORRECTION", "false")
    monkeypatch.setenv("ENABLE_PROTECTED_REGION_GUARD", "false")
    get_settings.cache_clear()
    source = np.full((190, 260, 3), 35, dtype=np.uint8)
    source[20:160, 25:166] = (67, 111, 144)
    source[40:132, 190:244] = (82, 150, 203)
    album_id, page_id, parent_id = await _seed_review_data(source)

    def vertices(points):
        return [{"x": x / 259, "y": y / 189} for x, y in points]

    def bounds(points):
        return {f"{axis}{suffix}": limit(point[axis] for point in points)
                for axis in ("x", "y") for suffix, limit in (("1", min), ("2", max))}

    outer = vertices([(25, 20), (165, 20), (165, 159), (25, 159)])
    child_quad = vertices([(37, 84), (153, 84), (153, 147), (37, 147)])
    child_polygon = vertices([(37, 84), (96, 84), (153, 84), (153, 147), (96, 147), (37, 147)])
    if claim == "scalloped":
        parent_group = child_group = {"scalloped_ownership_group": outer}
        reason = SCALLOPED_PRINT_CONFLICT
        overlap_revision = "scalloped_ownership_overlap_revision"
    else:
        parent_group = {"paper_grid_review": {"outline": outer}}
        child_group = {"paper_grid_review": {"outline": outer, "cut": child_quad[:2]}}
        reason = PAPER_GRID_OVERLAP_REVIEW
        overlap_revision = "paper_grid_overlap_revision"
    async with get_session_factory()() as session:
        parent = await session.get(ExtractedPhoto, parent_id)
        parent.segmentation_mask = {"source": "classical_border_quad", "polygon": outer, **parent_group}
        parent.bounding_box = bounds(outer)
        child = ExtractedPhoto(
            page_id=page_id, album_id=album_id, storage_path=parent.storage_path,
            original_storage_path=parent.original_storage_path, bounding_box=bounds(child_polygon),
            segmentation_mask={"source": "yolo_seg_mask", "polygon": child_polygon, **child_group},
            segmentation_confidence=.66, geometry_valid=True, status=PhotoStatus.review_needed,
        )
        session.add(child)
        await session.commit()
        child_id = child.id

    parent_url = f"/api/v1/photos/{parent_id}"
    child_url = f"/api/v1/photos/{child_id}"

    async def save(url, quad, *, motif=None):
        mask = {"polygon": quad, "warp_quad": quad, "print_size_mm": {"width": 140, "height": 139}}
        if motif is not None:
            mask["image_quad"] = motif
        return await client.patch(f"{url}/bounding-box", json={"bounding_box": bounds(quad), "segmentation_mask": mask})

    async def held_downloads():
        for variant in ("full_print", "image_only", "source_corrected_trimmed"):
            assert (await client.get(f"{parent_url}/image", params={"variant": variant})).status_code == 404

    geometry_queue = {"type": "geometry", "album_id": str(album_id), "include_rejected": "false"}
    parent_task = {**geometry_queue, "photo_id": str(parent_id), "limit": 1}
    child_task = {**geometry_queue, "photo_id": str(child_id), "limit": 1}
    first, = (await client.get("/api/v1/review/queue", params={**geometry_queue, "limit": 1})).json()
    assert first["id"] == str(child_id)
    targeted, = (await client.get("/api/v1/review/queue", params=parent_task)).json()
    assert targeted["id"] == str(parent_id)

    result = await save(parent_url, outer, motif=outer)
    assert result.status_code == 200, result.text
    modes = result.json()["enhancement_applied"]["correction"]["crop_modes"]
    assert all(mode["status"] == "review_needed" and reason in mode["reason_codes"] for mode in modes.values())
    targeted, = (await client.get("/api/v1/review/queue", params=parent_task)).json()
    assert targeted["id"] == str(parent_id) and targeted["payload"]["crop_modes"] == modes
    assert (await client.get("/api/v1/review/queue", params={**parent_task, "album_id": str(uuid4())})).json() == []
    await held_downloads()
    count_item, = (await client.get("/api/v1/review/queue", params={"type": "segmentation", "album_id": str(album_id)})).json()
    confirmed = await client.post("/api/v1/review/events", json={
        "task_type": "segmentation", "task_id": str(page_id), "answer": "correct", "idempotency_key": uuid4().hex,
        "metadata": {"photo_count_revision": count_item["payload"]["photo_count_revision"]},
    })
    assert confirmed.status_code == 201, confirmed.text
    await held_downloads()

    rejected = await submit_geometry_review("/api/v1/review/events", json={
        "task_type": "geometry", "task_id": str(child_id), "answer": "wrong_object", "idempotency_key": uuid4().hex,
    })
    assert rejected.status_code == 201, rejected.text
    assert (await client.get("/api/v1/review/queue", params=child_task)).json() == []
    reprocessed = await client.post(f"{parent_url}/reprocess")
    assert reprocessed.status_code == 200, reprocessed.text
    assert reprocessed.json()["crop_download_urls"]["full_print"]
    assert (await client.get(f"{parent_url}/image?variant=full_print")).status_code == 200
    async with get_session_factory()() as session:
        parent = await session.get(ExtractedPhoto, parent_id)
        metadata = dict(parent.enhancement_applied)
        correction = dict(metadata["correction"])
        correction.pop(overlap_revision)
        parent.enhancement_applied = {**metadata, "correction": correction}
        await session.commit()
    await held_downloads()
    reprocessed = await client.post(f"{parent_url}/reprocess")
    assert reprocessed.status_code == 200 and reprocessed.json()["crop_download_urls"]["full_print"], reprocessed.text
    approved = await submit_geometry_review("/api/v1/review/events", json={
        "task_type": "geometry", "task_id": str(parent_id), "answer": "correct", "idempotency_key": uuid4().hex,
    })
    assert approved.status_code == 201, approved.text
    before_restore = (await client.get("/api/v1/review/queue", params=geometry_queue)).json()
    assert str(parent_id) not in {row["id"] for row in before_restore}
    assert (await client.get("/api/v1/review/queue", params=parent_task)).json() == []

    restored = await client.patch(f"{child_url}/bounding-box", json={"bounding_box": bounds(child_polygon),
                                   "segmentation_mask": {"polygon": child_polygon}})
    assert restored.status_code == 200 and not restored.json()["reviewed_non_print"], restored.text
    assert "warp_quad" not in restored.json()["segmentation_mask"]
    await held_downloads()
    after_restore = (await client.get("/api/v1/review/queue", params=geometry_queue)).json()
    assert str(parent_id) in {row["id"] for row in after_restore}
    disjoint = vertices([(190, 40), (243, 40), (243, 131), (190, 131)])
    moved = await save(child_url, disjoint)
    assert moved.status_code == 200, moved.text
    reprocessed = await client.post(f"{parent_url}/reprocess")
    assert reprocessed.status_code == 200 and reprocessed.json()["crop_download_urls"]["full_print"], reprocessed.text


async def test_failed_manual_addition_rolls_back_before_an_identical_retry(client, monkeypatch) -> None:
    from app.config import get_settings
    from app.database import get_session_factory
    from app.models import ExtractedPhoto
    from app.routers import photos

    monkeypatch.setenv("ENABLE_PHOTO_ORIENTATION_CORRECTION", "false")
    get_settings.cache_clear()
    image = np.full((120, 160, 3), 35, dtype=np.uint8)
    image[15:80, 12:90] = (64, 111, 145)
    album_id, page_id, _ = await _seed_review_data(image)
    photo_list = f"/api/v1/albums/{album_id}/photos"
    existing = (await client.get(photo_list)).json()
    body = {"request_id": str(uuid4()), "warp_quad": [
        {"x": x / 159, "y": y / 119} for x, y in [(12, 15), (89, 15), (89, 79), (12, 79)]
    ]}
    url = f"/api/v1/pages/{page_id}/manual-photos"

    # Leave the existing proposal on the other side of the album page.
    async with get_session_factory()() as session:
        neighbor = await session.get(ExtractedPhoto, UUID(existing[0]["id"]))
        neighbor.segmentation_mask = {"polygon": [
            {"x": x, "y": y} for x, y in ((.72, .2), (.9, .2), (.9, .7), (.72, .7))
        ]}
        await session.commit()

    async def fail(*args, **kwargs):
        raise RuntimeError("simulated enhancement failure")

    processed = get_settings().storage_path / "processed"
    before = set(processed.rglob("*.jpg"))
    with monkeypatch.context() as faults:
        faults.setattr(photos, "enhancement_task_logic", fail)
        with pytest.raises(RuntimeError, match="simulated enhancement failure"):
            await client.post(url, json=body)
    assert len((await client.get(photo_list)).json()) == 1
    assert set(processed.rglob("*.jpg")) == before
    assert (await client.post(url, json=body)).status_code == 201
    assert len((await client.get(photo_list)).json()) == 2


async def test_wide_motif_suggestion_requires_explicit_confirmation_before_download(client, monkeypatch) -> None:
    from app.config import get_settings

    monkeypatch.setenv("ENABLE_PRECISE_PHOTO_CROPS", "true")
    monkeypatch.setenv("ENABLE_BORDER_TRIM", "true")
    monkeypatch.setenv("ENABLE_PROTECTED_REGION_GUARD", "false")
    monkeypatch.setenv("ENABLE_PHOTO_ORIENTATION_CORRECTION", "false")
    get_settings.cache_clear()
    page = np.full((280, 350, 3), (61, 81, 103), np.uint8)
    page[29:251, 32:312] = (229, 237, 245)
    page[78:201, 94:247] = (77, 101, 147)
    _, _, photo_id = await _seed_review_data(page)
    url = f"/api/v1/photos/{photo_id}"
    box = {"x1": 32 / 350, "y1": 29 / 280, "x2": 312 / 350, "y2": 251 / 280}
    outer = [{"x": x / 349, "y": y / 279} for x, y in ((32, 29), (311, 29), (311, 250), (32, 250))]
    mask = {"polygon": outer, "warp_quad": outer, "source": "manual_quad"}
    edits = {"bounding_box": box, "segmentation_mask": mask}

    proposed = await client.patch(f"{url}/bounding-box", json=edits)
    saved_outer_only = await client.patch(f"{url}/bounding-box", json=edits)

    for response in (proposed, saved_outer_only):
        assert response.status_code == 200, response.text
        payload = response.json()
        motif = payload["enhancement_applied"]["correction"]["crop_modes"]["image_only"]
        assert motif["reason_codes"] == ["WIDE_MOTIF_CONFIRMATION_REQUIRED"]
        assert len(motif["suggested_warp_quad"]) == 4
        assert payload["crop_download_urls"]["image_only"] is None
        assert "image_quad" not in payload["segmentation_mask"]
    assert (await client.get(f"{url}/image?variant=image_only")).status_code == 404

    accepted = await client.patch(f"{url}/bounding-box", json={
        **edits, "segmentation_mask": {**mask, "image_quad": motif["suggested_warp_quad"]},
    })

    assert accepted.status_code == 200, accepted.text
    download = accepted.json()["crop_download_urls"]["image_only"]
    assert download and (await client.get(download)).status_code == 200


async def test_measured_print_size_recovers_both_oblique_crops_and_survives_reprocessing(submit_geometry_review, client, monkeypatch, tmp_path) -> None:
    from app.config import get_settings
    from app.database import get_session_factory
    from app.models import ExtractedPhoto, Page

    monkeypatch.setenv("ENABLE_PRECISE_PHOTO_CROPS", "true")
    monkeypatch.setenv("ENABLE_BORDER_TRIM", "false")
    monkeypatch.setenv("ENABLE_PHOTO_ORIENTATION_CORRECTION", "false")
    monkeypatch.setenv("ENABLE_PROTECTED_REGION_GUARD", "false")
    get_settings.cache_clear()
    width, height = 960, 780
    camera = np.float64([[900, 0, width / 2], [0, 900, height / 2], [0, 0, 1]])
    physical = np.float64([[-75, -50, 0], [75, -50, 0], [75, 50, 0], [-75, 50, 0]])
    projected, _ = cv2.projectPoints(physical, np.float64([0, 0.65, 0]), np.float64([0, 0, 350]), camera, None)
    print_corners = np.float32([[0, 0], [150, 0], [150, 100], [0, 100]])
    motif_corners = np.float32([[11, 17], [139, 17], [139, 89], [11, 89]])
    capture = cv2.getPerspectiveTransform(print_corners, projected.reshape(4, 2).astype(np.float32))
    original = np.full((101, 151, 3), 235, dtype=np.uint8)
    original[17:90, 11:140] = (35, 100, 166)
    image = cv2.warpPerspective(original, capture, (width, height), borderValue=(55, 55, 55))

    def vertices(points):
        captured = cv2.perspectiveTransform(points[None], capture)[0] / [width - 1, height - 1]
        return [{"x": float(x), "y": float(y)} for x, y in captured]

    outer, inner = vertices(print_corners), vertices(motif_corners)
    _, page_id, photo_id = await _seed_review_data(image)
    url = f"/api/v1/photos/{photo_id}"
    mask = {"polygon": outer, "warp_quad": outer, "image_quad": inner, "source": "manual_quad"}
    body = {
        "bounding_box": {"x1": 0.0, "y1": 0.0, "x2": 1.0, "y2": 1.0},
        "segmentation_mask": mask,
    }
    # Four correct corners still leave the metric shape of this single-axis view underconstrained.
    unmeasured = await client.patch(f"{url}/bounding-box", json=body)
    assert unmeasured.status_code == 200, unmeasured.text
    correction = unmeasured.json()["enhancement_applied"]["correction"]
    assert correction["crop_modes"]["full_print"]["reason_codes"] == ["PRINT_ASPECT_UNCERTAIN"]
    saved_review = unmeasured.json()["saved_full_print_review_url"]
    assert saved_review == f'{url}/image?variant=saved_full_print_review&v={correction["untrimmed_asset_id"]}'
    image_review = await client.get(saved_review)
    normal_untrimmed = await client.get(f"{url}/image?variant=source_corrected_untrimmed")
    assert image_review.status_code == normal_untrimmed.status_code == 200
    assert image_review.headers["content-type"] == "image/jpeg"
    assert image_review.headers["cache-control"] == "private, no-store"
    assert image_review.content == normal_untrimmed.content
    assert (await client.get(saved_review.split("&", 1)[0])).status_code == 404
    for variant in ("full_print", "image_only", "source_corrected_trimmed"):
        assert (await client.get(f"{url}/image?variant={variant}")).status_code == 404
    geometry, = (await client.get(f"/api/v1/review/queue?type=geometry&photo_id={photo_id}")).json()
    assert geometry["payload"]["saved_full_print_review_url"] == saved_review
    approve = await submit_geometry_review("/api/v1/review/events", json={"task_type": "geometry", "task_id": str(photo_id),
                                "answer": "correct", "idempotency_key": uuid4().hex})
    assert approve.status_code == 409
    reapplied = await client.post(f"{url}/reprocess")
    assert reapplied.status_code == 200, reapplied.text
    current = reapplied.json()["saved_full_print_review_url"]
    assert current and current != saved_review
    assert (await client.get(saved_review)).status_code == 404
    assert (await client.get(current)).content == image_review.content

    async with get_session_factory()() as session:
        photo = await session.get(ExtractedPhoto, photo_id)
        photo.segmentation_confidence = 0.95
        capture_hash = (await session.get(Page, page_id)).file_hash_sha256
        await session.commit()
    calibration = tmp_path / "camera-calibrations.json"
    calibration.write_text(json.dumps({"schema_version": 1, "captures": {capture_hash: {
        "width": width, "height": height, "camera_matrix": camera.tolist(), "calibration_id": "test-calibration",
    }}}), encoding="utf-8")
    monkeypatch.setenv("CAMERA_CALIBRATION_PATH", str(calibration))
    get_settings.cache_clear()

    preview = await client.post(f"{url}/reprocess")
    assert preview.status_code == 200, preview.text
    preview_photo = (await client.get(url)).json()
    correction = preview_photo["enhancement_applied"]["correction"]
    assert correction["aspect_ratio_estimation"]["method"] == "calibrated_camera"
    assert "CAMERA_ASPECT_REVIEW" in correction["crop_modes"]["full_print"]["reason_codes"]
    assert preview_photo["urls"]["full_print"] is None
    assert preview_photo["saved_full_print_review_url"] is None
    assert (await client.get(current)).status_code == 404
    assert (await client.get(preview_photo["urls"]["source_corrected"])).status_code == 200

    calibrated = await client.patch(f"{url}/bounding-box", json=body)
    assert calibrated.status_code == 200, calibrated.text
    payload = calibrated.json()
    assert payload["enhancement_applied"]["correction"]["camera_calibration"]["capture_sha256"] == capture_hash
    downloads = payload["crop_download_urls"]
    for mode, expected in (("full_print", 1.5), ("image_only", 128 / 72)):
        response = await client.get(downloads[mode])
        assert response.status_code == 200
        crop = cv2.imdecode(np.frombuffer(response.content, np.uint8), cv2.IMREAD_COLOR)
        assert (crop.shape[1] - 1) / (crop.shape[0] - 1) == pytest.approx(expected, rel=0.007)

    calibration.unlink()
    assert (await client.get(downloads["full_print"])).status_code == 404
    assert (await client.get(downloads["image_only"])).status_code == 404

    size = {"width": 150, "height": 100}
    mask["print_size_mm"] = size
    measured = await client.patch(f"{url}/bounding-box", json=body)
    assert measured.status_code == 200, measured.text
    assert measured.json()["aspect_ratio"] == pytest.approx(1.5)
    assert measured.json()["segmentation_mask"]["print_size_mm"] == size
    assert measured.json()["saved_full_print_review_url"] is None
    aspect = measured.json()["enhancement_applied"]["correction"]["aspect_ratio_estimation"]
    assert aspect == {"method": "known_print_aspect", "estimated_ratio": pytest.approx(1.5)}
    queue = await client.get("/api/v1/review/queue?type=geometry")
    assert queue.status_code == 200
    item = next(item for item in queue.json() if item["id"] == str(photo_id))
    assert item["payload"]["print_size_mm"] == size

    invalid = await client.patch(f"{url}/bounding-box", json={**body, "segmentation_mask": {
        **mask, "print_size_mm": {"width": 150, "height": 0},
    }})
    assert invalid.status_code == 422, invalid.text
    mask.pop("print_size_mm")
    mask["warp_quad"] = [outer[-1], *outer[:-1]]
    legacy_edit = await client.patch(f"{url}/bounding-box", json=body)
    assert legacy_edit.status_code == 200, legacy_edit.text
    assert legacy_edit.json()["segmentation_mask"]["print_size_mm"] == size
    assert legacy_edit.json()["segmentation_mask"]["warp_quad"] == outer
    reprocessed = await client.post(f"{url}/reprocess")
    assert reprocessed.status_code == 200, reprocessed.text
    payload = (await client.get(url)).json()
    assert payload["segmentation_mask"]["print_size_mm"] == size
    for mode, expected in (("full_print", 1.5), ("image_only", 128 / 72)):
        response = await client.get(payload["urls"][mode])
        assert response.status_code == 200
        crop = cv2.imdecode(np.frombuffer(response.content, np.uint8), cv2.IMREAD_COLOR)
        assert (crop.shape[1] - 1) / (crop.shape[0] - 1) == pytest.approx(expected, rel=0.007)
        expected_color = (235, 235, 235) if mode == "full_print" else (35, 100, 166)
        np.testing.assert_allclose(crop[5, crop.shape[1] // 2], expected_color, atol=6)

    # Numbering from another edge must rotate the result, with its measured axes intact.
    mask["warp_quad"] = [outer[-1], *outer[:-1]]
    mask["print_size_mm"] = {"width": 100, "height": 150}
    size_edit = {key: value for key, value in mask.items() if key != "image_quad"}
    renumbered = await client.patch(f"{url}/bounding-box", json={**body, "segmentation_mask": size_edit})
    assert renumbered.status_code == 200, renumbered.text
    assert renumbered.json()["segmentation_mask"]["image_quad"] == inner
    for mode, expected in (("full_print", 100 / 150), ("image_only", 72 / 128)):
        geometry = renumbered.json()["enhancement_applied"]["correction"]["crop_modes"][mode]
        assert geometry["status"] == "ready"
        assert (geometry["shape"]["width"] - 1) / (geometry["shape"]["height"] - 1) == pytest.approx(expected, rel=0.007)
    transformed = renumbered.json()["enhancement_applied"]["correction"]["warp_quad"]
    assert transformed[0] == pytest.approx(outer[-1], abs=1e-6)

    events_url = "/api/v1/review/events"
    old_revision = payload["geometry_review_revision"]
    new_revision = renumbered.json()["geometry_review_revision"]
    current, = (await client.get("/api/v1/review/queue", params={"type": "geometry", "photo_id": str(photo_id)})).json()
    assert old_revision != new_revision == current["payload"]["geometry_review_revision"]
    for action in ("wrong_object", "correct"):
        event = {"task_type": "geometry", "task_id": str(photo_id), "answer": action, "idempotency_key": uuid4().hex}
        for metadata in ({}, {"geometry_review_revision": old_revision}):
            rejected = await client.post(events_url, json={**event, "metadata": metadata})
            assert rejected.status_code == 409 and rejected.json()["detail"] == "geometry_review_stale"
    assert (await client.get("/api/v1/review/progress")).json()["by_type"] == {}
    fresh = (await client.get(url)).json()
    assert not fresh["reviewed_non_print"] and fresh["geometry_review_revision"] == new_revision
    for mode in ("full_print", "image_only"):
        assert (await client.get(fresh["crop_download_urls"][mode])).status_code == 200
    event = {"task_type": "geometry", "task_id": str(photo_id), "answer": "correct", "idempotency_key": uuid4().hex,
             "metadata": {"geometry_review_revision": new_revision}}
    accepted = await client.post(events_url, json=event)
    assert accepted.status_code == 201, accepted.text

    motif_edit = await client.patch(f"{url}/bounding-box", json={**body, "segmentation_mask": {
        **mask, "image_quad": vertices(np.float32([[11, 17], [139, 17], [116, 89], [38, 89]])),
    }})
    assert motif_edit.status_code == 200, motif_edit.text
    retry = await client.post(events_url, json=event)
    assert retry.status_code == 201 and retry.json() == {"id": accepted.json()["id"], "saved": True, "duplicate": True}
    modes = motif_edit.json()["enhancement_applied"]["correction"]["crop_modes"]
    assert modes["full_print"]["status"] == "ready"
    assert modes["image_only"]["reason_codes"] == ["IMAGE_RECTANGLE_INCONSISTENT"]
    assert (await client.get(f"{url}/image?variant=image_only")).status_code == 404

    monkeypatch.setenv("ENABLE_PRECISE_PHOTO_CROPS", "false")
    get_settings.cache_clear()
    assert (await client.post(f"{url}/reprocess")).status_code == 200
    original = await client.get(f"{url}/image?variant=source_corrected_untrimmed")
    crop = cv2.imdecode(np.frombuffer(original.content, np.uint8), cv2.IMREAD_COLOR)
    assert original.status_code == 200
    assert (crop.shape[1] - 1) / (crop.shape[0] - 1) == pytest.approx(100 / 150, rel=0.007)

    monkeypatch.setenv("ENABLE_PRECISE_PHOTO_CROPS", "true")
    get_settings.cache_clear()
    mask["print_size_mm"] = None
    cleared = await client.patch(f"{url}/bounding-box", json=body)
    assert cleared.status_code == 200, cleared.text
    assert "print_size_mm" not in cleared.json()["segmentation_mask"]
    assert (await client.get(f"{url}/image?variant=full_print")).status_code == 404


async def test_unvalidated_automatic_crop_is_available_only_as_a_review_preview(client, monkeypatch) -> None:
    from app.config import get_settings

    monkeypatch.setenv("ENABLE_BORDER_TRIM", "false")
    monkeypatch.setenv("ENABLE_PHOTO_ORIENTATION_CORRECTION", "false")
    monkeypatch.setenv("REQUIRE_CONFIDENCE_CALIBRATION", "true")
    get_settings.cache_clear()
    _, _, photo_id = await _seed_review_data()
    url = f"/api/v1/photos/{photo_id}"
    result = await client.post(f"{url}/reprocess")
    assert result.status_code == 200, result.text
    payload = (await client.get(url)).json()
    full_mode = payload["enhancement_applied"]["correction"]["crop_modes"]["full_print"]
    assert full_mode["geometry_status"] == "ready"
    assert full_mode["status"] == "review_needed"
    assert payload["urls"]["full_print"] is None
    assert (await client.get(payload["urls"]["source_corrected"])).status_code == 200

    confirmed = await client.patch(
        f"{url}/bounding-box",
        json={
            "bounding_box": payload["bounding_box"],
            "segmentation_mask": {
                "polygon": full_mode["warp_quad"],
                "warp_quad": full_mode["warp_quad"],
                "source": "manual_quad",
            },
        },
    )
    assert confirmed.status_code == 200, confirmed.text
    available = (await client.get(url)).json()
    assert (await client.get(available["urls"]["full_print"])).status_code == 200


async def test_review_can_remove_a_false_nested_candidate_from_occlusion_evidence(submit_geometry_review, client, monkeypatch) -> None:
    from app.config import get_settings
    from app.database import get_session_factory
    from app.models import ExtractedPhoto

    monkeypatch.setenv("ENABLE_PRECISE_PHOTO_CROPS", "true")
    monkeypatch.setenv("ENABLE_BORDER_TRIM", "false")
    monkeypatch.setenv("ENABLE_PROTECTED_REGION_GUARD", "false")
    monkeypatch.setenv("ENABLE_PHOTO_ORIENTATION_CORRECTION", "false")
    get_settings.cache_clear()
    image = np.full((200, 290, 3), 60, dtype=np.uint8)
    image[18:182, 21:272] = (99, 140, 190)
    image[51:148, 98:195] = (179, 50, 75)
    album_id, page_id, whole_id = await _seed_review_data(image)

    def edit(x1, y1, x2, y2):
        quad = [{"x": x / 289, "y": y / 199} for x, y in [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]]
        return {
            "bounding_box": {"x1": x1 / 289, "y1": y1 / 199, "x2": x2 / 289, "y2": y2 / 199},
            "segmentation_mask": {"source": "manual_quad", "polygon": quad, "warp_quad": quad},
        }

    nested_id = uuid4()
    nested_edit = edit(98, 51, 194, 147)
    async with get_session_factory()() as session:
        session.add(ExtractedPhoto(id=nested_id, page_id=page_id, album_id=album_id, storage_path="", **nested_edit))
        await session.commit()

    whole = f"/api/v1/photos/{whole_id}"
    nested = f"/api/v1/photos/{nested_id}"
    reviewed = await client.patch(f"{whole}/bounding-box", json=edit(21, 18, 271, 181))
    assert reviewed.status_code == 200, reviewed.text
    full_mode = reviewed.json()["enhancement_applied"]["correction"]["crop_modes"]["full_print"]
    assert full_mode["reason_codes"] == ["PRINT_POSSIBLY_OCCLUDED", "MANUAL_NEIGHBOR_OVERLAP"]
    expected = nested_edit["segmentation_mask"]["warp_quad"]
    for actual, point in zip(full_mode["neighbor_overlap_outline"][:-1], expected, strict=True):
        assert actual == pytest.approx(point)
    assert (await client.patch(f"{nested}/bounding-box", json=nested_edit)).status_code == 200
    assert (await client.get(f"{nested}/image?variant=full_print")).status_code == 404

    rejection = await submit_geometry_review("/api/v1/review/events", json={
        "task_type": "geometry", "task_id": str(nested_id), "answer": "wrong_object", "idempotency_key": str(uuid4()),
    })
    assert rejection.status_code == 201, rejection.text
    assert (await client.get(f"{nested}/image?variant=full_print")).status_code == 404
    assert (await client.get(f"{nested}/image?variant=source_corrected")).status_code == 200
    result = await client.post(f"{whole}/reprocess")
    assert result.status_code == 200, result.text
    assert (await client.get(f"{whole}/image?variant=full_print")).status_code == 200

    # A new explicit edit supersedes the rejection, without ignoring the new candidate.
    assert (await client.patch(f"{nested}/bounding-box", json=nested_edit)).status_code == 200
    assert (await client.post(f"{whole}/reprocess")).status_code == 200
    assert (await client.get(f"{whole}/image?variant=full_print")).status_code == 404


@pytest.mark.parametrize("corner_mount", [False, True], ids=["full-print-visible", "motif-only-visible"])
@pytest.mark.parametrize("with_refiner", [False, True], ids=["primary-detector", "visible-mask-refiner"])
async def test_automatic_crop_download_requires_current_album_validation(
    submit_geometry_review, client, monkeypatch, tmp_path, corner_mount, with_refiner, training_inventory,
) -> None:
    from app.config import get_settings
    from app.database import get_session_factory
    from app.models import ExtractedPhoto, Page
    from app.pipeline.confidence import CALIBRATION_SCHEMA, calibrate_confidence
    from app.pipeline.crop_validation import (
        CROP_VALIDATION_SCHEMA, crop_validation_evidence, crop_validation_profile, detector_calibration_profile,
    )
    from app.pipeline.segmentation_filter import SOURCE_CONTOUR_REVISION
    from app.services.pipeline import perspective_correction_task_logic

    model_path = tmp_path / "registered.onnx"
    model_path.write_bytes(b"registered-checkpoint-for-api-test")
    digest = hashlib.sha256(model_path.read_bytes()).hexdigest()
    registry_path = tmp_path / "manifest.json"
    registry = {"models": {model_path.name: {
        "sha256": digest, "license": "test", "input": {"width": 640},
        "training_provenance": training_inventory(model_path, digest),
    }}}
    refinement_key = "source_contour_refinement"
    refiner_metadata = {refinement_key: {"revision": SOURCE_CONTOUR_REVISION}}
    refiner = tmp_path / "visible-mask.onnx"
    refiner_bytes = b"registered-visible-mask-for-api-test"
    if with_refiner:
        refiner.write_bytes(refiner_bytes)
        refiner_digest = hashlib.sha256(refiner_bytes).hexdigest()
        registry["models"][refiner.name] = {
            "sha256": refiner_digest, "license": "test",
            "training_provenance": training_inventory(refiner, refiner_digest),
        }
        monkeypatch.setenv("VISIBLE_MASK_MODEL_PATH", str(refiner))
        refinement_key = "visible_mask_refinement"
        refiner_metadata = {refinement_key: {"model_sha256": refiner_digest, "refined_detections": 1}}
    registry_path.write_text(json.dumps(registry), encoding="utf-8")
    calibration_path = tmp_path / "calibration.json"
    validation_path = tmp_path / "crop-validation.json"
    monkeypatch.setenv("ENABLE_PRECISE_PHOTO_CROPS", "true")
    monkeypatch.setenv("ENABLE_BORDER_TRIM", "false")
    monkeypatch.setenv("ENABLE_PROTECTED_REGION_GUARD", "true")
    monkeypatch.setenv("ENABLE_PHOTO_ORIENTATION_CORRECTION", "false")
    monkeypatch.setenv("REQUIRE_CONFIDENCE_CALIBRATION", "true")
    monkeypatch.setenv("YOLO_MODEL_PATH", str(model_path))
    monkeypatch.setenv("MODEL_MANIFEST_PATH", str(registry_path))
    monkeypatch.setenv("REQUIRE_REGISTERED_MODELS", "true")
    monkeypatch.setenv("SEGMENTATION_CALIBRATION_PATH", str(calibration_path))
    monkeypatch.setenv("CROP_VALIDATION_REPORT_PATH", str(validation_path))
    get_settings.cache_clear()
    detector = detector_calibration_profile(get_settings())
    calibration_path.write_text(json.dumps({
        "schema_version": CALIBRATION_SCHEMA,
        "independent_holdout_id": "fixture-confidence-holdout",
        "source_groups": ["test-calibration-album"], "sample_count": 20,
        "input_sha256": "a" * 64, "dataset_manifest_sha256": "b" * 64,
        "source_images": {"e" * 64: "test-calibration-album"},
        "detector_profile": detector,
        "strata": {"default": {"raw": [0, 1], "calibrated": [0, 1]}},
    }), encoding="utf-8")
    profile = crop_validation_profile(get_settings())
    image = np.full((120, 160, 3), 105, dtype=np.uint8)
    image[24:108, 16:128] = 240
    margin = 13 if corner_mount else 6
    left, right, top, bottom = 16 + margin, 127 - margin, 24 + margin, 107 - margin
    image[top:bottom + 1, left:right + 1] = (50, 90, 150)
    if corner_mount:
        cv2.fillConvexPoly(image, np.int32([[13, 21], [36, 21], [13, 44]]), (48, 55, 61))
    album_id, page_id, photo_id = await _seed_review_data(image)
    outer = [{"x": x / 159, "y": y / 119} for x, y in [(16, 24), (127, 24), (127, 107), (16, 107)]]
    inner = [{"x": x / 159, "y": y / 119} for x, y in [(left, top), (right, top), (right, bottom), (left, bottom)]]
    async with get_session_factory()() as session:
        page = await session.get(Page, page_id)
        page.processing_metadata = {"steps": {"segmentation": {
            "model_artifact": {"sha256": digest}, "validation_profile": profile,
            **refiner_metadata,
        }}}
        photo = await session.get(ExtractedPhoto, photo_id)
        _, calibration = calibrate_confidence(
            photo.segmentation_confidence, "default", calibration_path, detector_profile=detector,
        )
        assert calibration["calibrated"] is True
        photo.enhancement_applied = {**photo.enhancement_applied, "segmentation_confidence": calibration}
        photo.segmentation_mask = {"source": "classical_boundary", "polygon": outer, "warp_quad": outer, "image_quad": inner}
        await session.commit()

    url = f"/api/v1/photos/{photo_id}"
    crop_variants = ("image_only",) if corner_mount else ("full_print", "image_only")
    variants = (*crop_variants, "source_corrected_trimmed")
    result = await client.post(f"{url}/reprocess")
    assert result.status_code == 200, result.text
    payload = (await client.get(url)).json()
    crop_modes = payload["enhancement_applied"]["correction"]["crop_modes"]
    for mode in crop_variants:
        assert crop_modes[mode]["geometry_status"] == "ready"
        assert crop_modes[mode]["reason_codes"] == ["CROP_VALIDATION_UNAVAILABLE"]
    assert all(payload["urls"][name] is None for name in ("full_print", "image_only", "source_corrected_trimmed"))
    assert (await client.get(payload["urls"]["source_corrected"])).status_code == 200

    counts = {
        "verified": 20, "obstructed": 0, "obstructed_routed_to_review": 0, "verified_ready_failures": 0,
        "real": {"expected": 20, "automatic_release_passes": 19},
        "real_by_group": {
            "test-final-album-a": {"expected": 10, "automatic_release_passes": 10},
            "test-final-album-b": {"expected": 10, "automatic_release_passes": 9},
        },
    }
    report = {
        "schema_version": CROP_VALIDATION_SCHEMA,
        "model_sha256": digest,
        "validation_profile": profile,
        "legacy_crop": False,
        "geometry_only": False,
        "classical_only": False,
        "split": {"album_independent": True, "role": "test"},
        "dataset_manifest_sha256": "d" * 64,
        "metrics": {"page_count": 2, "expected_photos": 20},
        "crop_modes": {"full_print": counts, "image_only": counts},
        "release_source_images": {"a" * 64: "test-final-album-a", "b" * 64: "test-final-album-b"},
        "training_provenance_sha256": {
            name: model["training_provenance"]["sha256"] for name, model in registry["models"].items()
        },
        "validation_gates": {
            "locked_real_album_manifest": True, "locked_reference_artifacts": True, "calibration_holdout_disjoint": True,
            "training_calibration_disjoint": True, "training_release_disjoint": True,
            "shared_recoverable_album_groups": True,
            "standard_detection_thresholds": True,
            "detection": True, "final_instance_segmentation": True,
            "rendered_crop_modes": {"full_print": True, "image_only": True},
        },
    }
    validation_path.write_text(json.dumps(report | {"schema_version": CROP_VALIDATION_SCHEMA - 1}), encoding="utf-8")
    result = await client.post(f"{url}/reprocess")
    assert result.status_code == 200, result.text
    stale = (await client.get(url)).json()["urls"]
    assert all(stale[mode] is None for mode in crop_variants)

    missing_shared_groups = report["validation_gates"] | {"shared_recoverable_album_groups": False}
    validation_path.write_text(json.dumps(report | {"validation_gates": missing_shared_groups}), encoding="utf-8")
    result = await client.post(f"{url}/reprocess")
    assert result.status_code == 200, result.text
    unavailable = (await client.get(url)).json()["urls"]
    assert all(unavailable[mode] is None for mode in crop_variants)

    failing_group_counts = {
        **counts, "real": {"expected": 20, "automatic_release_passes": 18},
        "real_by_group": counts["real_by_group"] | {
            "test-final-album-b": {"expected": 10, "automatic_release_passes": 8},
        },
    }
    separate_groups = {
        **counts, "real_by_group": {
            "test-final-album-b": counts["real_by_group"]["test-final-album-b"],
            "test-final-album-c": counts["real_by_group"]["test-final-album-a"],
        },
    }
    for overrides in (
        {"split": {"album_independent": True, "role": "development"}},
        {"training_provenance_sha256": {}},
        {"release_source_images": {"c" * 64: "test-final-album-a", "b" * 64: "test-final-album-b"}},
        {"release_source_images": {"e" * 64: "test-final-album-a", "b" * 64: "test-final-album-b"}},
        {"crop_modes": {"full_print": counts, "image_only": failing_group_counts}},
        {
            "metrics": {"page_count": 3, "expected_photos": 20},
            "release_source_images": report["release_source_images"] | {"f" * 64: "test-final-album-c"},
            "crop_modes": {"full_print": counts, "image_only": separate_groups},
        },
    ):
        validation_path.write_text(json.dumps(report | overrides), encoding="utf-8")
        assert crop_validation_evidence(get_settings(), digest) is None

    monkeypatch.setenv("REQUIRE_CONFIDENCE_CALIBRATION", "false")
    get_settings.cache_clear()
    uncalibrated_profile = crop_validation_profile(get_settings())
    validation_path.write_text(json.dumps(report | {"validation_profile": uncalibrated_profile}), encoding="utf-8")
    assert crop_validation_evidence(get_settings(), digest) is None
    monkeypatch.setenv("REQUIRE_CONFIDENCE_CALIBRATION", "true")
    get_settings.cache_clear()

    validation_path.write_text(json.dumps(report), encoding="utf-8")
    async with get_session_factory()() as session:
        page = await session.get(Page, page_id)
        steps = page.processing_metadata["steps"]
        page.processing_metadata = {"steps": {**steps, "segmentation": {
            **steps["segmentation"], refinement_key: {"error": "missing evaluated refinement"},
        }}}
        await session.commit()
    result = await client.post(f"{url}/reprocess")
    assert result.status_code == 200, result.text
    unavailable = (await client.get(url)).json()["urls"]
    assert all(unavailable[mode] is None for mode in crop_variants)
    async with get_session_factory()() as session:
        page = await session.get(Page, page_id)
        steps = page.processing_metadata["steps"]
        page.processing_metadata = {"steps": {
            **steps, "segmentation": {**steps["segmentation"], **refiner_metadata},
        }}
        await session.commit()
    result = await client.post(f"{url}/reprocess")
    assert result.status_code == 200, result.text
    payload = (await client.get(url)).json()
    released = payload["urls"]
    for name in variants:
        assert (await client.get(released[name])).status_code == 200
    album_photos = (await client.get(f"/api/v1/albums/{album_id}/photos")).json()
    assert album_photos[0]["crop_download_urls"] == {mode: released[mode] for mode in ("full_print", "image_only")}

    if corner_mount:
        correction = payload["enhancement_applied"]["correction"]
        assert correction["crop_modes"]["image_only"]["independent_of_full_print"] is True
        assert payload["status"] == "review_needed"
        assert released["full_print"] is None
        assert (await client.get(f"{url}/image?variant=full_print")).status_code == 404
        assert (await client.get(released["source_corrected_untrimmed"])).status_code == 200
        approval = await submit_geometry_review("/api/v1/review/events", json={
            "task_type": "geometry", "task_id": str(photo_id), "answer": "correct", "idempotency_key": uuid4().hex,
        })
        assert approval.status_code == 409
        crop = cv2.imdecode(np.frombuffer((await client.get(released["image_only"])).content, np.uint8), cv2.IMREAD_COLOR)
        assert crop.shape[:2] == (bottom - top + 1, right - left + 1)
        assert np.max(np.abs(crop[8:-8, 8:-8].astype(np.int16) - [50, 90, 150])) <= 5
        async with get_session_factory()() as session:
            assert (await perspective_correction_task_logic(session, photo_id, get_settings(), skip_if_exists=True))["idempotent_replay"]
            Path(correction["trimmed_path"]).unlink()
            repaired = await perspective_correction_task_logic(session, photo_id, get_settings(), skip_if_exists=True)
            assert repaired["review_required"] and Path(repaired["trimmed_path"]).is_file()
        released = (await client.get(url)).json()["urls"]

    monkeypatch.setenv("ENABLE_BORDER_TRIM", "true")
    get_settings.cache_clear()
    assert (await client.get(released["image_only"])).status_code == 404
    monkeypatch.setenv("ENABLE_BORDER_TRIM", "false")
    get_settings.cache_clear()
    model_path.write_bytes(b"another-checkpoint")
    checkpoint_variant = "image_only" if corner_mount else "full_print"
    assert (await client.get(released[checkpoint_variant])).status_code == 404
    model_path.write_bytes(b"registered-checkpoint-for-api-test")
    assert (await client.get(released[checkpoint_variant])).status_code == 200
    registry_path.write_text("{}", encoding="utf-8")
    assert (await client.get(released["image_only"])).status_code == 404
    registry_path.write_text(json.dumps(registry), encoding="utf-8")
    assert (await client.get(released["image_only"])).status_code == 200
    if with_refiner:
        refiner.write_bytes(b"replacement-mask-model")
        assert (await client.get(released["image_only"])).status_code == 404
        refiner.write_bytes(refiner_bytes)
        assert (await client.get(released["image_only"])).status_code == 200
    validation_path.unlink()
    revoked = (await client.get(url)).json()["urls"]
    for name in variants:
        assert revoked[name] is None
        assert (await client.get(released[name])).status_code == 404
    assert (await client.get(revoked["source_corrected"])).status_code == 200
    approval = await submit_geometry_review("/api/v1/review/events", json={
        "task_type": "geometry", "task_id": str(photo_id), "answer": "correct", "idempotency_key": uuid4().hex,
    })
    assert approval.status_code == 409

    async with get_session_factory()() as session:
        page = await session.get(Page, page_id)
        steps = page.processing_metadata["steps"] | {"protected_regions": {"regions": [{
            "type": "face", "source": "manual", "box": {"x1": 0.05, "y1": 0.39, "x2": 0.15, "y2": 0.60},
        }]}}
        page.processing_metadata = {"steps": steps}
        await session.commit()
    result = await client.post(f"{url}/reprocess")
    assert result.status_code == 200, result.text
    payload = (await client.get(url)).json()
    assert payload["enhancement_applied"]["correction"]["status"] == "rejected"
    assert payload["enhancement_applied"]["correction"]["crop_modes"]["image_only"]["status"] == "review_needed"
    assert (await client.get(released["source_corrected_trimmed"])).status_code == 404


async def test_reprocessing_keeps_the_review_preview_for_an_automatic_face_hint(client, monkeypatch) -> None:
    from app.config import get_settings
    from app.database import get_session_factory
    from app.models import Page

    monkeypatch.setenv("ENABLE_PRECISE_PHOTO_CROPS", "true")
    monkeypatch.setenv("ENABLE_PROTECTED_REGION_GUARD", "true")
    monkeypatch.setenv("ENABLE_PHOTO_ORIENTATION_CORRECTION", "false")
    get_settings.cache_clear()
    _, page_id, photo_id = await _seed_review_data()
    async with get_session_factory()() as session:
        page = await session.get(Page, page_id)
        page.processing_metadata = {"steps": {"protected_regions": {"regions": [{
            "type": "face", "source": "opencv_haarcascade_frontalface_default",
            "box": {"x1": 0.05, "y1": 0.39, "x2": 0.15, "y2": 0.60},
        }]}}}
        await session.commit()
    url = f"/api/v1/photos/{photo_id}"
    result = await client.post(f"{url}/reprocess")
    assert result.status_code == 200, result.text
    payload = (await client.get(url)).json()
    full_mode = payload["enhancement_applied"]["correction"]["crop_modes"]["full_print"]
    assert "POSSIBLE_CONTENT_CUTOFF" in full_mode["reason_codes"]
    assert payload["urls"]["full_print"] is None
    assert (await client.get(payload["urls"]["source_corrected_untrimmed"])).status_code == 200


async def test_curved_motif_preview_is_an_inline_png_and_remains_a_review_asset(client, monkeypatch, curved_motif_page) -> None:
    from app.config import get_settings
    from app.database import get_session_factory
    from app.models import Asset, ExtractedPhoto, Page
    from app.pipeline.image_ops import save_png, sha256_file
    from app.services.pipeline import perspective_correction_task_logic

    monkeypatch.setenv("ENABLE_PRECISE_PHOTO_CROPS", "true")
    monkeypatch.setenv("ENABLE_PHOTO_ORIENTATION_CORRECTION", "false")
    monkeypatch.setenv("ENABLE_PROTECTED_REGION_GUARD", "false")
    get_settings.cache_clear()
    page_image, outer = curved_motif_page
    album_id, page_id, photo_id = await _seed_review_data(page_image)
    height, width = page_image.shape[:2]
    async with get_session_factory()() as session:
        page = await session.get(Page, page_id)
        photo = await session.get(ExtractedPhoto, photo_id)
        source = Path(page.storage_path).with_suffix(".png")
        save_png(source, page_image)
        page.storage_path = str(source)
        page.file_hash_sha256 = sha256_file(source)
        photo.segmentation_confidence = None
        photo.segmentation_mask = {"source": "manual_quad", "warp_quad": [
            {"x": float(x) / (width - 1), "y": float(y) / (height - 1)} for x, y in outer
        ]}
        await session.commit()

    url = f"/api/v1/photos/{photo_id}"
    result = await client.post(f"{url}/reprocess")
    assert result.status_code == 200, result.text
    photo = (await client.get(url)).json()
    correction = photo["enhancement_applied"]["correction"]
    assert correction["crop_modes"]["image_only"]["status"] == "review_needed"
    assert photo["urls"]["image_only"] is None
    assert (await client.get(f"{url}/image?variant=image_only")).status_code == 404
    assert (await client.get(photo["urls"]["full_print"])).headers["content-type"] == "image/jpeg"
    preview_url = photo["urls"]["image_only_preview"]
    preview = await client.get(preview_url)
    assert preview.status_code == 200 and preview.headers["content-type"] == "image/png"
    assert preview.headers["content-disposition"].startswith("inline")
    decoded = cv2.imdecode(np.frombuffer(preview.content, np.uint8), cv2.IMREAD_UNCHANGED)
    assert decoded.shape[2] == 4 and np.any(decoded[..., 3] == 0) and np.any(decoded[..., 3] == 255)
    variants = (await client.get(f"{url}/variants")).json()
    assert "image_only_preview" not in variants["variants"]
    queue = (await client.get(f"/api/v1/review/queue?type=geometry&album_id={album_id}")).json()
    assert queue[0]["payload"]["variant_urls"]["image_only_preview"] == preview_url

    async with get_session_factory()() as session:
        asset = await session.get(Asset, UUID(correction["image_only_preview_asset_id"]))
        assert asset.role == "motif_review_preview" and asset.media_type == "image/png"
        Path(asset.storage_path).unlink()
        assert (await client.get(preview_url)).status_code == 404
        repair = await perspective_correction_task_logic(session, photo_id, get_settings(), skip_if_exists=True)
        assert not repair.get("idempotent_replay")
    repaired = (await client.get(url)).json()["urls"]["image_only_preview"]
    assert repaired != preview_url and (await client.get(repaired)).content == preview.content

    monkeypatch.setenv("ENABLE_BORDER_TRIM", "false")
    get_settings.cache_clear()
    assert (await client.post(f"{url}/reprocess")).status_code == 200
    assert (await client.get(repaired)).status_code == 404


@pytest.mark.parametrize("jpeg_caption", [False, True])
async def test_stock_motif_preview_requires_an_empty_margin_and_keeps_exports_held(submit_geometry_review, client, monkeypatch, jpeg_caption):
    from app.config import get_settings
    from app.pipeline.segmentation import SegmentationDetection, SegmentationResult
    from app.services import pipeline

    monkeypatch.setenv("AUTO_ANALYZE_UPLOADS", "true")
    monkeypatch.setenv("MAX_UPLOAD_SIZE", "1048576")
    monkeypatch.setenv("MAX_REQUEST_SIZE", "2097152")
    monkeypatch.setenv("ENABLE_PRECISE_PHOTO_CROPS", "true")
    monkeypatch.setenv("ENABLE_BORDER_TRIM", "true")
    monkeypatch.setenv("ENABLE_PROTECTED_REGION_GUARD", "false")
    monkeypatch.setenv("ENABLE_PHOTO_ORIENTATION_CORRECTION", "false")
    monkeypatch.setattr(pipeline, "run_paddle_ocr", lambda *_: [])
    get_settings.cache_clear()

    page = np.full((670, 820, 3), (124, 136, 146), np.uint8)
    page[80:620, 120:711] = (189, 212, 228)
    page[98:602, 138:693] = (109, 125, 143)
    if jpeg_caption:
        cv2.putText(page, "SEP 1954", (207, 90), cv2.FONT_HERSHEY_SIMPLEX, .26, (151, 174, 190), 1, cv2.LINE_AA)
    polygon = [{"x": x / 819, "y": y / 669} for x, y in ((120, 67), (710, 67), (710, 619), (120, 619))]
    box = {"x1": 120 / 819, "y1": 67 / 669, "x2": 710 / 819, "y2": 619 / 669}
    detection = SegmentationDetection(box, {"source": "yolo_seg_mask", "polygon": polygon}, .9, 1., True, [])
    monkeypatch.setattr(pipeline, "segment_album_page", lambda *_, **__: SegmentationResult([detection], {}))

    if jpeg_caption:
        ok, upload = cv2.imencode(".jpg", page, [cv2.IMWRITE_JPEG_QUALITY, 75])
        name, media = "caption.jpg", "image/jpeg"
    else:
        ok, upload = cv2.imencode(".png", page)
        name, media = "empty-margin.png", "image/png"
    assert ok
    album_id = (await client.post("/api/v1/albums", json={"name": "Measured motif review"})).json()["id"]
    album_url = f"/api/v1/albums/{album_id}"
    uploaded = await client.post(f"{album_url}/pages/upload", files={"files": (name, upload.tobytes(), media)})
    assert uploaded.status_code == 201, uploaded.text
    photo, = (await client.get(f"{album_url}/photos")).json()
    url = f"/api/v1/photos/{photo['id']}"
    photo = (await client.get(url)).json()
    for reprocess in (False, True):
        if reprocess:
            assert (await client.post(f"{url}/reprocess")).status_code == 200
            photo = (await client.get(url)).json()
        correction = photo["enhancement_applied"]["correction"]
        full, motif = (correction["crop_modes"][mode] for mode in ("full_print", "image_only"))
        assert all(mode["status"] == "review_needed" and "PRINT_BOUNDARY_AMBIGUOUS" in mode["reason_codes"]
                   for mode in (full, motif))
        assert correction["boundary_refinement"]["bottom"]["review_stock_fit"]["support"] == 1
        for variant in ("full_print", "image_only", "source_corrected_trimmed"):
            assert (await client.get(f"{url}/image?variant={variant}")).status_code == 404
        assert (await client.get(photo["urls"]["source_corrected_untrimmed"])).status_code == 200
        preview_url = photo["urls"].get("image_only_preview")
        if jpeg_caption:
            assert preview_url is None and "outline_preview" not in motif
            assert motif["margin_mark_regions"]
        else:
            assert full["suggested_warp_quad"] == full["warp_quad"]
            assert motif["outline_preview"]["source"] == "paper_stock_motif"
            assert motif["outline_preview"]["source_outline"] == motif["warp_quad"]
            preview = await client.get(preview_url)
            assert preview.status_code == 200 and preview.headers["content-type"] == "image/png"
            assert preview.headers["content-disposition"].startswith("inline")
            variants = (await client.get(f"{url}/variants")).json()["variants"]
            assert "image_only_preview" not in variants
            queue, = (await client.get(f"/api/v1/review/queue?type=geometry&album_id={album_id}")).json()
            assert queue["payload"]["variant_urls"]["image_only_preview"] == preview_url

    approval = await submit_geometry_review("/api/v1/review/events", json={
        "task_type": "geometry", "task_id": photo["id"], "answer": "correct", "idempotency_key": str(uuid4()),
    })
    assert approval.status_code == 409
    if jpeg_caption:
        regions = motif["margin_mark_regions"]
        saved = await client.patch(f"{url}/bounding-box", json={
            "bounding_box": photo["bounding_box"], "confirm_print_outline": True,
            "segmentation_mask": {"source": "manual_quad", "warp_quad": full["warp_quad"],
                                  "polygon": full["warp_quad"], "print_size_mm": {"width": 118, "height": 107.8}},
        })
        assert saved.status_code == 200, saved.text
        for reprocess in (False, True):
            replay = await client.post(f"{url}/reprocess") if reprocess else saved
            assert replay.status_code == 200, replay.text
            detail = (await client.get(url)).json()
            modes = detail["enhancement_applied"]["correction"]["crop_modes"]
            assert modes["image_only"]["reason_codes"] == ["IMAGE_MARGIN_MARKS_REVIEW"]
            assert modes["image_only"]["margin_mark_regions"] == regions
            assert (await client.get(f"{url}/image?variant=full_print")).status_code == 200
            for variant in ("image_only", "image_only_preview", "source_corrected_trimmed"):
                assert (await client.get(f"{url}/image?variant={variant}")).status_code == 404
    else:
        implicit = await client.patch(f"{url}/bounding-box", json={
            "bounding_box": photo["bounding_box"],
            "segmentation_mask": {"source": "manual_quad", "warp_quad": full["warp_quad"],
                                  "polygon": full["warp_quad"], "image_quad": motif["warp_quad"]},
        })
        assert implicit.status_code == 409 and implicit.json()["detail"] == "print_outline_review_required", implicit.text


async def test_full_print_outline_preview_stays_out_of_exports_and_respects_review(submit_geometry_review, client, monkeypatch) -> None:
    from types import SimpleNamespace

    from app.config import get_settings
    from app.database import get_session_factory
    from app.models import Asset, ExtractedPhoto
    from app.services import pipeline

    monkeypatch.setenv("ENABLE_PRECISE_PHOTO_CROPS", "true")
    monkeypatch.setenv("ENABLE_BORDER_TRIM", "false")
    monkeypatch.setenv("ENABLE_PHOTO_ORIENTATION_CORRECTION", "false")
    monkeypatch.setenv("ENABLE_PROTECTED_REGION_GUARD", "false")
    get_settings.cache_clear()
    image = np.full((240, 320, 3), 55, np.uint8)
    image[25:215, 30:290] = (224, 232, 240)
    image[39:200, 45:275] = (70, 100, 150)
    album_id, _, photo_id = await _seed_review_data(image)
    quad = [{"x": x / 319, "y": y / 239} for x, y in [(30, 25), (289, 25), (289, 214), (30, 214)]]
    mask = {"source": "manual_quad", "polygon": quad, "warp_quad": quad}
    box = {"x1": 30 / 319, "y1": 25 / 239, "x2": 289 / 319, "y2": 214 / 239}
    async with get_session_factory()() as session:
        photo = await session.get(ExtractedPhoto, photo_id)
        photo.segmentation_confidence = None
        photo.segmentation_mask = mask
        photo.bounding_box = box
        await session.commit()

    pixels = np.zeros((19, 26, 4), np.uint8)
    pixels[2:17, 2:24] = (53, 102, 201, 255)
    preview_enabled = True
    render_crop = pipeline.correct_photo_crop

    def render_with_preview(*args, **kwargs):
        crop = render_crop(*args, **kwargs)
        if preview_enabled:
            full_mode = crop.metadata["crop_modes"]["full_print"]
            full_mode.update(
                status="review_needed",
                reason_codes=["PRINT_SCALLOPED_OUTLINE_REVIEW"],
                outline_preview={"shape": {"height": pixels.shape[0], "width": pixels.shape[1]}},
            )
        return SimpleNamespace(
            rectified_untrimmed=crop.rectified_untrimmed, image=crop.image, metadata=crop.metadata,
            image_only=crop.image_only, image_only_preview=crop.image_only_preview,
            full_print_preview=pixels if preview_enabled else None,
        )

    monkeypatch.setattr(pipeline, "correct_photo_crop", render_with_preview)
    url = f"/api/v1/photos/{photo_id}"
    result = await client.post(f"{url}/reprocess")
    assert result.status_code == 200, result.text
    photo = (await client.get(url)).json()
    correction = photo["enhancement_applied"]["correction"]
    assert photo["urls"]["full_print"] is None
    assert photo["crop_download_urls"]["full_print"] is None
    assert (await client.get(f"{url}/image?variant=full_print")).status_code == 404
    assert (await client.get(photo["urls"]["source_corrected_untrimmed"])).headers["content-type"] == "image/jpeg"
    preview_url = photo["urls"]["full_print_preview"]
    preview = await client.get(preview_url)
    assert preview.status_code == 200 and preview.headers["content-type"] == "image/png"
    assert preview.headers["content-disposition"].startswith("inline")
    assert preview.headers["cache-control"] == "private, no-store"
    decoded = cv2.imdecode(np.frombuffer(preview.content, np.uint8), cv2.IMREAD_UNCHANGED)
    assert np.array_equal(decoded, pixels)
    variants = (await client.get(f"{url}/variants")).json()
    assert "full_print_preview" not in variants["variants"]
    assert variants["selected_variant"] != "full_print_preview"
    queue = (await client.get(f"/api/v1/review/queue?type=geometry&album_id={album_id}")).json()
    assert queue[0]["payload"]["variant_urls"]["full_print_preview"] == preview_url

    async with get_session_factory()() as session:
        asset = await session.get(Asset, UUID(correction["full_print_preview_asset_id"]))
        assert asset.role == "full_print_review_preview" and asset.media_type == "image/png"
        replay = await pipeline.perspective_correction_task_logic(session, photo_id, get_settings(), skip_if_exists=True)
        assert replay["idempotent_replay"]
        Path(asset.storage_path).unlink()
        assert (await client.get(preview_url)).status_code == 404
        repair = await pipeline.perspective_correction_task_logic(session, photo_id, get_settings(), skip_if_exists=True)
        assert not repair.get("idempotent_replay")
    repaired = (await client.get(url)).json()["urls"]["full_print_preview"]
    assert repaired != preview_url and (await client.get(repaired)).content == preview.content
    rejected = await submit_geometry_review("/api/v1/review/events", json={
        "task_type": "geometry", "task_id": str(photo_id), "answer": "wrong_object", "idempotency_key": str(uuid4()),
    })
    assert rejected.status_code == 201, rejected.text
    assert (await client.get(repaired)).status_code == 404

    preview_enabled = False
    edit = await client.patch(f"{url}/bounding-box", json={"bounding_box": box, "segmentation_mask": mask})
    assert edit.status_code == 200, edit.text
    photo = (await client.get(url)).json()
    assert photo["urls"]["full_print_preview"] is None
    assert (await client.get(photo["crop_download_urls"]["full_print"])).headers["content-type"] == "image/jpeg"


async def test_invalid_neighbor_mask_can_still_show_an_occlusion(client, monkeypatch) -> None:
    from app.config import get_settings
    from app.database import get_session_factory
    from app.models import ExtractedPhoto

    monkeypatch.setenv("ENABLE_PRECISE_PHOTO_CROPS", "true")
    monkeypatch.setenv("ENABLE_PHOTO_ORIENTATION_CORRECTION", "false")
    get_settings.cache_clear()
    album_id, page_id, photo_id = await _seed_review_data()

    def vertices(points):
        return [{"x": x / 159, "y": y / 119} for x, y in points]

    outer = vertices([(16, 24), (127, 24), (127, 107), (16, 107)])
    visible = vertices([(16, 24), (63, 24), (63, 43), (81, 43), (81, 24), (127, 24), (127, 107), (16, 107)])
    neighbor = vertices([(63, 2), (81, 2), (81, 43), (63, 43)])
    async with get_session_factory()() as session:
        photo = await session.get(ExtractedPhoto, photo_id)
        photo.segmentation_mask = {"source": "yolo_seg_mask", "warp_quad": outer, "polygon": visible}
        session.add(ExtractedPhoto(
            page_id=page_id, album_id=album_id, storage_path="", geometry_valid=False,
            bounding_box={"x1": 63 / 159, "y1": 2 / 119, "x2": 81 / 159, "y2": 43 / 119},
            segmentation_mask={"source": "yolo_seg_mask", "polygon": neighbor},
        ))
        await session.commit()
    url = f"/api/v1/photos/{photo_id}"
    result = await client.post(f"{url}/reprocess")
    assert result.status_code == 200, result.text
    modes = (await client.get(url)).json()["enhancement_applied"]["correction"]["crop_modes"]
    assert "PRINT_POSSIBLY_OCCLUDED" in modes["full_print"]["reason_codes"]


async def test_visible_rim_is_persisted_without_releasing_a_curved_print(client, monkeypatch) -> None:
    from app.config import get_settings
    from app.database import get_session_factory
    from app.evaluation.crop_fixtures import _page
    from app.evaluation.cropping import evaluate_page_crops
    from app.models import ExtractedPhoto

    monkeypatch.setenv("ENABLE_PHOTO_ORIENTATION_CORRECTION", "false")
    get_settings.cache_clear()
    fixture = _page("curl", np.random.default_rng(31523))
    height, width = fixture.image.shape[:2]
    instance = fixture.prints[0]
    contours, _ = cv2.findContours(instance.reference_modes["image_only"].mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    points = cv2.approxPolyDP(max(contours, key=cv2.contourArea), 0.45, True).reshape(-1, 2) / [width - 1, height - 1]
    raw = [{"x": float(x), "y": float(y)} for x, y in points]
    _, _, photo_id = await _seed_review_data(fixture.image)
    async with get_session_factory()() as session:
        photo = await session.get(ExtractedPhoto, photo_id)
        photo.segmentation_mask = {"polygon": raw, "visible_polygon": raw, "source": "yolo_seg_mask"}
        await session.commit()

    url = f"/api/v1/photos/{photo_id}"
    repro = await client.post(f"{url}/reprocess")
    assert repro.status_code == 200, repro.text
    first = (await client.get(url)).json()
    assert first["urls"]["full_print"] is None and first["urls"]["image_only"] is None
    assert first["urls"]["source_corrected"]
    visible = first["segmentation_mask"]
    assert visible["visible_polygon_source"] == "visible_paper_rim"
    assert visible["polygon"] == visible["proposal_visible_polygon"] == raw
    reference = [instance.visible_polygon]
    refined = [[(point["x"], point["y"]) for point in visible["visible_polygon"]]]
    before = evaluate_page_crops(reference, [points.tolist()], width=width, height=height)
    after = evaluate_page_crops(reference, refined, width=width, height=height)
    assert after.mask_iou >= 0.95 and after.mask_iou > before.mask_iou + 0.10

    assert (await client.post(f"{url}/reprocess")).status_code == 200
    assert (await client.get(url)).json()["segmentation_mask"] == visible
