from uuid import uuid4

import cv2
import numpy as np
import pytest

from app.pipeline.embossed_album_pockets import EMBOSSED_POCKET_PROPOSAL, embossed_pocket_traces
from app.pipeline.projective_prints import PROJECTIVE_PRINT_OUTLINE_REVIEW
from app.pipeline.segmentation import SegmentationDetection, SegmentationResult, _defer_embossed_pocket_claims


_POCKET = np.float32([[223, 148], [1423, 148], [1423, 1048], [223, 1048]])
_PEER = np.float32([[301, 1184], [1345, 1184], [1345, 1692], [301, 1692]])


def _album(*, pattern="embossed", insert="none"):
    y, x = np.ogrid[:1770, :1640]
    horizontal = np.cos(2 * np.pi * x / 13)
    vertical = np.cos(2 * np.pi * y / 22)
    embossing = 13 * horizontal * vertical + 5 * (2 * vertical ** 2 - 1)
    carrier = {"embossed": 237 + embossing, "stripes": (237 + 13 * vertical) + 0 * x,
               "plain": np.full((1770, 1640), 237)}[pattern]
    raster = np.uint8(np.round(np.broadcast_to(carrier[..., None], (1770, 1640, 3)))).copy()
    photo = raster[1184:1693, 301:1346]
    photo[:] = (171, 199, 217)
    cv2.circle(photo, (350, 280), 210, (32, 70, 98), -1)
    cv2.rectangle(photo, (640, 60), (900, 220), (104, 113, 110), -1)
    if insert == "visible":
        # A visible scene under the same lattice must stay in the active list.
        region = raster[420:785, 580:1000]
        region[:] = cv2.subtract(region, np.full_like(region, 49))
    elif insert == "near_blank":
        # A low-amplitude genuine insert can preserve the optical cue.
        region = raster[420:785, 580:1000]
        region[:] = cv2.subtract(region, np.full_like(region, 3))
    elif insert == "pale_scene":
        shadow = np.zeros(raster.shape[:2], dtype=np.uint8)
        cv2.ellipse(shadow, (810, 605), (253, 69), -28, 0, 360, 11, -1)
        cv2.circle(shadow, (1000, 748), 96, 9, -1)
        raster[:] = cv2.subtract(raster, cv2.cvtColor(shadow, cv2.COLOR_GRAY2BGR))
    elif insert == "small_detail":
        shadow = np.zeros(raster.shape[:2], dtype=np.uint8)
        cv2.ellipse(shadow, (500, 396), (60, 45), 28, 0, 360, 38, -1)
        raster[:] = cv2.subtract(raster, cv2.cvtColor(shadow, cv2.COLOR_GRAY2BGR))
    return raster


def _model_result():
    def detection(quad, confidence, contrast, cluster_id):
        normalized = quad / np.array([1639, 1769])
        x1, y1 = normalized.min(axis=0)
        x2, y2 = normalized.max(axis=0)
        records = [{"x": x, "y": y} for x, y in normalized]
        return SegmentationDetection({"x1": x1, "x2": x2, "y1": y1, "y2": y2}, {
            "source": "yolo_seg_mask", "detector_source": "yolo_onnx", "polygon": records,
            "fusion": {"proposal_families": ["model"], "proposal_count": 1, "cluster_id": cluster_id,
                       "boundary_evidence": {"border_contrast": contrast}},
        }, confidence, (x2 - x1) / (y2 - y1), True, [])

    return SegmentationResult([detection(_POCKET, .58, .42, 0), detection(_PEER, .89, .88, 1)], {
        "fallback_used": True, "fusion": {"model_proposals": 2, "classical_proposals": 1},
    })


@pytest.mark.parametrize("pattern,insert,held", [
    ("embossed", "none", True),
    ("embossed", "near_blank", True),
    ("embossed", "visible", False),
    ("embossed", "pale_scene", False),
    ("embossed", "small_detail", False),
    ("stripes", "none", False),
    ("plain", "none", False),
])
def test_only_repeated_source_phase_without_visible_scene_moves_to_count_review(pattern, insert, held):
    source = _album(pattern=pattern, insert=insert)
    baseline = _model_result()
    result = _defer_embossed_pocket_claims(source, baseline, .70)
    assert not embossed_pocket_traces(source, _PEER)
    if not held:
        assert result == baseline
        return
    assert result.detections == baseline.detections[1:]
    proposal, = result.metadata["fusion"]["deferred_print_proposals"]
    assert proposal["reason"] == EMBOSSED_POCKET_PROPOSAL
    assert proposal["source_mask"]["polygon"] == baseline.detections[0].mask["polygon"]
    assert len(proposal["embossing_edge_traces"]) == 2


async def test_album_pocket_count_acceptance_remains_separate_from_both_crop_modes(client, monkeypatch):
    from app.config import get_settings
    from app.services import pipeline

    for name, value in {
        "AUTO_ANALYZE_UPLOADS": "true", "ENABLE_PRECISE_PHOTO_CROPS": "true",
        "ENABLE_PROTECTED_REGION_GUARD": "false", "ENABLE_PHOTO_ORIENTATION_CORRECTION": "false",
        "MAX_UPLOAD_SIZE": "4000000", "MAX_REQUEST_SIZE": "5000000",
    }.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(pipeline, "run_paddle_ocr", lambda *_: [])
    baseline = _model_result()
    monkeypatch.setattr(pipeline, "segment_album_page", lambda _, __, source_image, **_kwargs:
                        _defer_embossed_pocket_claims(source_image, baseline, .70))
    get_settings.cache_clear()
    source = _album(insert="near_blank")
    ok, image = cv2.imencode(".jpg", source, [cv2.IMWRITE_JPEG_QUALITY, 75])
    assert ok
    album_id = (await client.post("/api/v1/albums", json={"name": "Embossed album pocket"})).json()["id"]
    album = f"/api/v1/albums/{album_id}"
    upload = await client.post(f"{album}/pages/upload", files={"files": ("album.jpg", image.tobytes(), "image/jpeg")})
    assert upload.status_code == 201, upload.text

    queue_url = f"/api/v1/review/queue?album_id={album_id}&type=segmentation"
    task, = (await client.get(queue_url)).json()
    initial = task["payload"]
    assert initial["detection_count"] == 1
    proposal, = initial["deferred_print_proposals"]
    assert proposal["reason"] == EMBOSSED_POCKET_PROPOSAL
    existing = (await client.get(f"{album}/photos")).json()
    manual_url = f"/api/v1/pages/{task['page_id']}/manual-photos"
    creation = {"request_id": str(uuid4()), "warp_quad": proposal["polygon"],
                "deferred_proposal_id": proposal["id"], "photo_count_revision": initial["photo_count_revision"],
                "print_size_mm": {"width": 90, "height": 70}}
    unlinked = {key: creation[key] for key in ("request_id", "warp_quad")}
    for unconfirmed in (creation, unlinked):
        denied = await client.post(manual_url, json=unconfirmed)
        assert denied.status_code == 409 and denied.json()["detail"] == "album_pocket_print_confirmation_required"
    confirmed = {**creation, "confirm_album_pocket_print": True}
    oversized_quad = [dict(point) for point in proposal["polygon"]]
    for point in oversized_quad[2:]:
        point["y"] = .96
    for changes, detail, status in [
        ({"photo_count_revision": "old"}, "deferred_print_proposal_stale", 409),
        ({"warp_quad": baseline.detections[1].mask["polygon"]}, "album_pocket_outline_outside_proposal", 422),
        ({"warp_quad": oversized_quad}, "album_pocket_outline_outside_proposal", 422),
    ]:
        denied = await client.post(manual_url, json={**confirmed, **changes})
        assert denied.status_code == status and denied.json()["detail"] == detail
    assert (await client.get(queue_url)).json() == [task]

    saved = await client.post(manual_url, json=confirmed)
    assert saved.status_code == 201, saved.text
    restored = saved.json()
    repeated = await client.post(manual_url, json=confirmed)
    assert repeated.status_code == 200 and repeated.json() == restored
    final = (await client.get(queue_url)).json()[0]["payload"]
    assert final["detection_count"] == 2 and final["deferred_print_proposals"] == []
    assert [photo for photo in (await client.get(f"{album}/photos")).json() if photo["id"] != restored["id"]] == existing
    original_url = f"/api/v1/photos/{restored['id']}"
    assert (await client.post(f"{original_url}/reprocess")).status_code == 200
    after = (await client.get(original_url)).json()
    provenance = after["enhancement_applied"]["review_provenance"]
    assert provenance["deferred_print_proposal"] == proposal
    modes = after["enhancement_applied"]["correction"]["crop_modes"]
    assert set(modes) == {"full_print", "image_only"}
    assert all(mode["status"] == "review_needed" and PROJECTIVE_PRINT_OUTLINE_REVIEW in mode["reason_codes"]
               for mode in modes.values())
    for variant in ("full_print", "image_only"):
        assert (await client.get(f"{original_url}/image?variant={variant}")).status_code == 404
    assert (await client.get(f"{original_url}/image?variant=source_corrected_untrimmed")).status_code == 200
