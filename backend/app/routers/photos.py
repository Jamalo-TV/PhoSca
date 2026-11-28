import asyncio
from collections.abc import Callable
from copy import deepcopy
from datetime import datetime, timezone
from math import hypot
from pathlib import Path
import shutil
from typing import Annotated
from uuid import UUID, uuid4, uuid5

import cv2
from anyio import CapacityLimiter, to_thread
from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from fastapi.responses import FileResponse
from PIL import Image
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings, get_settings
from app.core.auth import Principal, get_principal, require_write_principal
from app.database import get_async_session
from app.models import AuditLog, ExtractedPhoto, OCRResult, Page, PageStatus, PhotoStatus, ProcessingRun, StageRun
from app.pipeline.adjacent_stock_review import ADJACENT_STOCK_REVIEW_SOURCE
from app.pipeline.bowed_edge_review import MEASURED_PRINT_PREVIEW_SOURCES
from app.pipeline.bowed_scene_preview import BOWED_SCENE_PREVIEW
from app.pipeline.capture_fabric_review import CAPTURE_FABRIC_PROPOSAL
from app.pipeline.contact_print_review import CONTACT_PRINT_EDGE_CHECK, CONTACT_PRINT_EDGE_SCOPE
from app.pipeline.film_neighbor_edge import FILM_NEIGHBOR_EDGE
from app.pipeline.inward_scene_edge import INWARD_SCENE_EDGE_SCOPE
from app.pipeline.crop_validation import route_ready_crop_modes_to_review
from app.pipeline.crossing_prints import CONTESTED_TILE_OUTLINE, REVIEWED_CROSSING_OUTLINE, reviewed_outline_covers_traces
from app.pipeline.near_capture_review import NEAR_CAPTURE_PRINT_TRACES, near_capture_guide_conflicts
from app.pipeline.dark_album_review import DARK_ALBUM_PROPOSAL
from app.pipeline.decisions import (
    MANUAL_NEIGHBOR_OVERLAP, MOTIF_MARGIN_MARKS_REVIEW, is_reviewed_non_print, photo_is_reviewed_non_print,
    remember_ready_print_for_neighbor_review,
)
from app.pipeline.edge_trace import measure_edge_trace
from app.pipeline.embossed_album_pockets import EMBOSSED_CARRIER_PROPOSAL, EMBOSSED_POCKET_PROPOSAL
from app.pipeline.fragmented_prints import SOURCE_FRAGMENT_PROPOSAL
from app.pipeline.fusion_ownership import NESTED_EDGE_CONTOUR
from app.pipeline.geometry import normalized_polygon_overlap, normalized_vertices_to_pixels
from app.pipeline.image_ops import load_image, normalized_box_to_pixels
from app.pipeline.inward_corner_review import INWARD_CORNER_REVIEW_SOURCE
from app.pipeline.manual_paper_boundary import (
    PAPER_GRID_OVERLAP_REVIEW, backed_edge_review_reference, capture_outline_review_required, hold_overlapping_paper_claim_modes,
    original_prediction_mask, paper_curve_from_review_traces, paper_curve_review_reference,
    overlapping_paper_claims, paper_claim_groups, shared_rim_cut_is_inside, shared_rim_cut_reference,
)
from app.pipeline.merged_print_review import MERGED_PRINT_PAIR
from app.pipeline.possible_print_regions import POSSIBLE_PRINT_REGIONS
from app.pipeline.motif_boundary import PARTIAL_MOTIF_SEAMS
from app.pipeline.motif_comparison import MEASURED_MOTIF_COMPARISON, MOTIF_COMPARISON_PAPER_EDIT
from app.pipeline.faint_paper_review import (
    ADJACENT_FAINT_PAPER_STEPS, FAINT_PAPER_REVIEW_REQUIRES_CONFIRMATION, FAINT_PAPER_WITNESS_MARGIN,
)
from app.pipeline.motif_margin import HELD_PRINT_MOTIF_MARGIN_SOURCE, PAPER_STOCK_MOTIF_SOURCE
from app.pipeline.multipanel_prints import DISPUTED_GRID_PREVIEW_SOURCE, GRID_OUTLINE_REVIEW_SOURCE
from app.pipeline.neighboring_print_review import FOLLOWUP_PRINT_EDGE_GUIDE, NEIGHBORING_PRINT_EDGE_TRACES
from app.pipeline.off_print_advisory import MANUAL_MARGIN_CONTEXT
from app.pipeline.partial_print_review import PARTIAL_PRINT_EXTENT_TRACES
from app.pipeline.perspective import (
    manual_print_intersects, review_outline_intersects_claim, review_traces_intersect_claim, validate_review_quads,
)
from app.pipeline.presets import available_presets
from app.pipeline.print_edges import OPPOSING_OUTER_EDGES_CHECK
from app.pipeline.print_outline_suggestion import POSSIBLE_OUTER_PRINT_PREVIEW, POSSIBLE_OVEREXTENDED_PRINT_EDGE
from app.pipeline.projective_prints import PROJECTIVE_PRINT_PROPOSAL
from app.pipeline.protected_regions import (
    OFF_PRINT_ADVISORY, ProtectedRegionCut, ProtectedRegionRemoval, album_mark_correction_matches, album_mark_matches_quad,
    album_mark_source_matches,
)
from app.pipeline.ribbed_scene_edge import RIBBED_SCENE_EDGE, RIBBED_SCENE_EDGE_SCOPE
from app.pipeline.ribbed_outline_review import MEASURED_RIBBED_PRINT_OUTLINE
from app.pipeline.rounded_corner_review import ROUNDED_CORNER_SOURCE, ROUNDED_CORNER_WEDGES
from app.pipeline.tile_print_review import CONTESTED_TILE_FOLLOWUP, TILED_PRINT_PROPOSAL, tiled_outline_conflicts
from app.pipeline.variants import (
    CROP_REVIEW_PREVIEWS,
    FULL_PRINT_REVIEW_PREVIEW,
    MOTIF_REVIEW_PREVIEW,
    SAVED_FULL_PRINT_REVIEW,
    released_variant_from_photo,
    saved_full_print_review_asset,
    variant_path_from_photo,
    variant_urls,
    variants_from_metadata,
)
from app.pipeline.visible_scene_edge import VISIBLE_SCENE_EDGE
from app.schemas import BoundingBoxUpdate, OCRRead, OCRUpdate, PhotoDetail, PhotoRead
from app.schemas.common import PrintSizeMM
from app.schemas.pipeline import ManualPhotoCreate
from app.services.pipeline import (
    deduplication_task_logic,
    enhancement_task_logic,
    page_fusion_evidence,
    persistence_task_logic,
    perspective_correction_task_logic,
    photo_views_for_current_crop_checks,
    resolve_nested_edge_parent_reviews,
    retire_album_mark_previews,
    retire_rounded_corner_preview,
    retire_source_fragment_proposals,
)

router = APIRouter(prefix="/api/v1", tags=["photos"])
_edge_trace_decoders = CapacityLimiter(1)
_EdgeCoordinate = Annotated[float, Query(ge=0, le=1, allow_inf_nan=False)]


async def _photo_for_principal(
    session: AsyncSession, photo_id: UUID, principal: Principal | None, *, for_update: bool = False
) -> ExtractedPhoto:
    query = select(ExtractedPhoto).where(ExtractedPhoto.id == photo_id)
    if principal is not None:
        query = query.where(ExtractedPhoto.album.has(tenant_id=principal.tenant_id))
    if for_update:
        query = query.with_for_update().execution_options(populate_existing=True)
    photo = await session.scalar(query)
    if photo is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Photo not found.")
    return photo


async def _page_for_principal(
    session: AsyncSession, page_id: UUID, principal: Principal | None, *, for_update: bool = False,
) -> Page:
    query = select(Page).where(Page.id == page_id)
    if principal is not None:
        query = query.where(Page.album.has(tenant_id=principal.tenant_id))
    if for_update:
        query = query.with_for_update()
    page = await session.scalar(query)
    if page is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Page not found.")
    return page


async def _locked_review_photo(
    session: AsyncSession, photo_id: UUID, principal: Principal | None,
) -> tuple[Page, ExtractedPhoto]:
    photo = await _photo_for_principal(session, photo_id, principal)
    page = await _page_for_principal(session, photo.page_id, principal, for_update=True)
    return page, await _photo_for_principal(session, photo_id, principal, for_update=True)


def _photo_urls(photo: ExtractedPhoto | PhotoRead) -> dict[str, str | None]:
    return variant_urls(photo)


def _current_paper_claims(photos: list[ExtractedPhoto]) -> list[dict]:
    claims = []
    for photo in photos:
        mask = photo.segmentation_mask or {}
        claims.append(mask)
        correction = (photo.enhancement_applied or {}).get("correction") or {}
        full = (correction.get("crop_modes") or {}).get("full_print") or {}
        if full.get("warp_quad"):
            claims.append({"warp_quad": full["warp_quad"]})
        if not any(mask.get(key) for key in ("polygon", "visible_polygon", "warp_quad")):
            box = photo.bounding_box
            claims.append({"polygon": [{"x": box[x], "y": box[y]} for x, y in (
                ("x1", "y1"), ("x2", "y1"), ("x2", "y2"), ("x1", "y2"),
            )]})
    return claims


def _matches_review_outline(outline: list[dict], claimed: list[dict] | None) -> bool:
    return bool(claimed and normalized_polygon_overlap(
        [(point["x"], point["y"]) for point in claimed], [(point["x"], point["y"]) for point in outline], raster_size=512,
    ).iou >= .90)


def _matches_standalone_print_outline(outline: list[dict], proposed: list[dict]) -> bool:
    match = normalized_polygon_overlap(
        [(point["x"], point["y"]) for point in proposed], [(point["x"], point["y"]) for point in outline],
        raster_size=512,
    )
    return match.first_containment >= .55 and match.second_containment >= .75


def _matches_merged_print_claim(outline: list[dict], mask: dict) -> bool:
    if MERGED_PRINT_PAIR not in mask:
        return False
    proposed = (mask.get(MERGED_PRINT_PAIR) or {}).get("outlines") or []
    parent = mask.get("polygon")
    if any(_matches_standalone_print_outline(outline, part) for part in proposed):
        return True
    return bool(parent and _matches_standalone_print_outline(outline, parent))


def _resolve_merged_print_outline(mask: dict, payload: BoundingBoxUpdate, edited: dict, width: int, height: int) -> bool:
    """Check that the edit resolves the current ownership question."""
    parts = (mask.get(MERGED_PRINT_PAIR) or {}).get("outlines") or []
    entered = edited.get("warp_quad")
    if (len(parts) != 2 or payload.segmentation_mask is None or payload.segmentation_mask.warp_quad is None
            or entered is None or edited.get("image_quad") is not None):
        return False
    points = [(point["x"], point["y"]) for point in entered]
    paper = normalized_vertices_to_pixels(points, width, height)
    candidates = [[(point["x"], point["y"]) for point in part] for part in parts]

    def encloses(outline: list[tuple[float, float]]) -> bool:
        corners = normalized_vertices_to_pixels(outline, width, height)
        return all(cv2.pointPolygonTest(paper, tuple(corner), True) >= -.05 for corner in corners)

    if payload.confirm_merged_print_as_one:
        parent = mask.get("polygon")
        return bool(parent and _matches_standalone_print_outline(entered, parent) and all(map(encloses, candidates)))
    overlaps = [normalized_polygon_overlap(part, points, raster_size=512) for part in candidates]
    return any(
        overlap.second_containment >= .75 and other.first_containment <= .02 and encloses(candidate)
        for candidate, overlap, other in zip(candidates, overlaps, reversed(overlaps), strict=True)
    )


def _matches_contested_tile(photo: ExtractedPhoto, outline: list[dict]) -> bool:
    mask = photo.segmentation_mask or {}
    if mask.get("crossing_print_suggestion_source") == CONTESTED_TILE_OUTLINE:
        claimed = mask.get("crossing_print_suggestion")
    elif (mask.get(REVIEWED_CROSSING_OUTLINE) is True
          and (original_prediction_mask(photo.enhancement_applied or {}) or {}).get("crossing_print_suggestion_source")
          == CONTESTED_TILE_OUTLINE):
        claimed = mask.get("warp_quad")
    else:
        return False
    return _matches_review_outline(outline, claimed)


def _offer_contested_tile_followup(
    page: Page, previous: dict, photo: ExtractedPhoto, neighbors: list[ExtractedPhoto], width: int, height: int,
) -> None:
    outline = previous.get(CONTESTED_TILE_FOLLOWUP)
    edited = photo.segmentation_mask or {}
    if (not outline or previous.get("crossing_print_suggestion_source") != CONTESTED_TILE_OUTLINE
            or edited.get(REVIEWED_CROSSING_OUTLINE) is not True or not edited.get("warp_quad")
            or not _same_visible_outline(previous.get("crossing_print_suggestion"), edited["warp_quad"],
                                         native_size=(width, height))):
        return
    pending = page_fusion_evidence(page.processing_metadata).get("deferred_print_proposals") or []
    claims = [*_current_paper_claims([photo, *neighbors]), *pending]
    for neighbor in neighbors:
        mask = neighbor.segmentation_mask or {}
        if alternative := mask.get("crossing_print_suggestion"):
            claims.append({"polygon": alternative})
        claims.extend({"polygon": alternative} for alternative in (mask.get(MERGED_PRINT_PAIR) or {}).get("outlines", ()))
    if tiled_outline_conflicts(outline, claims, width, height):
        return
    next_id = max((candidate["id"] for candidate in pending), default=-1) + 1
    metadata = deepcopy(page.processing_metadata or {})
    segmentation = metadata.setdefault("steps", {}).setdefault("segmentation", {})
    segmentation.setdefault("fusion", {})["deferred_print_proposals"] = [
        *pending, {"id": next_id, "reason": TILED_PRINT_PROPOSAL, "source": CONTESTED_TILE_FOLLOWUP, "polygon": deepcopy(outline)},
    ]
    guide = previous.get(FOLLOWUP_PRINT_EDGE_GUIDE) or {}
    if guide.get("traces") and guide.get("witnesses") and not segmentation.get(FOLLOWUP_PRINT_EDGE_GUIDE) and not any(
        tiled_outline_conflicts(witness, [*claims, {"polygon": outline}], width, height)
        for witness in guide["witnesses"]
    ):
        segmentation[FOLLOWUP_PRINT_EDGE_GUIDE] = deepcopy(guide)
    page.processing_metadata = {**metadata, "photo_count_revision": str(uuid4())}


def _source_size(page: Page) -> tuple[int, int]:
    try:
        with Image.open(page.storage_path) as source:
            width, height = source.size
            return (height, width) if source.getexif().get(274) in {5, 6, 7, 8} else (width, height)
    except OSError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Photo source image could not be opened.") from exc


def _rect_mask_from_box(box: dict[str, float], width: int, height: int) -> dict:
    x1, y1, x2, y2 = normalized_box_to_pixels(box, width, height)
    if x2 <= x1 or y2 <= y1:
        raise ValueError("Bounding box does not enclose any source pixels.")
    polygon = [
        {"x": x / max(width - 1, 1), "y": y / max(height - 1, 1)}
        for x, y in [(x1, y1), (x2 - 1, y1), (x2 - 1, y2 - 1), (x1, y2 - 1)]
    ]
    return {
        "polygon": polygon,
        "visible_polygon": polygon,
        "warp_quad": polygon,
        "source": "manual_rect",
    }


def _align_measured_quad(previous: list[dict], updated: list[dict], width: int, height: int) -> list[dict]:
    """Match a legacy edit's corners to the previously measured edges."""
    orders = (cycle[index:] + cycle[:index] for cycle in (updated, updated[::-1]) for index in range(4))
    return min(orders, key=lambda points: sum(
        (width * (old["x"] - new["x"])) ** 2 + (height * (old["y"] - new["y"])) ** 2
        for old, new in zip(previous, points, strict=True)
    ))


async def _prediction_provenance(session: AsyncSession, photo: ExtractedPhoto) -> dict:
    page = await session.get(Page, photo.page_id)
    if page is None:
        return {}
    run = await session.scalar(
        select(ProcessingRun).where(ProcessingRun.page_id == page.id).order_by(ProcessingRun.created_at.desc()).limit(1)
    )
    stage = None
    if run is not None:
        stage = await session.scalar(
            select(StageRun)
            .where(StageRun.processing_run_id == run.id, StageRun.stage == "segmentation")
            .order_by(StageRun.attempt.desc())
            .limit(1)
        )
    return {
        "page_id": str(page.id),
        "page_sha256": page.file_hash_sha256,
        "run_id": str(run.id) if run else None,
        "pipeline_version": run.pipeline_version if run else None,
        "config_fingerprint": run.config_fingerprint if run else None,
        "segmentation_stage_id": str(stage.id) if stage else None,
        "segmentation_stage_attempt": stage.attempt if stage else None,
        "segmentation_stage_details": deepcopy(stage.details or {}) if stage else {},
        "segmentation": deepcopy((page.processing_metadata or {}).get("steps", {}).get("segmentation", {})),
    }


def _geometry_snapshot(photo: ExtractedPhoto) -> dict:
    return {
        "bounding_box": deepcopy(photo.bounding_box),
        "segmentation_mask": deepcopy(photo.segmentation_mask),
        "segmentation_confidence": photo.segmentation_confidence,
        "aspect_ratio": photo.aspect_ratio,
        "geometry_valid": photo.geometry_valid,
        "status": photo.status.value,
    }


def _claim_missing_print_trace(
    page: Page, review_provenance: dict, quad: list[dict] | None, width: int, height: int,
    *, polygon: list[dict] | None = None,
) -> None:
    outline = quad or polygon
    if len(outline or []) < 3:
        return
    metadata = deepcopy(page.processing_metadata or {})
    segmentation = (metadata.get("steps") or {}).get("segmentation") or {}
    traces = segmentation.get("missing_print_edge_traces", [])
    claims_curve = len(quad or []) == 4 and reviewed_outline_covers_traces(quad, traces)
    if claims_curve:
        curve = paper_curve_from_review_traces(quad, traces, width, height)
        if curve is not None:
            review_provenance["paper_trace_review"] = curve
        segmentation.pop("missing_print_edge_traces")
    changed = claims_curve
    for key in (NEAR_CAPTURE_PRINT_TRACES, NEIGHBORING_PRINT_EDGE_TRACES):
        open_edges = segmentation.get(key, [])
        if reviewed_outline_covers_traces(outline, open_edges) or near_capture_guide_conflicts(outline, open_edges, width, height):
            segmentation.pop(key)
            changed = True
    guide = segmentation.get(FOLLOWUP_PRINT_EDGE_GUIDE) or {}
    if guide and (
        reviewed_outline_covers_traces(outline, guide.get("traces", []))
        or any(tiled_outline_conflicts(witness, [{"polygon": outline}], width, height) for witness in guide.get("witnesses", []))
    ):
        segmentation.pop(FOLLOWUP_PRINT_EDGE_GUIDE)
        changed = True
    if changed:
        page.processing_metadata = metadata


async def _render_photo_edit(
    session: AsyncSession, photo_id: UUID, settings: Settings, *,
    reject_invalid_edit: bool = True, preset: str | None = "auto",
    previous_manual_crop: dict | None = None,
    explain_protected_cut: bool = False,
    after_render: Callable[[], None] | None = None,
) -> ExtractedPhoto:
    output_directories: set[Path] = set()
    try:
        try:
            correction = await perspective_correction_task_logic(
                session, photo_id, settings, reject_invalid_edit=reject_invalid_edit,
                commit=False, output_directories=output_directories,
                **({"previous_manual_crop": previous_manual_crop} if previous_manual_crop is not None else {}),
            )
        except ProtectedRegionRemoval as exc:
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=exc.detail) from exc
        except ProtectedRegionCut as exc:
            if not reject_invalid_edit:
                raise
            detail = exc.detail if explain_protected_cut else str(exc)
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=detail) from exc
        except ValueError as exc:
            if not reject_invalid_edit:
                raise
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
        if not correction.get("review_required"):
            await enhancement_task_logic(
                session, photo_id, settings, preset=preset, commit=False, output_directories=output_directories,
            )
            await deduplication_task_logic(session, photo_id, settings, commit=False)
            await persistence_task_logic(session, photo_id, settings, commit=False)
        if after_render is not None:
            after_render()
    except BaseException:
        await session.rollback()
        for directory in output_directories:
            await asyncio.to_thread(shutil.rmtree, directory, ignore_errors=True)
        raise
    await session.commit()
    photo = await session.get(ExtractedPhoto, photo_id, populate_existing=True)
    if photo is None:
        raise ValueError(f"Photo not found after correction: {photo_id}")
    return photo


async def _review_overlapping_crops(
    session: AsyncSession, photo: ExtractedPhoto, width: int, height: int, *, neighbors: list[ExtractedPhoto] | None = None,
) -> None:
    mask = photo.segmentation_mask or {}
    quad = mask.get("warp_quad")
    groups = paper_claim_groups(mask, original_prediction_mask(photo.enhancement_applied or {}))
    if quad is None and not groups and not (mask.get("visible_polygon") or mask.get("polygon")):
        return
    if neighbors is None:
        neighbors = (await session.scalars(
            select(ExtractedPhoto)
            .where(ExtractedPhoto.page_id == photo.page_id, ExtractedPhoto.id != photo.id)
            .with_for_update()
            .execution_options(populate_existing=True)
        )).all()
    for neighbor in neighbors:
        metadata = deepcopy(neighbor.enhancement_applied or {})
        if is_reviewed_non_print(metadata.get("review_rejected_detection"), neighbor.bounding_box, neighbor.segmentation_mask):
            continue
        correction = metadata.get("correction") or {}
        pair = (neighbor.segmentation_mask or {}).get(MERGED_PRINT_PAIR) or {}
        separator = pair.get("separator")
        if (any(review_outline_intersects_claim(outline, mask, width, height) for outline in pair.get("outlines", []))
                or (separator and review_traces_intersect_claim([separator], mask, width, height))):
            neighbor.segmentation_mask = {**neighbor.segmentation_mask, MERGED_PRINT_PAIR: None}
        possible = (neighbor.segmentation_mask or {}).get(POSSIBLE_PRINT_REGIONS) or {}
        if any(tiled_outline_conflicts(outline, [mask], width, height) for outline in possible.get("outlines", [])):
            neighbor.segmentation_mask = {**neighbor.segmentation_mask, POSSIBLE_PRINT_REGIONS: None}
        modes = correction.get("crop_modes") or {}
        full_mode = modes.get("full_print") or {}
        film_edge = full_mode.get(FILM_NEIGHBOR_EDGE)
        if film_edge and any(review_outline_intersects_claim(film_edge[key], mask, width, height)
                             for key in ("source_scope", "carrier_outline")):
            full_mode.pop(FILM_NEIGHBOR_EDGE)
            neighbor.enhancement_applied = metadata
        if (full_mode.get("suggested_edge_source") == VISIBLE_SCENE_EDGE and full_mode.get("warp_quad")
                and review_outline_intersects_claim(full_mode["warp_quad"], mask, width, height)):
            full_mode.pop("suggested_edge_source", None)
            full_mode.pop("suggested_edge_traces", None)
            neighbor.enhancement_applied = metadata
        if full_mode.get("suggested_edge_source") == ADJACENT_FAINT_PAPER_STEPS and review_traces_intersect_claim(
            full_mode.get("suggested_edge_traces") or [], mask, width, height, margin_px=FAINT_PAPER_WITNESS_MARGIN,
        ):
            full_mode.pop("suggested_edge_source", None)
            full_mode.pop("suggested_edge_traces", None)
            neighbor.enhancement_applied = metadata
        if full_mode.get("suggested_edge_source") == RIBBED_SCENE_EDGE and any(
            tiled_outline_conflicts(scope, [mask], width, height) for scope in full_mode.get(RIBBED_SCENE_EDGE_SCOPE, [])
        ):
            full_mode.pop(RIBBED_SCENE_EDGE_SCOPE)
            full_mode.pop("suggested_edge_source")
            full_mode.pop("suggested_edge_traces", None)
            neighbor.enhancement_applied = metadata
        inward_scope = full_mode.get(INWARD_SCENE_EDGE_SCOPE)
        if inward_scope and any(
            review_outline_intersects_claim(inward_scope, {"polygon": mask.get(key)}, width, height, convex_scope=True)
            for key in ("polygon", "visible_polygon", "warp_quad")
        ):
            full_mode.pop(INWARD_SCENE_EDGE_SCOPE, None)
            full_mode.pop("suggested_edge_source", None)
            full_mode.pop("suggested_edge_traces", None)
            neighbor.enhancement_applied = metadata
        preview = full_mode.get("outline_preview") or {}
        preview_source = preview.get("source")
        strip = full_mode.get("source_strip_review") or {}
        strip_scope = strip.get("evidence_scope") or strip.get("source_outline")
        if strip_scope and review_outline_intersects_claim(strip_scope, mask, width, height):
            full_mode.pop("source_strip_review", None)
            neighbor.enhancement_applied = metadata
        extent_traces = full_mode.get(PARTIAL_PRINT_EXTENT_TRACES)
        contact_scope = full_mode.get(CONTACT_PRINT_EDGE_SCOPE)
        if extent_traces and (review_traces_intersect_claim(extent_traces, mask, width, height)
                              or (contact_scope and review_outline_intersects_claim(contact_scope, mask, width, height))):
            full_mode.pop(PARTIAL_PRINT_EXTENT_TRACES, None)
            full_mode.pop(CONTACT_PRINT_EDGE_SCOPE, None)
            neighbor.enhancement_applied = metadata
        mark_offer = correction.get("album_page_mark_review") or {}
        reviewed_mark = (metadata.get("review_provenance") or {}).get("reviewed_album_mark") or {}
        for review in (mark_offer, reviewed_mark):
            if review.get("source") != OFF_PRINT_ADVISORY or not any(
                review_outline_intersects_claim(mark["claim_outline"], mask, width, height)
                for mark in review["regions"]
            ):
                continue
            if review is mark_offer:
                correction.pop("album_page_mark_review", None)
                correction.pop("off_print_advisory_regions", None)
            else:
                reason = "album_page_mark_review_stale"
                route_ready_crop_modes_to_review(modes, [reason])
                (correction.get("protected_regions") or {}).pop("reviewed_album_mark_id", None)
                retire_album_mark_previews(correction)
                metadata["review_reasons"] = list(dict.fromkeys([reason, *(metadata.get("review_reasons") or [])]))
                neighbor.status = PhotoStatus.review_needed
            neighbor.enhancement_applied = metadata
        if full_mode.get("suggested_edge_source") == ROUNDED_CORNER_SOURCE and any(
            review_outline_intersects_claim(scope, mask, width, height) for scope in full_mode.get(ROUNDED_CORNER_WEDGES, [])
        ):
            retire_rounded_corner_preview(correction)
            neighbor.enhancement_applied = metadata
        assisted_previews = (GRID_OUTLINE_REVIEW_SOURCE, POSSIBLE_OVEREXTENDED_PRINT_EDGE, INWARD_CORNER_REVIEW_SOURCE)
        measured_preview = preview_source in MEASURED_PRINT_PREVIEW_SOURCES
        fields = ("source_search_band", "source_inset_band") if measured_preview else ("source_outline",)
        evidence_scope = preview.get("source_search_band") if preview_source in (
            MEASURED_RIBBED_PRINT_OUTLINE, BOWED_SCENE_PREVIEW,
        ) else None
        if ((evidence_scope and review_outline_intersects_claim(evidence_scope, mask, width, height, convex_scope=True))
                or (preview_source in (DISPUTED_GRID_PREVIEW_SOURCE, POSSIBLE_OUTER_PRINT_PREVIEW) and preview.get("source_outline")
             and review_outline_intersects_claim(preview["source_outline"], mask, width, height))
                or (quad is not None and (preview_source in assisted_previews or measured_preview) and any(
                    manual_print_intersects(preview[key], quad, width, height, minimum_overlap_px=1. if measured_preview else None)
                    for key in fields if preview.get(key)
                ))):
            full_mode.pop("outline_preview")
            correction.pop("album_page_mark_review", None)
            if full_mode.get("suggested_edge_source") in (
                POSSIBLE_OVEREXTENDED_PRINT_EDGE, BOWED_SCENE_PREVIEW, *MEASURED_PRINT_PREVIEW_SOURCES,
            ):
                full_mode.pop("suggested_edge_source", None)
                full_mode.pop("suggested_edge_traces", None)
            if full_mode.get("suggested_outline_source") == INWARD_CORNER_REVIEW_SOURCE:
                full_mode.pop("suggested_warp_quad", None)
                full_mode.pop("suggested_outline_source", None)
            correction.pop(f"{FULL_PRINT_REVIEW_PREVIEW}_path", None)
            correction.pop(f"{FULL_PRINT_REVIEW_PREVIEW}_asset_id", None)
            neighbor.enhancement_applied = metadata
            session.add(AuditLog(
                entity_type="photo", entity_id=neighbor.id, action="overlapping_crop_preview_invalidated",
                details={"overlapping_photo_id": str(photo.id)},
            ))
        motif_mode = modes.get("image_only") or {}
        if motif_mode.get("suggested_seam_source") == PARTIAL_MOTIF_SEAMS and review_traces_intersect_claim(
            motif_mode.get("suggested_seam_traces") or [], mask, width, height,
        ):
            motif_mode.pop("suggested_seam_traces", None)
            motif_mode.pop("suggested_seam_source", None)
            neighbor.enhancement_applied = metadata
        motif_preview = motif_mode.get("outline_preview") or {}
        motif_outline = motif_preview.get("source_outline")
        if (motif_preview.get("source") in {PAPER_STOCK_MOTIF_SOURCE, MEASURED_MOTIF_COMPARISON} and motif_outline
                and review_outline_intersects_claim(motif_outline, mask, width, height)):
            for key in ("outline_preview", "warp_quad", "page_to_output", "shape", "sampling"):
                motif_mode.pop(key, None)
            if motif_preview["source"] == MEASURED_MOTIF_COMPARISON:
                for key in ("suggested_warp_quad", "suggested_seam_traces", "suggested_seam_source"):
                    motif_mode.pop(key, None)
                full_mode.pop("motif_paper_edge_traces", None)
            correction.pop(f"{MOTIF_REVIEW_PREVIEW}_path", None)
            correction.pop(f"{MOTIF_REVIEW_PREVIEW}_asset_id", None)
            neighbor.enhancement_applied = metadata
            session.add(AuditLog(
                entity_type="photo", entity_id=neighbor.id, action="overlapping_crop_preview_invalidated",
                details={"overlapping_photo_id": str(photo.id), "mode": "image_only"},
            ))
        elif full_mode.get("motif_paper_edge_traces") and review_traces_intersect_claim(
            full_mode["motif_paper_edge_traces"], mask, width, height,
        ):
            full_mode.pop("motif_paper_edge_traces")
            neighbor.enhancement_applied = metadata
        if motif_mode.get("margin_mark_source") == HELD_PRINT_MOTIF_MARGIN_SOURCE:
            marks = motif_mode.get("margin_mark_regions") or []
            current_marks = [mark for mark in marks if not review_outline_intersects_claim(mark, mask, width, height)]
            if len(current_marks) < len(marks):
                if current_marks:
                    motif_mode["margin_mark_regions"] = current_marks
                else:
                    motif_mode.pop("margin_mark_regions", None)
                    motif_mode.pop("margin_mark_source", None)
                    motif_mode["reason_codes"] = [
                        reason for reason in motif_mode.get("reason_codes") or [] if reason != MOTIF_MARGIN_MARKS_REVIEW
                    ]
                    metadata["review_reasons"] = [
                        reason for reason in metadata.get("review_reasons") or [] if reason != MOTIF_MARGIN_MARKS_REVIEW
                    ]
                neighbor.enhancement_applied = metadata
        if groups and any(mode.get("status") == "ready" for mode in modes.values()):
            neighbor_mask = neighbor.segmentation_mask or {}
            neighbor_groups = paper_claim_groups(neighbor_mask, original_prediction_mask(metadata))
            conflicts = overlapping_paper_claims(mask, groups, neighbor_mask, neighbor_groups, width, height)
            if neighbor.segmentation_confidence is not None:
                conflicts.discard(PAPER_GRID_OVERLAP_REVIEW)
            if conflicts:
                hold_overlapping_paper_claim_modes(modes, conflicts)
                reasons = list(dict.fromkeys([*sorted(conflicts), *(metadata.get("review_reasons") or [])]))
                neighbor.enhancement_applied = {**metadata, "review_reasons": reasons}
                neighbor.status = PhotoStatus.review_needed
                session.add(AuditLog(entity_type="photo", entity_id=neighbor.id, action="crop_review_reopened",
                                     details={"reason": min(conflicts), "overlapping_photo_id": str(photo.id)}))
                continue
        if quad is None:
            continue
        if not any(mode.get("status") == "ready" for mode in modes.values()):
            continue
        neighbor_quad = full_mode.get("warp_quad")
        if not manual_print_intersects(neighbor_quad, quad, width, height):
            continue
        if neighbor.segmentation_confidence is None and full_mode.get("status") == "ready":
            remember_ready_print_for_neighbor_review(metadata, neighbor_quad)
        reasons = ["PRINT_POSSIBLY_OCCLUDED", MANUAL_NEIGHBOR_OVERLAP]
        route_ready_crop_modes_to_review(modes, reasons)
        full_mode["neighbor_overlap_outline"] = [*quad, quad[0]]
        metadata["review_reasons"] = list(dict.fromkeys([*(metadata.get("review_reasons") or []), *reasons]))
        neighbor.enhancement_applied = metadata
        neighbor.status = PhotoStatus.review_needed
        session.add(AuditLog(
            entity_type="photo", entity_id=neighbor.id, action="crop_review_reopened",
                             details={"reason": MANUAL_NEIGHBOR_OVERLAP, "overlapping_photo_id": str(photo.id)},
        ))


def _replay_manual_photo(
    photo: ExtractedPhoto, page_id: UUID, creation_request: dict, response: Response,
) -> ExtractedPhoto:
    provenance = (photo.enhancement_applied or {}).get("review_provenance") or {}
    creation = provenance.get("manual_creation") or {}
    if bool(creation.get("replace_mounted_fragments")) != bool(creation_request.get("replace_mounted_fragments")):
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="mounted_print_fragments_stale")
    if bool(creation.get("confirm_dark_backing")) != bool(creation_request.get("confirm_dark_backing")):
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="dark_backing_confirmation_required")
    if bool(creation.get("confirm_album_pocket_print")) != bool(creation_request.get("confirm_album_pocket_print")):
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="album_pocket_print_confirmation_required")
    if bool(creation.get("confirm_standalone_print")) != bool(creation_request.get("confirm_standalone_print")):
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="standalone_print_confirmation_required")
    if photo.page_id != page_id or any(creation.get(key) != value for key, value in creation_request.items()):
        if creation_request["adjacent_suggestion"] is not None:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="adjacent_print_suggestion_saved")
        if creation_request["deferred_proposal_id"] is not None:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="deferred_print_proposal_saved")
        if creation.get("print_size_mm") != creation_request["print_size_mm"]:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="This request already saved different measurements.")
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="This request already saved different corners.")
    response.status_code = status.HTTP_200_OK
    return photo


def _same_visible_outline(
    points: object, quad: list[dict], *, native_size: tuple[int, int] | None = None,
) -> bool:
    if not isinstance(points, list) or len(points) != 4:
        return False
    for cycle in (quad, quad[::-1]):
        for index in range(4):
            ordered = cycle[index:] + cycle[:index]
            if points == ordered:
                return True
            if native_size is not None and all(
                hypot((p["x"] - q["x"]) * (native_size[0] - 1), (p["y"] - q["y"]) * (native_size[1] - 1)) <= 2
                for p, q in zip(points, ordered, strict=True)
            ):
                return True
    return False


def _source_fragment_replacements(deferred: dict, quad: list[dict], active: list[ExtractedPhoto]) -> list[ExtractedPhoto]:
    def points(polygon):
        return [(point["x"], point["y"]) for point in polygon]

    polygons = deferred.get("fragment_polygons") or []
    if len(polygons) != 2 or polygons[0] == polygons[1]:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="source_print_fragments_stale")
    replacements = []
    for polygon in polygons:
        candidates = [photo for photo in active if photo.segmentation_confidence is not None
                      and (photo.segmentation_mask or {}).get("source_fragment_group_review") == deferred["id"]
                      and photo.segmentation_mask.get("polygon") == polygon
                      and not variant_path_from_photo(photo, "full_print")
                      and not variant_path_from_photo(photo, "image_only")]
        if len(candidates) != 1:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="source_print_fragments_stale")
        replacements.append(candidates[0])
    edited = points(quad)
    measured = points(deferred["polygon"])
    if (normalized_polygon_overlap(measured, edited, raster_size=1024).iou < .96
            or any(normalized_polygon_overlap(points(part), edited, raster_size=1024).first_containment < minimum
                   for part, minimum in zip(polygons, (.97, .84), strict=True))):
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="source_print_outline_excludes_fragments")
    for photo in active:
        if photo in replacements:
            continue
        mask = photo.segmentation_mask or {}
        polygon = mask.get("visible_polygon") or mask.get("polygon") or mask.get("warp_quad")
        if not polygon:
            continue
        overlap = normalized_polygon_overlap(edited, points(polygon), raster_size=1024)
        if overlap.first_containment >= .002 and overlap.second_containment >= .02:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="source_print_fragments_stale")
    return replacements


@router.get("/presets")
async def get_presets() -> dict[str, dict]:
    return available_presets()


@router.get("/photos/{photo_id}", response_model=PhotoDetail)
async def get_photo(
    photo_id: UUID,
    session: AsyncSession = Depends(get_async_session),
    principal: Principal | None = Depends(get_principal),
) -> PhotoDetail:
    photo = await _photo_for_principal(session, photo_id, principal)
    await session.refresh(photo, attribute_names=["ocr_results"])
    view, = await photo_views_for_current_crop_checks(session, [photo])
    data = PhotoDetail.model_validate(view)
    data.ocr_results = [OCRRead.model_validate(result) for result in photo.ocr_results]
    data.urls = _photo_urls(view)
    return data


@router.get("/photos/{photo_id}/image")
async def get_photo_image(
    photo_id: UUID,
    variant: str | None = Query(default=None, pattern="^[a-zA-Z0-9_-]+$"),
    v: str | None = Query(default=None),
    session: AsyncSession = Depends(get_async_session),
    principal: Principal | None = Depends(get_principal),
) -> FileResponse:
    photo = await _photo_for_principal(session, photo_id, principal)
    rejected = photo_is_reviewed_non_print(photo)
    if rejected and variant is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Image file not found.")
    selected_variant = variant or released_variant_from_photo(photo)
    metadata = photo.enhancement_applied or {}
    mark = (metadata.get("review_provenance") or {}).get("reviewed_album_mark")
    correction = metadata.get("correction") or {}
    mode = (correction.get("crop_modes") or {}).get("full_print") or {}
    rounded_preview = selected_variant == FULL_PRINT_REVIEW_PREVIEW and mode.get("suggested_edge_source") == ROUNDED_CORNER_SOURCE
    saved_preview = selected_variant == SAVED_FULL_PRINT_REVIEW
    if mark and (selected_variant in {"full_print", "image_only", "source_corrected_trimmed"} or rounded_preview or saved_preview):
        page = await session.get(Page, photo.page_id)
        if (page is None or not get_settings().enable_protected_region_guard
                or not album_mark_correction_matches(mark, photo.segmentation_mask or {}, correction)
                or not album_mark_source_matches(mark, page.file_hash_sha256, page.processing_metadata)):
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Image file not found.")
    if saved_preview:
        asset = saved_full_print_review_asset(photo)
        selected_path = asset[1] if asset and asset[0] == v else None
    else:
        selected_path = variant_path_from_photo(photo, selected_variant)
    if not selected_path or not Path(selected_path).exists():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Image file not found.")
    preview = selected_variant in CROP_REVIEW_PREVIEWS
    filename = Path(selected_path).name
    if selected_variant in {"full_print", "image_only"}:
        filename = f"photo-{photo.id}-{selected_variant}.jpg"
    return FileResponse(
        selected_path,
        media_type="image/png" if preview else "image/jpeg",
        filename=filename,
        content_disposition_type="inline",
        headers={"Cache-Control": "private, no-store"} if preview or saved_preview or rejected else None,
    )


@router.get("/photos/{photo_id}/variants")
async def get_photo_variants(
    photo_id: UUID,
    session: AsyncSession = Depends(get_async_session),
    principal: Principal | None = Depends(get_principal),
) -> dict:
    photo = await _photo_for_principal(session, photo_id, principal)
    view, = await photo_views_for_current_crop_checks(session, [photo])
    return {
        "selected_variant": released_variant_from_photo(view),
        "variants": variants_from_metadata(view.enhancement_applied),
        "urls": _photo_urls(view),
    }


@router.get("/pages/{page_id}/image")
async def get_page_image(
    page_id: UUID,
    session: AsyncSession = Depends(get_async_session),
    principal: Principal | None = Depends(get_principal),
) -> FileResponse:
    page = await _page_for_principal(session, page_id, principal)
    path = Path(page.storage_path)
    if not path.exists():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Image file not found.")
    return FileResponse(
        path,
        media_type="application/octet-stream",
        filename=path.name,
        content_disposition_type="attachment",
    )


def _measure_page_edge(path: Path, anchors: list[tuple[float, float]], pixel_limit: int) -> list:
    try:
        with Image.open(path) as header:
            if header.width * header.height > pixel_limit:
                raise HTTPException(status_code=413, detail="edge_trace_image_too_large")
        image = load_image(path)
    except Image.DecompressionBombError as exc:
        raise HTTPException(status_code=413, detail="edge_trace_image_too_large") from exc
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=404, detail="edge_trace_source_unavailable") from exc
    try:
        return measure_edge_trace(image, anchors)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.get("/pages/{page_id}/edge-trace")
async def get_page_edge_trace(
    page_id: UUID, response: Response,
    start_x: _EdgeCoordinate, start_y: _EdgeCoordinate,
    middle_x: _EdgeCoordinate, middle_y: _EdgeCoordinate,
    end_x: _EdgeCoordinate, end_y: _EdgeCoordinate,
    session: AsyncSession = Depends(get_async_session),
    settings: Settings = Depends(get_settings),
    principal: Principal | None = Depends(get_principal),
) -> dict:
    page = await _page_for_principal(session, page_id, principal)
    source = Path(page.storage_path)
    # Return the database connection before waiting for a decoder.
    await session.rollback()
    response.headers["Cache-Control"] = "private, no-store"
    anchors = [(start_x, start_y), (middle_x, middle_y), (end_x, end_y)]
    traces = await to_thread.run_sync(
        _measure_page_edge, source, anchors, settings.max_image_pixels,
        limiter=_edge_trace_decoders,
    )
    return {"traces": traces}


@router.get("/pages/{page_id}/ocr", response_model=list[OCRRead])
async def get_page_ocr(
    page_id: UUID,
    session: AsyncSession = Depends(get_async_session),
    principal: Principal | None = Depends(get_principal),
) -> list[OCRResult]:
    await _page_for_principal(session, page_id, principal)
    return (
        await session.scalars(
            select(OCRResult).where(OCRResult.page_id == page_id).order_by(OCRResult.created_at.asc())
        )
    ).all()


@router.post("/pages/{page_id}/manual-photos", response_model=PhotoRead, status_code=status.HTTP_201_CREATED)
async def create_manual_photo(
    page_id: UUID,
    payload: ManualPhotoCreate,
    response: Response,
    explain_protected_cut: bool = False,
    session: AsyncSession = Depends(get_async_session),
    settings: Settings = Depends(get_settings),
    principal: Principal | None = Depends(require_write_principal),
) -> ExtractedPhoto:
    page = await _page_for_principal(session, page_id, principal, for_update=True)
    quad = [point.model_dump() for point in payload.warp_quad]
    size = payload.print_size_mm
    suggestion = payload.adjacent_suggestion
    deferred_id = payload.deferred_proposal_id
    creation_request = {
        "request_id": str(payload.request_id),
        "warp_quad": quad,
        "print_size_mm": size.model_dump() if size else None,
        "adjacent_suggestion": suggestion.model_dump(mode="json", exclude_none=True) if suggestion else None,
        "deferred_proposal_id": deferred_id,
        "photo_count_revision": payload.photo_count_revision,
    }
    if payload.replace_mounted_fragments:
        creation_request["replace_mounted_fragments"] = True
    if payload.confirm_dark_backing:
        creation_request["confirm_dark_backing"] = True
    if payload.confirm_album_pocket_print:
        creation_request["confirm_album_pocket_print"] = True
    if payload.confirm_standalone_print:
        creation_request["confirm_standalone_print"] = True
    # Two sessions can edit the same outline with different request UUIDs.
    # Its source and displayed coordinates identify a single addition.
    identity = str(payload.request_id)
    if suggestion is not None:
        identity = f"adjacent-print:{suggestion.model_dump_json(exclude_none=True)}"
    elif deferred_id is not None:
        identity = f"deferred-print:{payload.photo_count_revision}:{deferred_id}"
    photo_id = uuid5(page.id, identity)
    existing = await session.get(ExtractedPhoto, photo_id)
    if existing is None:
        previous_request = ExtractedPhoto.enhancement_applied["review_provenance"]["manual_creation"]["request_id"]
        existing = await session.scalar(select(ExtractedPhoto).where(
            ExtractedPhoto.page_id == page_id, previous_request.as_string() == creation_request["request_id"],
        ))
    if existing is not None:
        return _replay_manual_photo(existing, page_id, creation_request, response)
    if page.status not in {PageStatus.completed, PageStatus.review_needed}:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Wait for page analysis before adding a print.")
    if not settings.enable_precise_photo_crops:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Manual additions require precise crop modes.")
    deferred = None
    page_metadata = page.processing_metadata or {}
    pending = page_fusion_evidence(page_metadata).get("deferred_print_proposals") or []
    if deferred_id is not None:
        if payload.photo_count_revision is None or payload.photo_count_revision != page_metadata.get("photo_count_revision"):
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="deferred_print_proposal_stale")
        deferred = next((candidate for candidate in pending if candidate["id"] == deferred_id), None)
        if deferred is None:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="deferred_print_proposal_stale")
    width, height = _source_size(page)
    mask = {"source": "manual_added", "manual_added": True, "polygon": quad, "visible_polygon": quad, "warp_quad": quad}
    if (deferred or {}).get("reason") == PROJECTIVE_PRINT_PROPOSAL:
        mask["projective_outline_review"] = True
    nested_edge_offer = (deferred or {}).get("reason") == NESTED_EDGE_CONTOUR
    tiled_offer = (deferred or {}).get("reason") == TILED_PRINT_PROPOSAL
    fabric_offer = (deferred or {}).get("reason") == CAPTURE_FABRIC_PROPOSAL
    standalone_offer = nested_edge_offer or tiled_offer or fabric_offer
    if payload.confirm_standalone_print and not standalone_offer:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="standalone_print_confirmation_stale")
    if standalone_offer:
        if not payload.confirm_standalone_print:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="standalone_print_confirmation_required")
        if not _matches_standalone_print_outline(quad, deferred["polygon"]):
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                                detail="standalone_print_outline_outside_proposal")
        mask["projective_outline_review"] = True
    elif deferred is None and any(
        candidate.get("reason") in {NESTED_EDGE_CONTOUR, TILED_PRINT_PROPOSAL, CAPTURE_FABRIC_PROPOSAL}
        and _same_visible_outline(candidate.get("polygon"), quad, native_size=(width, height))
        for candidate in pending
    ):
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="standalone_print_confirmation_required")
    embossed_reasons = (EMBOSSED_POCKET_PROPOSAL, EMBOSSED_CARRIER_PROPOSAL)
    is_pocket_offer = (deferred or {}).get("reason") in embossed_reasons
    if payload.confirm_album_pocket_print and not is_pocket_offer:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="album_pocket_print_confirmation_stale")
    if is_pocket_offer:
        if not payload.confirm_album_pocket_print:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="album_pocket_print_confirmation_required")
        match = normalized_polygon_overlap(
            [(point["x"], point["y"]) for point in deferred["polygon"]],
            [(point["x"], point["y"]) for point in quad], raster_size=512,
        )
        if match.second_containment < .75:
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="album_pocket_outline_outside_proposal")
        if inner := deferred.get("contained_print_polygon"):
            existing = normalized_polygon_overlap(
                [(point["x"], point["y"]) for point in inner], [(point["x"], point["y"]) for point in quad],
            )
            if existing.iou >= .85 or existing.second_containment >= .90:
                raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                                    detail="album_carrier_matches_detected_print")
        mask["projective_outline_review"] = True
    elif deferred is None and any(
        candidate.get("reason") in embossed_reasons
        and _same_visible_outline(candidate.get("polygon"), quad, native_size=(width, height))
        for candidate in pending
    ):
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="album_pocket_print_confirmation_required")
    is_dark_offer = (deferred or {}).get("reason") == DARK_ALBUM_PROPOSAL
    if payload.confirm_dark_backing and not is_dark_offer:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="dark_backing_confirmation_stale")
    if is_dark_offer or (deferred is None and any(
        candidate.get("reason") == DARK_ALBUM_PROPOSAL
        and _same_visible_outline(candidate.get("polygon"), quad, native_size=(width, height))
        for candidate in pending
    )):
        if not payload.confirm_dark_backing:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="dark_backing_confirmation_required")
        match = normalized_polygon_overlap(
            [(point["x"], point["y"]) for point in deferred["polygon"]],
            [(point["x"], point["y"]) for point in quad], raster_size=512,
        )
        if match.iou < .30 and match.second_containment < .75:
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="dark_backing_outline_outside_proposal")
        mask["dark_backing_outline_review"] = True
    if size:
        mask["print_size_mm"] = size.model_dump()
    try:
        validate_review_quads(mask, width, height, require_cyclic=True)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    if not tiled_offer and any(
        candidate.get("reason") == TILED_PRINT_PROPOSAL
        and _matches_standalone_print_outline(quad, candidate["polygon"])
        for candidate in pending
    ):
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="pending_tiled_print_confirmation_required")
    neighbors = list(await session.scalars(
        select(ExtractedPhoto).where(ExtractedPhoto.page_id == page_id)
        .with_for_update().execution_options(populate_existing=True)
    ))
    active = [
        photo for photo in neighbors if not is_reviewed_non_print(
            (photo.enhancement_applied or {}).get("review_rejected_detection"), photo.bounding_box, photo.segmentation_mask,
        )
    ]
    if any(_matches_contested_tile(photo, quad) for photo in active):
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="contested_tile_correction_required")
    if any(_matches_merged_print_claim(quad, photo.segmentation_mask or {}) for photo in active):
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="merged_print_pair_claim_conflict")
    if tiled_offer or fabric_offer:
        claims = [candidate for candidate in pending if candidate["id"] != deferred_id
                  and not (fabric_offer and candidate.get("reason") == CAPTURE_FABRIC_PROPOSAL)]
        claims.extend(_current_paper_claims(active))
        outlines = (quad, deferred["polygon"]) if tiled_offer else (quad,)
        if any(tiled_outline_conflicts(outline, claims, width, height) for outline in outlines):
            detail = "capture_fabric_claim_conflict" if fabric_offer else "tiled_print_claim_conflict"
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=detail)
    if (nested_edge_offer or (deferred or {}).get("reason") == EMBOSSED_CARRIER_PROPOSAL) and any(
        _same_visible_outline((photo.segmentation_mask or {}).get("warp_quad"), quad, native_size=(width, height))
        for photo in active
    ):
        detail = "standalone_print_matches_detected_print" if nested_edge_offer else "album_carrier_matches_detected_print"
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=detail)
    replaced_fragments = []
    source_group = (deferred or {}).get("reason") == SOURCE_FRAGMENT_PROPOSAL
    if source_group:
        if not payload.replace_mounted_fragments:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="source_print_fragments_confirmation_required")
        replaced_fragments = _source_fragment_replacements(deferred, quad, active)
        mask["mounted_outline_review"] = True
    elif deferred is None and any(
        candidate.get("reason") == SOURCE_FRAGMENT_PROPOSAL
        and _same_visible_outline(candidate.get("polygon"), quad, native_size=(width, height))
        for candidate in page_fusion_evidence(page.processing_metadata).get("deferred_print_proposals") or []
    ):
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="source_print_fragments_confirmation_required")
    elif payload.replace_mounted_fragments:
        polygons = (deferred or {}).get("fragment_polygons") or []
        if (deferred or {}).get("reason") != "possible_mounted_print" or not 3 <= len(polygons) <= 6:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="mounted_print_fragments_stale")
        crop_points = [(point["x"], point["y"]) for point in quad]
        for polygon in polygons:
            candidates = [
                fragment for fragment in active if fragment.segmentation_confidence is not None
                and (fragment.segmentation_mask or {}).get("polygon") == polygon
                and not variant_path_from_photo(fragment, "full_print")
                and not variant_path_from_photo(fragment, "image_only")
            ]
            if len(candidates) != 1 or candidates[0] in replaced_fragments:
                raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="mounted_print_fragments_stale")
            fragment_points = [(point["x"], point["y"]) for point in polygon]
            if normalized_polygon_overlap(fragment_points, crop_points).first_containment < .98:
                raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                                    detail="mounted_print_outline_excludes_fragments")
            replaced_fragments.append(candidates[0])
        mask["mounted_outline_review"] = True
    suggestion_key = "nested_print_suggestion" if suggestion and suggestion.kind == "nested" else "adjacent_print_suggestion"
    anchor = None
    if suggestion:
        anchor = next((photo for photo in active if photo.id == suggestion.photo_id), None)
        if (anchor is None or (anchor.segmentation_mask or {}).get(suggestion_key)
                != creation_request["adjacent_suggestion"]["polygon"]):
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="adjacent_print_suggestion_stale")
    elif deferred is None:
        # A manually marked outline can exactly match an open suggestion.
        matches = [
            (photo, key) for photo in active for key in ("adjacent_print_suggestion", "nested_print_suggestion")
            if _same_visible_outline((photo.segmentation_mask or {}).get(key), quad)
        ]
        anchor, suggestion_key = matches[0] if len(matches) == 1 else (None, None)
    if anchor is not None:
        anchor.segmentation_mask = {
            key: value for key, value in (anchor.segmentation_mask or {}).items() if key != suggestion_key
        }
    # An older editor could save a suggestion as an unlinked manual photo.
    # Equal current corners describe the same visible footprint on this page.
    if any((photo.segmentation_mask or {}).get("manual_added") is True
           and _same_visible_outline((photo.segmentation_mask or {}).get("warp_quad"), quad) for photo in active):
        if anchor is not None:
            await session.commit()
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="adjacent_print_suggestion_saved")
    box = {
        "x1": min(point["x"] for point in quad), "y1": min(point["y"] for point in quad),
        "x2": max(point["x"] for point in quad), "y2": max(point["y"] for point in quad),
    }
    aspect_ratio = size.aspect_ratio if size else (
        (box["x2"] - box["x1"]) * (width - 1) / ((box["y2"] - box["y1"]) * (height - 1))
    )
    photo = ExtractedPhoto(
        id=photo_id, page_id=page.id, album_id=page.album_id, storage_path="",
        bounding_box=box, segmentation_mask=mask, segmentation_confidence=None,
        aspect_ratio=aspect_ratio,
        geometry_valid=True, status=PhotoStatus.pending,
    )
    creation = {
        **deepcopy(creation_request),
        "recorded_at": datetime.now(timezone.utc).isoformat(),
    }
    if replaced_fragments:
        creation["replaced_fragment_ids"] = [str(fragment.id) for fragment in replaced_fragments]
        for fragment in replaced_fragments:
            metadata = deepcopy(fragment.enhancement_applied or {})
            metadata["review_rejected_detection"] = {
                "bounding_box": deepcopy(fragment.bounding_box), "segmentation_mask": deepcopy(fragment.segmentation_mask),
            }
            metadata["review_reasons"] = list(dict.fromkeys([
                *(metadata.get("review_reasons") or []), "GEOMETRY_REVIEW_WRONG_OBJECT",
            ]))
            fragment.enhancement_applied = metadata
            fragment.status = PhotoStatus.review_needed
            action = "source_print_fragment_reviewed" if source_group else "mounted_print_fragment_reviewed"
            session.add(AuditLog(entity_type="photo", entity_id=fragment.id, action=action,
                                 details={"replacement_photo_id": str(photo_id), "deferred_proposal_id": deferred_id}))
    review_provenance = {
        "manual_creation": creation,
        "prediction_provenance": await _prediction_provenance(session, photo),
    }
    if deferred is not None:
        review_provenance["deferred_print_proposal"] = deepcopy(deferred)
        metadata = deepcopy(page.processing_metadata)
        fusion = page_fusion_evidence(metadata)
        fusion["deferred_print_proposals"] = [candidate for candidate in pending if candidate["id"] != deferred_id]
        page.processing_metadata = metadata
        resolve_nested_edge_parent_reviews(active, [deferred])
    retire_source_fragment_proposals(page, claim=quad)
    _claim_missing_print_trace(page, review_provenance, quad, width, height)
    photo.enhancement_applied = {"review_provenance": review_provenance}
    page.processing_metadata = {**(page.processing_metadata or {}), "photo_count_revision": str(photo_id)}
    session.add(photo)
    session.add(AuditLog(
        entity_type="photo", entity_id=photo_id, action="manual_photo_created",
        details={"page_id": str(page_id), "page_sha256": page.file_hash_sha256, **creation},
    ))
    try:
        await session.flush()
    except IntegrityError:
        await session.rollback()
        existing = await session.get(ExtractedPhoto, photo_id)
        if existing is None:
            raise
        return _replay_manual_photo(existing, page_id, creation_request, response)
    await _review_overlapping_crops(session, photo, width, height)
    return await _render_photo_edit(session, photo_id, settings, explain_protected_cut=explain_protected_cut)


@router.patch("/photos/{photo_id}/bounding-box", response_model=PhotoRead)
async def update_photo_bounding_box(
    photo_id: UUID,
    payload: BoundingBoxUpdate,
    explain_protected_cut: bool = False,
    session: AsyncSession = Depends(get_async_session),
    settings: Settings = Depends(get_settings),
    principal: Principal | None = Depends(require_write_principal),
) -> ExtractedPhoto:
    page, photo = await _locked_review_photo(session, photo_id, principal)
    before = _geometry_snapshot(photo)
    box = payload.bounding_box.as_dict()
    if box["x2"] <= box["x1"] or box["y2"] <= box["y1"]:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Bounding box must have positive area.")
    try:
        width, height = _source_size(page)
        segmentation_mask = (
            payload.segmentation_mask.as_dict() if payload.segmentation_mask else _rect_mask_from_box(box, width, height)
        )
        original_mask = photo.segmentation_mask or {}
        correction = (photo.enhancement_applied or {}).get("correction") or {}
        full_print = (correction.get("crop_modes") or {}).get("full_print") or {}
        saved_review = (photo.enhancement_applied or {}).get("review_provenance") or {}
        shared_rim_traces = saved_review.get("shared_rim_cut_traces") or shared_rim_cut_reference(correction)
        explicit = payload.segmentation_mask is not None and payload.segmentation_mask.warp_quad is not None
        detector_source = original_mask.get("detector_source")
        measured_paper_edit = detector_source == "yolo_onnx" and full_print.get(MOTIF_COMPARISON_PAPER_EDIT) is True
        faint_paper = detector_source == "yolo_onnx" and full_print.get(FAINT_PAPER_REVIEW_REQUIRES_CONFIRMATION) is True
        possible_regions = detector_source == "yolo_onnx" and POSSIBLE_PRINT_REGIONS in original_mask
        if photo.segmentation_confidence is not None and measured_paper_edit:
            outer = segmentation_mask.get("warp_quad") or []
            motif_mode = (correction.get("crop_modes") or {}).get("image_only") or {}
            if (not explicit or segmentation_mask.get("image_quad") is not None
                    or any(_same_visible_outline(previous, outer, native_size=(width, height)) for previous in (
                        full_print.get("warp_quad"), original_mask.get("warp_quad") or original_mask.get("polygon"),
                        motif_mode.get("suggested_warp_quad"),
                    ))):
                raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="print_outline_review_required")
        contact_outline = detector_source in {"contrast_region", "yolo_onnx"} and full_print.get(CONTACT_PRINT_EDGE_CHECK) is True
        opposing_outline = detector_source == "yolo_onnx" and full_print.get(OPPOSING_OUTER_EDGES_CHECK) is True
        unsupported_contrast = (detector_source == "contrast_region"
                                and "PRINT_BOUNDARY_UNSUPPORTED" in (full_print.get("reason_codes") or []))
        confirm_contrast_outline = (
            photo.segmentation_confidence is not None and (possible_regions or (
                full_print.get("status") == "review_needed"
                and (contact_outline or opposing_outline or unsupported_contrast or measured_paper_edit or faint_paper)
                and not capture_outline_review_required(full_print)
            ))
        )
        if confirm_contrast_outline:
            if not payload.confirm_print_outline or not explicit or segmentation_mask.get("image_quad") is not None:
                raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="contrast_print_outline_review_required")
            # The unsupported automatic outline cannot claim paper for the replacement.
            segmentation_mask["polygon"] = deepcopy(segmentation_mask["warp_quad"])
            segmentation_mask["visible_polygon"] = deepcopy(segmentation_mask["warp_quad"])
        if full_print.get("suggested_warp_quad") and photo.segmentation_confidence is not None:
            outer = segmentation_mask.get("warp_quad") or []
            previous_outlines = (full_print.get("warp_quad"), original_mask.get("warp_quad") or original_mask.get("polygon"))
            if not explicit or (not payload.confirm_print_outline and (
                full_print.get("suggested_outline_source") in (ADJACENT_STOCK_REVIEW_SOURCE, INWARD_CORNER_REVIEW_SOURCE)
                or any(_same_visible_outline(previous, outer, native_size=(width, height)) for previous in previous_outlines)
            )):
                raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="print_outline_review_required")
        if original_mask.get("crossing_print_suggestion"):
            if payload.segmentation_mask is None or payload.segmentation_mask.warp_quad is None:
                raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="crossing_print_outline_required")
            outer = segmentation_mask.get("warp_quad")
            if (segmentation_mask.get("image_quad") is not None
                    or _same_visible_outline(original_mask.get("polygon"), outer, native_size=(width, height))
                    or _same_visible_outline(full_print.get("warp_quad"), outer, native_size=(width, height))):
                raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="crossing_print_outline_required")
            # A new outer cannot inherit visible pixels from the crossing proposal.
            segmentation_mask["polygon"] = deepcopy(outer)
            segmentation_mask["visible_polygon"] = deepcopy(outer)
            segmentation_mask[REVIEWED_CROSSING_OUTLINE] = True
        for flag in ("album_window_occluded", "manual_added", "capture_edge_clipped", REVIEWED_CROSSING_OUTLINE):
            if (photo.segmentation_mask or {}).get(flag) is True:
                segmentation_mask[flag] = True
        if margin_state := original_mask.get(MANUAL_MARGIN_CONTEXT):
            segmentation_mask[MANUAL_MARGIN_CONTEXT] = deepcopy(margin_state)
        if corner_state := original_mask.get(ROUNDED_CORNER_WEDGES):
            segmentation_mask[ROUNDED_CORNER_WEDGES] = deepcopy(corner_state)
        # A paper edit does not settle a separate mark in the proposed motif margin.
        motif_mode = (correction.get("crop_modes") or {}).get("image_only") or {}
        margin_regions = original_mask.get("paper_stock_margin_regions")
        if not margin_regions and motif_mode.get("margin_mark_source") == PAPER_STOCK_MOTIF_SOURCE:
            margin_regions = motif_mode.get("margin_mark_regions")
        if margin_regions:
            segmentation_mask["paper_stock_margin_regions"] = deepcopy(margin_regions)
        for kind in ("mounted", "projective", "dark_backing"):
            review, confirmed = f"{kind}_outline_review", f"{kind}_outline_confirmed"
            if original_mask.get(review) is True:
                segmentation_mask[review] = True
                explicit_outline = payload.segmentation_mask is not None and payload.segmentation_mask.warp_quad is not None
                if original_mask.get(confirmed) is True or (explicit_outline and payload.confirm_print_outline):
                    segmentation_mask[confirmed] = True
        if "visible_polygon" not in segmentation_mask:
            segmentation_mask["visible_polygon"] = deepcopy(segmentation_mask.get("polygon"))
        if "warp_quad" not in segmentation_mask and len(segmentation_mask.get("polygon") or []) == 4:
            segmentation_mask["warp_quad"] = deepcopy(segmentation_mask["polygon"])
        if original_mask.get("mounted_paper_edge_traces") and _same_visible_outline(
            full_print.get("warp_quad"), segmentation_mask.get("warp_quad"), native_size=(width, height),
        ):
            segmentation_mask["mounted_paper_edge_traces"] = deepcopy(original_mask["mounted_paper_edge_traces"])
        size = payload.segmentation_mask.print_size_mm if payload.segmentation_mask else None
        edited_fields = payload.segmentation_mask.model_fields_set if payload.segmentation_mask else set()
        previous_mask = (photo.segmentation_mask or {}) if photo.segmentation_confidence is None else {}
        if "orientation_degrees" not in edited_fields and previous_mask.get("orientation_degrees") is not None:
            segmentation_mask["orientation_degrees"] = previous_mask["orientation_degrees"]
        previous_quad = previous_mask.get("warp_quad") or []
        updated_quad = segmentation_mask.get("warp_quad") or []
        if "print_size_mm" not in edited_fields and previous_mask.get("print_size_mm") is not None:
            size = PrintSizeMM.model_validate(previous_mask["print_size_mm"])
            segmentation_mask["print_size_mm"] = size.model_dump()
            if previous_quad and updated_quad:
                segmentation_mask["warp_quad"] = _align_measured_quad(previous_quad, updated_quad, width, height)
        if "image_quad" not in edited_fields and previous_mask.get("image_quad") and previous_quad and updated_quad:
            old_corners = {(p["x"], p["y"]) for p in previous_quad}
            if old_corners == {(p["x"], p["y"]) for p in updated_quad}:
                segmentation_mask["image_quad"] = deepcopy(previous_mask["image_quad"])
        manual_addition = segmentation_mask.get("manual_added") is True
        if manual_addition and segmentation_mask.get("warp_quad") is None:
            raise ValueError("manual_print_requires_warp_quad")
        validate_review_quads(segmentation_mask, width, height, require_cyclic=manual_addition)
        if shared_rim_traces and (
            payload.segmentation_mask is None or payload.segmentation_mask.warp_quad is None
            or not shared_rim_cut_is_inside(segmentation_mask["warp_quad"], shared_rim_traces, width, height)
        ):
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="shared_rim_correction_required")
        if manual_addition:
            segmentation_mask["polygon"] = deepcopy(segmentation_mask["warp_quad"])
            segmentation_mask["visible_polygon"] = deepcopy(segmentation_mask["warp_quad"])
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    if MERGED_PRINT_PAIR in original_mask:
        if not _resolve_merged_print_outline(original_mask, payload, segmentation_mask, width, height):
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="merged_print_pair_resolution_required")
        segmentation_mask["polygon"] = deepcopy(segmentation_mask["warp_quad"])
        segmentation_mask["visible_polygon"] = deepcopy(segmentation_mask["warp_quad"])
        merged_resolution = "one_shared_print" if payload.confirm_merged_print_as_one else "one_child"
    elif payload.confirm_merged_print_as_one:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="merged_print_pair_resolution_stale")
    else:
        merged_resolution = None
    pending = page_fusion_evidence(page.processing_metadata).get("deferred_print_proposals") or []
    if any(
        candidate.get("reason") == TILED_PRINT_PROPOSAL
        and any(_matches_standalone_print_outline(outline, candidate["polygon"]) for outline in (
            segmentation_mask.get("warp_quad"), segmentation_mask.get("visible_polygon"), segmentation_mask.get("polygon"),
        ) if outline)
        for candidate in pending
    ):
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="pending_tiled_print_confirmation_required")
    neighbors = await session.scalars(
        select(ExtractedPhoto).where(ExtractedPhoto.page_id == page.id, ExtractedPhoto.id != photo.id)
        .with_for_update().execution_options(populate_existing=True)
    )
    active_neighbors = [neighbor for neighbor in neighbors if not photo_is_reviewed_non_print(neighbor)]
    if any(
        _matches_merged_print_claim(outline, neighbor.segmentation_mask or {})
        for neighbor in active_neighbors
        for outline in (segmentation_mask.get("warp_quad"), segmentation_mask.get("visible_polygon"), segmentation_mask.get("polygon"))
        if outline
    ):
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="merged_print_pair_claim_conflict")
    if original_mask.get("crossing_print_suggestion_source") == CONTESTED_TILE_OUTLINE:
        claims = _current_paper_claims(active_neighbors)
        claims.extend(pending)
        if tiled_outline_conflicts(segmentation_mask["warp_quad"], claims, width, height):
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="contested_tile_claim_conflict")
    prediction_provenance = await _prediction_provenance(session, photo)
    after = {
        "bounding_box": deepcopy(box),
        "segmentation_mask": deepcopy(segmentation_mask),
        "segmentation_confidence": None,
        "aspect_ratio": size.aspect_ratio if size else (box["x2"] - box["x1"]) / max(box["y2"] - box["y1"], 1e-9),
        "geometry_valid": True,
        "status": PhotoStatus.pending.value,
    }
    metadata = deepcopy(photo.enhancement_applied or {})
    was_non_print = photo_is_reviewed_non_print(photo)
    previous_crop = {}
    if not was_non_print:
        previous_crop = dict(original_mask)
        if not previous_crop:
            try:
                previous_crop = _rect_mask_from_box(before["bounding_box"], width, height)
            except ValueError:
                pass  # A previously invalid box supplies no usable comparison.
        if full_print.get("warp_quad"):
            previous_crop["warp_quad"] = full_print["warp_quad"]
            if before["segmentation_confidence"] is None and full_print.get("status") == "ready":
                previous_crop["ready_before_edit"] = True
    if was_non_print:
        page.processing_metadata = {**(page.processing_metadata or {}), "photo_count_revision": str(uuid4())}
        metadata["review_reasons"] = [
            reason for reason in metadata.get("review_reasons") or [] if reason != "GEOMETRY_REVIEW_WRONG_OBJECT"
        ]
    if merged_resolution:
        page.processing_metadata = {**(page.processing_metadata or {}), "photo_count_revision": str(uuid4())}
    metadata.pop("review_rejected_detection", None)
    review_provenance = deepcopy(metadata.get("review_provenance") or {})
    if shared_rim_traces:
        review_provenance["shared_rim_cut_traces"] = deepcopy(shared_rim_traces)
    if payload.album_page_mark_review_id is not None:
        offer = correction.get("album_page_mark_review") or {}
        off_print = offer.get("source") == OFF_PRINT_ADVISORY
        if (payload.segmentation_mask is None or payload.segmentation_mask.warp_quad is None
                or str(payload.album_page_mark_review_id) != offer.get("id")
                or not album_mark_source_matches(offer, page.file_hash_sha256, page.processing_metadata)
                or (not off_print and (full_print.get("outline_preview") or {}).get("source") != POSSIBLE_OVEREXTENDED_PRINT_EDGE)
                or photo_is_reviewed_non_print(photo)):
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="album_page_mark_review_stale")
        if off_print and payload.album_page_mark_review_indices != list(range(len(offer["regions"]))):
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="album_page_mark_choices_required")
        if not off_print and payload.album_page_mark_review_indices is not None:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="album_page_mark_review_stale")
        review_provenance["reviewed_album_mark"] = {
            **deepcopy(offer), "previous_quad": deepcopy(full_print["warp_quad"]),
            "warp_quad": deepcopy(segmentation_mask["warp_quad"]), "reviewed_at": datetime.now(timezone.utc).isoformat(),
        }
        if off_print:
            review_provenance["reviewed_album_mark"]["confirmed_indices"] = payload.album_page_mark_review_indices
        session.add(AuditLog(entity_type="photo", entity_id=photo.id, action="album_page_mark_reviewed",
                             details=deepcopy(review_provenance["reviewed_album_mark"])))
    elif payload.album_page_mark_review_indices is not None:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="album_page_mark_review_stale")
    elif (reviewed_mark := review_provenance.get("reviewed_album_mark")) and not album_mark_matches_quad(
        reviewed_mark, segmentation_mask.get("warp_quad"),
    ):
        review_provenance.pop("reviewed_album_mark")
    if before["segmentation_confidence"] is not None:
        previous_correction = metadata.get("correction") or {}
        print_mode = (previous_correction.get("crop_modes") or {}).get("full_print") or {}
        if len(print_mode.get("warp_quad") or []) == 4:
            before["print_warp_quad"] = deepcopy(print_mode["warp_quad"])
            if "PRINT_POSSIBLY_OCCLUDED" in (print_mode.get("reason_codes") or []):
                before["neighbor_occlusion_unverified"] = True
        if print_mode:
            before["capture_outline_unverified"] = capture_outline_review_required(print_mode)
        curve = paper_curve_review_reference(previous_correction)
        if curve is not None:
            review_provenance.setdefault("paper_curve_review", deepcopy(curve))
        backed_edge = backed_edge_review_reference(print_mode)
        if backed_edge is not None:
            review_provenance.setdefault("backed_edge_review", deepcopy(backed_edge))
    review_provenance.setdefault("original_prediction", before)
    review_provenance.setdefault("prediction_provenance", prediction_provenance)
    _claim_missing_print_trace(
        page, review_provenance, segmentation_mask.get("warp_quad"), width, height, polygon=segmentation_mask.get("polygon"),
    )
    revisions = list(review_provenance.get("correction_revisions") or [])
    revisions.append(
        {
            "revision": len(revisions) + 1,
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "before": before,
            "after": after,
            "source": segmentation_mask.get("source"),
            **({"contrast_print_outline_confirmed": True} if confirm_contrast_outline else {}),
            **({"merged_print_resolution": merged_resolution} if merged_resolution else {}),
        }
    )
    review_provenance["correction_revisions"] = revisions
    metadata["review_provenance"] = review_provenance
    photo.bounding_box = box
    photo.segmentation_mask = segmentation_mask
    photo.segmentation_confidence = None
    photo.aspect_ratio = after["aspect_ratio"]
    photo.geometry_valid = True
    photo.enhancement_applied = metadata
    photo.status = PhotoStatus.pending
    session.add(
        AuditLog(
            entity_type="photo",
            entity_id=photo.id,
            action="bounding_box_updated",
            details={
                "before": before,
                "after": after,
                "prediction_provenance": prediction_provenance,
                **({"merged_print_resolution": merged_resolution} if merged_resolution else {}),
                # Retain the legacy fields for existing audit consumers.
                "bounding_box": box,
                "segmentation_mask": segmentation_mask,
            },
        )
    )
    prior_mask = before["segmentation_mask"] or {}
    prior_crop_modes = (metadata.get("correction") or {}).get("crop_modes") or {}
    prior_outer = prior_mask.get("warp_quad") or (prior_crop_modes.get("full_print") or {}).get("warp_quad")
    outer = segmentation_mask.get("warp_quad")
    edited_claim = outer if outer and not _same_visible_outline(prior_outer, outer) else None
    edited_polygon = not outer and segmentation_mask.get("visible_polygon") != (
        prior_mask.get("visible_polygon") or prior_mask.get("polygon")
    )
    if retire_source_fragment_proposals(page, fragment=prior_mask.get("polygon"), claim=edited_claim):
        page.processing_metadata = {**(page.processing_metadata or {}), "photo_count_revision": str(uuid4())}
    if (was_non_print or segmentation_mask.get("manual_added") is True
            or paper_claim_groups(segmentation_mask, original_prediction_mask(metadata))
            or edited_claim or edited_polygon):
        await _review_overlapping_crops(session, photo, width, height, neighbors=active_neighbors)
    await session.flush()
    return await _render_photo_edit(
        session, photo.id, settings, previous_manual_crop=previous_crop, explain_protected_cut=explain_protected_cut,
        after_render=(lambda: _offer_contested_tile_followup(page, original_mask, photo, active_neighbors, width, height))
        if original_mask.get(CONTESTED_TILE_FOLLOWUP) else None,
    )


@router.patch("/ocr/{ocr_id}", response_model=OCRRead)
async def update_ocr_result(
    ocr_id: UUID,
    payload: OCRUpdate,
    session: AsyncSession = Depends(get_async_session),
    principal: Principal | None = Depends(require_write_principal),
) -> OCRResult:
    query = select(OCRResult).join(Page, Page.id == OCRResult.page_id).where(OCRResult.id == ocr_id)
    if principal is not None:
        query = query.where(Page.album.has(tenant_id=principal.tenant_id))
    ocr = await session.scalar(query)
    if ocr is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="OCR result not found.")
    ocr.text_content = payload.text_content
    ocr.text_type = payload.text_type
    ocr.is_verified = payload.is_verified
    session.add(
        AuditLog(entity_type="page", entity_id=ocr.page_id, action="ocr_updated", details={"ocr_id": str(ocr.id)})
    )
    await session.commit()
    if ocr.photo_id:
        await persistence_task_logic(session, ocr.photo_id)
    await session.refresh(ocr)
    return ocr


@router.post("/photos/{photo_id}/reprocess", response_model=PhotoRead)
async def reprocess_photo(
    photo_id: UUID,
    preset: str | None = Query(default="auto"),
    session: AsyncSession = Depends(get_async_session),
    settings: Settings = Depends(get_settings),
    principal: Principal | None = Depends(require_write_principal),
) -> ExtractedPhoto:
    page, photo = await _locked_review_photo(session, photo_id, principal)
    existing_modes = ((photo.enhancement_applied or {}).get("correction") or {}).get("crop_modes")
    if (photo.segmentation_confidence is None and (photo.segmentation_mask or {}).get("warp_quad")
            and (settings.enable_precise_photo_crops or existing_modes) and not photo_is_reviewed_non_print(photo)):
        await _review_overlapping_crops(session, photo, *_source_size(page))
    return await _render_photo_edit(session, photo.id, settings, reject_invalid_edit=False, preset=preset)
