import asyncio
from copy import deepcopy
from datetime import datetime, timezone
from math import hypot
from pathlib import Path
import shutil
from typing import Annotated
from uuid import UUID, uuid4, uuid5

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
from app.pipeline.crop_validation import route_ready_crop_modes_to_review
from app.pipeline.crossing_prints import REVIEWED_CROSSING_OUTLINE, reviewed_outline_covers_traces
from app.pipeline.decisions import is_reviewed_non_print
from app.pipeline.edge_trace import measure_edge_trace
from app.pipeline.image_ops import load_image, normalized_box_to_pixels
from app.pipeline.manual_paper_boundary import (
    capture_outline_review_required, paper_curve_from_review_traces, paper_curve_review_reference,
)
from app.pipeline.perspective import manual_print_intersects, validate_review_quads
from app.pipeline.presets import available_presets
from app.pipeline.variants import (
    CROP_REVIEW_PREVIEWS,
    released_variant_from_photo,
    variant_path_from_photo,
    variant_urls,
    variants_from_metadata,
)
from app.schemas import BoundingBoxUpdate, OCRRead, OCRUpdate, PhotoDetail, PhotoRead
from app.schemas.common import PrintSizeMM
from app.schemas.pipeline import ManualPhotoCreate
from app.services.pipeline import (
    deduplication_task_logic,
    enhancement_task_logic,
    persistence_task_logic,
    perspective_correction_task_logic,
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


def _photo_urls(photo: ExtractedPhoto) -> dict[str, str | None]:
    return variant_urls(photo)


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
) -> None:
    if len(quad or []) != 4:
        return
    metadata = deepcopy(page.processing_metadata or {})
    segmentation = (metadata.get("steps") or {}).get("segmentation") or {}
    traces = segmentation.get("missing_print_edge_traces", [])
    if not reviewed_outline_covers_traces(quad, traces):
        return
    curve = paper_curve_from_review_traces(quad, traces, width, height)
    if curve is not None:
        review_provenance["paper_trace_review"] = curve
    segmentation.pop("missing_print_edge_traces")
    page.processing_metadata = metadata


async def _render_photo_edit(
    session: AsyncSession, photo_id: UUID, settings: Settings, *,
    reject_invalid_edit: bool = True, preset: str | None = "auto",
) -> ExtractedPhoto:
    output_directories: set[Path] = set()
    try:
        try:
            correction = await perspective_correction_task_logic(
                session, photo_id, settings, reject_invalid_edit=reject_invalid_edit,
                commit=False, output_directories=output_directories,
            )
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


async def _review_overlapping_exports(
    session: AsyncSession, photo: ExtractedPhoto, width: int, height: int,
) -> None:
    mask = photo.segmentation_mask or {}
    quad = mask.get("warp_quad")
    if quad is None:
        return
    neighbors = await session.scalars(
        select(ExtractedPhoto)
        .where(ExtractedPhoto.page_id == photo.page_id, ExtractedPhoto.id != photo.id)
        .with_for_update()
        .execution_options(populate_existing=True)
    )
    for neighbor in neighbors:
        if (mask.get("manual_added") is not True and mask.get(REVIEWED_CROSSING_OUTLINE) is not True
                and (neighbor.segmentation_mask or {}).get(REVIEWED_CROSSING_OUTLINE) is not True):
            continue
        metadata = deepcopy(neighbor.enhancement_applied or {})
        modes = (metadata.get("correction") or {}).get("crop_modes") or {}
        if is_reviewed_non_print(metadata.get("review_rejected_detection"), neighbor.bounding_box, neighbor.segmentation_mask):
            continue
        if not any(mode.get("status") == "ready" for mode in modes.values()):
            continue
        neighbor_quad = (modes.get("full_print") or {}).get("warp_quad")
        if not manual_print_intersects(neighbor_quad, quad, width, height):
            continue
        reason = "PRINT_POSSIBLY_OCCLUDED"
        route_ready_crop_modes_to_review(modes, [reason])
        metadata["review_reasons"] = list(dict.fromkeys([*(metadata.get("review_reasons") or []), reason]))
        neighbor.enhancement_applied = metadata
        neighbor.status = PhotoStatus.review_needed
        session.add(AuditLog(
            entity_type="photo", entity_id=neighbor.id, action="crop_review_reopened",
            details={"reason": reason, "overlapping_photo_id": str(photo.id)},
        ))


def _replay_manual_photo(
    photo: ExtractedPhoto, page_id: UUID, creation_request: dict, response: Response,
) -> ExtractedPhoto:
    provenance = (photo.enhancement_applied or {}).get("review_provenance") or {}
    creation = provenance.get("manual_creation") or {}
    if photo.page_id != page_id or any(creation.get(key) != value for key, value in creation_request.items()):
        if creation_request["adjacent_suggestion"] is not None:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="adjacent_print_suggestion_saved")
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
    data = PhotoDetail.model_validate(photo)
    data.ocr_results = [OCRRead.model_validate(result) for result in photo.ocr_results]
    data.urls = _photo_urls(photo)
    return data


@router.get("/photos/{photo_id}/image")
async def get_photo_image(
    photo_id: UUID,
    variant: str | None = Query(default=None, pattern="^[a-zA-Z0-9_-]+$"),
    session: AsyncSession = Depends(get_async_session),
    principal: Principal | None = Depends(get_principal),
) -> FileResponse:
    photo = await _photo_for_principal(session, photo_id, principal)
    selected_variant = variant or released_variant_from_photo(photo)
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
        headers={"Cache-Control": "private, no-store"} if preview else None,
    )


@router.get("/photos/{photo_id}/variants")
async def get_photo_variants(
    photo_id: UUID,
    session: AsyncSession = Depends(get_async_session),
    principal: Principal | None = Depends(get_principal),
) -> dict:
    photo = await _photo_for_principal(session, photo_id, principal)
    return {
        "selected_variant": released_variant_from_photo(photo),
        "variants": variants_from_metadata(photo.enhancement_applied),
        "urls": _photo_urls(photo),
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
    session: AsyncSession = Depends(get_async_session),
    settings: Settings = Depends(get_settings),
    principal: Principal | None = Depends(require_write_principal),
) -> ExtractedPhoto:
    page = await _page_for_principal(session, page_id, principal, for_update=True)
    quad = [point.model_dump() for point in payload.warp_quad]
    size = payload.print_size_mm
    suggestion = payload.adjacent_suggestion
    creation_request = {
        "request_id": str(payload.request_id),
        "warp_quad": quad,
        "print_size_mm": size.model_dump() if size else None,
        "adjacent_suggestion": suggestion.model_dump(mode="json") if suggestion else None,
    }
    # Two sessions can edit the same outline with different request UUIDs.
    # Its source and displayed coordinates identify a single addition.
    identity = f"adjacent-print:{suggestion.model_dump_json()}" if suggestion else str(payload.request_id)
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
    width, height = _source_size(page)
    mask = {"source": "manual_added", "manual_added": True, "polygon": quad, "visible_polygon": quad, "warp_quad": quad}
    if size:
        mask["print_size_mm"] = size.model_dump()
    try:
        validate_review_quads(mask, width, height, require_cyclic=True)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    neighbors = list(await session.scalars(
        select(ExtractedPhoto).where(ExtractedPhoto.page_id == page_id)
        .with_for_update().execution_options(populate_existing=True)
    ))
    active = [
        photo for photo in neighbors if not is_reviewed_non_print(
            (photo.enhancement_applied or {}).get("review_rejected_detection"), photo.bounding_box, photo.segmentation_mask,
        )
    ]
    if suggestion:
        anchor = next((photo for photo in active if photo.id == suggestion.photo_id), None)
        if (anchor is None or (anchor.segmentation_mask or {}).get("adjacent_print_suggestion")
                != creation_request["adjacent_suggestion"]["polygon"]):
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="adjacent_print_suggestion_stale")
    else:
        # A manually marked outline can exactly match an open suggestion.
        matches = [
            photo for photo in active
            if _same_visible_outline((photo.segmentation_mask or {}).get("adjacent_print_suggestion"), quad)
        ]
        anchor = matches[0] if len(matches) == 1 else None
    if anchor is not None:
        anchor.segmentation_mask = {
            key: value for key, value in (anchor.segmentation_mask or {}).items() if key != "adjacent_print_suggestion"
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
    review_provenance = {
        "manual_creation": creation,
        "prediction_provenance": await _prediction_provenance(session, photo),
    }
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
    await _review_overlapping_exports(session, photo, width, height)
    return await _render_photo_edit(session, photo_id, settings)


@router.patch("/photos/{photo_id}/bounding-box", response_model=PhotoRead)
async def update_photo_bounding_box(
    photo_id: UUID,
    payload: BoundingBoxUpdate,
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
        if original_mask.get("crossing_print_suggestion"):
            if payload.segmentation_mask is None or payload.segmentation_mask.warp_quad is None:
                raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="crossing_print_outline_required")
            outer = segmentation_mask.get("warp_quad")
            correction = (photo.enhancement_applied or {}).get("correction") or {}
            full_print = (correction.get("crop_modes") or {}).get("full_print") or {}
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
        if "visible_polygon" not in segmentation_mask:
            segmentation_mask["visible_polygon"] = deepcopy(segmentation_mask.get("polygon"))
        if "warp_quad" not in segmentation_mask and len(segmentation_mask.get("polygon") or []) == 4:
            segmentation_mask["warp_quad"] = deepcopy(segmentation_mask["polygon"])
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
        if manual_addition:
            segmentation_mask["polygon"] = deepcopy(segmentation_mask["warp_quad"])
            segmentation_mask["visible_polygon"] = deepcopy(segmentation_mask["warp_quad"])
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
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
    if is_reviewed_non_print(metadata.get("review_rejected_detection"), photo.bounding_box, photo.segmentation_mask):
        page.processing_metadata = {**(page.processing_metadata or {}), "photo_count_revision": str(uuid4())}
        metadata["review_reasons"] = [
            reason for reason in metadata.get("review_reasons") or [] if reason != "GEOMETRY_REVIEW_WRONG_OBJECT"
        ]
    metadata.pop("review_rejected_detection", None)
    review_provenance = deepcopy(metadata.get("review_provenance") or {})
    if before["segmentation_confidence"] is not None:
        previous_correction = metadata.get("correction") or {}
        print_mode = (previous_correction.get("crop_modes") or {}).get("full_print") or {}
        if len(print_mode.get("warp_quad") or []) == 4:
            before["print_warp_quad"] = deepcopy(print_mode["warp_quad"])
        if print_mode:
            before["capture_outline_unverified"] = capture_outline_review_required(print_mode)
        curve = paper_curve_review_reference(previous_correction)
        if curve is not None:
            review_provenance.setdefault("paper_curve_review", deepcopy(curve))
    review_provenance.setdefault("original_prediction", before)
    review_provenance.setdefault("prediction_provenance", prediction_provenance)
    _claim_missing_print_trace(page, review_provenance, segmentation_mask.get("warp_quad"), width, height)
    revisions = list(review_provenance.get("correction_revisions") or [])
    revisions.append(
        {
            "revision": len(revisions) + 1,
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "before": before,
            "after": after,
            "source": segmentation_mask.get("source"),
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
                # Retain the legacy fields for existing audit consumers.
                "bounding_box": box,
                "segmentation_mask": segmentation_mask,
            },
        )
    )
    await session.flush()
    prior_mask = before["segmentation_mask"] or {}
    prior_crop_modes = (metadata.get("correction") or {}).get("crop_modes") or {}
    prior_outer = prior_mask.get("warp_quad") or (prior_crop_modes.get("full_print") or {}).get("warp_quad")
    outer = segmentation_mask.get("warp_quad")
    if segmentation_mask.get("manual_added") is True or (outer and not _same_visible_outline(prior_outer, outer)):
        await _review_overlapping_exports(session, photo, width, height)
    return await _render_photo_edit(session, photo.id, settings)


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
    _, photo = await _locked_review_photo(session, photo_id, principal)
    return await _render_photo_edit(session, photo.id, settings, reject_invalid_edit=False, preset=preset)
