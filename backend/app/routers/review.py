from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy import case, func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.core.auth import Principal, get_principal, require_write_principal
from app.database import get_async_session
from app.models import (
    ExtractedPhoto,
    OCRResult,
    Page,
    PageStatus,
    PhotoStatus,
    ProcessingRun,
    ReviewEvent,
    StageRun,
)
from app.pipeline.decisions import is_reviewed_non_print, motif_geometry_requires_review
from app.pipeline.crossing_fallback_review import CROSSING_FALLBACK_COUNT_REVIEW, CROSSING_FALLBACK_TRACES
from app.pipeline.manual_paper_boundary import PAPER_CLAIM_OVERLAP_CHECKS, paper_claims_need_overlap_check
from app.pipeline.multipanel_prints import FILM_GRID_REVIEW, PAPER_GRID_REVIEW, PAPER_JOIN_REVIEW
from app.pipeline.protected_regions import album_mark_source_matches
from app.pipeline.variants import variant_path_from_photo, variant_urls, variants_from_metadata
from app.schemas import PhotoRead, ReviewItem
from app.services.pipeline import page_fusion_evidence, photo_views_for_current_album_marks, retire_source_fragment_proposals

router = APIRouter(prefix="/api/v1/review", tags=["review"])


class MobileReviewAnswer(BaseModel):
    task_type: str = Field(pattern="^(segmentation|geometry|ocr|enhancement)$")
    task_id: UUID
    answer: str = Field(min_length=1, max_length=120)
    reason_code: str | None = Field(default=None, max_length=120)
    idempotency_key: str = Field(min_length=8, max_length=64)
    reviewer: str = Field(default="local-mobile", max_length=120)
    metadata: dict = Field(default_factory=dict)


GROUP_EXCLUSIONS = {
    "exclude_film_group": (3, 2, "film_grid_review_seam", "film_group_photo_ids", FILM_GRID_REVIEW, "film"),
    "exclude_paper_group": (2, 2, "paper_join_review_traces", "paper_group_photo_ids", PAPER_JOIN_REVIEW, "paper"),
    "exclude_crossing_group": (1, 4, CROSSING_FALLBACK_TRACES, "crossing_group_photo_ids", CROSSING_FALLBACK_COUNT_REVIEW, "crossing"),
}
CONFIRM_SHARED_PAPER_GRID = "confirm_shared_paper_grid"
COUNT_GROUP_ACTIONS = {*GROUP_EXCLUSIONS, CONFIRM_SHARED_PAPER_GRID}

ALLOWED_ANSWERS = {
    "segmentation": {"correct", "missing_print", "too_many", "unsure", *COUNT_GROUP_ACTIONS},
    "geometry": {
        "correct",
        "too_tight",
        "too_wide",
        "wrong_object",
        "split_print",
        "merged_prints",
        "protected_cut",
        "unsure",
    },
    "ocr": {"correct", "incorrect", "unsure"},
    "enhancement": {"left", "right", "equal", "both_bad", "unsure"},
}

# Only answers which actually finish the current task suppress it from the
# queue.  In particular, ``unsure`` and unresolved crop/count problems remain
# available for a later reviewer or a corrected revision.
COMPLETING_ANSWERS = {
    "segmentation": {"correct"},
    "geometry": {"correct"},
    "ocr": {"correct"},
    "enhancement": {"left", "right", "equal", "both_bad"},
}


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _manual_source(mask: dict | None) -> bool:
    return str((mask or {}).get("source", "")).startswith("manual")


async def _run_provenance(session: AsyncSession, page: Page) -> dict:
    run = await session.scalar(
        select(ProcessingRun).where(ProcessingRun.page_id == page.id).order_by(ProcessingRun.created_at.desc()).limit(1)
    )
    if run is None:
        return {
            "page_sha256": page.file_hash_sha256,
            "segmentation": deepcopy((page.processing_metadata or {}).get("steps", {}).get("segmentation", {})),
        }
    stage = await session.scalar(
        select(StageRun)
        .where(StageRun.processing_run_id == run.id, StageRun.stage == "segmentation")
        .order_by(StageRun.attempt.desc())
        .limit(1)
    )
    return {
        "run_id": str(run.id),
        "page_sha256": page.file_hash_sha256,
        "input_sha256": run.input_sha256,
        "pipeline_version": run.pipeline_version,
        "config_fingerprint": run.config_fingerprint,
        "run_manifest": deepcopy(run.manifest or {}),
        "segmentation_stage": {
            "id": str(stage.id),
            "attempt": stage.attempt,
            "status": stage.status.value,
            "details": deepcopy(stage.details or {}),
        }
        if stage is not None
        else None,
        "segmentation": deepcopy((page.processing_metadata or {}).get("steps", {}).get("segmentation", {})),
    }


def _photo_snapshot(photo: ExtractedPhoto, provenance: dict) -> dict:
    metadata = deepcopy(photo.enhancement_applied or {})
    review_provenance = deepcopy(metadata.get("review_provenance") or {})
    current_geometry = {
        "bounding_box": deepcopy(photo.bounding_box),
        "segmentation_mask": deepcopy(photo.segmentation_mask),
        "segmentation_confidence": photo.segmentation_confidence,
        "aspect_ratio": photo.aspect_ratio,
        "geometry_valid": photo.geometry_valid,
        "status": photo.status.value,
    }
    original_prediction = deepcopy(review_provenance.get("original_prediction"))
    if original_prediction is None:
        original_prediction = deepcopy(current_geometry)
        original_prediction["snapshot_source"] = "current_geometry_no_prior_revision"
    return {
        "photo_id": str(photo.id),
        "original_prediction": original_prediction,
        "prediction_provenance": deepcopy(review_provenance.get("prediction_provenance") or provenance),
        "current_geometry": current_geometry,
        "current_correction": deepcopy(metadata.get("correction") or {}),
        "orientation": deepcopy(
            (metadata.get("correction") or {}).get("orientation") or metadata.get("orientation") or {}
        ),
        "review_reasons": list(metadata.get("review_reasons") or []),
        "selected_variant": metadata.get("selected_variant"),
        "manual_geometry": _manual_source(photo.segmentation_mask),
    }


def _rejected_detection(photo: ExtractedPhoto) -> bool:
    return is_reviewed_non_print(
        (photo.enhancement_applied or {}).get("review_rejected_detection"),
        photo.bounding_box, photo.segmentation_mask,
    )


async def _task_context(
    session: AsyncSession,
    task_type: str,
    task_id: UUID,
    principal: Principal | None,
    *,
    lock_photos: bool = False,
) -> dict:
    if task_type == "segmentation":
        query = select(Page).where(Page.id == task_id)
        if principal is not None:
            query = query.where(Page.album.has(tenant_id=principal.tenant_id))
        page = await session.scalar(query.with_for_update())
        if page is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Segmentation review task not found.")
        photo_query = (
            select(ExtractedPhoto)
            .where(ExtractedPhoto.page_id == page.id)
            .order_by(ExtractedPhoto.created_at.asc())
        )
        if lock_photos:
            photo_query = photo_query.with_for_update()
        photos = (await session.scalars(photo_query)).all()
        return {"page": page, "photos": [photo for photo in photos if not _rejected_detection(photo)]}

    if task_type in {"geometry", "enhancement"}:
        query = select(ExtractedPhoto).where(ExtractedPhoto.id == task_id)
        if principal is not None:
            query = query.where(ExtractedPhoto.album.has(tenant_id=principal.tenant_id))
        photo = await session.scalar(query)
        if photo is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail=f"{task_type.title()} review task not found."
            )
        page = await session.get(Page, photo.page_id, with_for_update=task_type == "geometry")
        if task_type == "geometry":
            photo = await session.scalar(query.with_for_update().execution_options(populate_existing=True))
            if photo is None:
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Geometry review task not found.")
        return {"page": page, "photo": photo, "photos": [photo]}

    query = select(OCRResult).join(Page, Page.id == OCRResult.page_id).where(OCRResult.id == task_id)
    if principal is not None:
        query = query.where(Page.album.has(tenant_id=principal.tenant_id))
    ocr = await session.scalar(query)
    if ocr is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="OCR review task not found.")
    page = await session.get(Page, ocr.page_id)
    return {"page": page, "ocr": ocr, "photos": []}


async def _snapshot_task(session: AsyncSession, payload: MobileReviewAnswer, context: dict) -> dict:
    page: Page = context["page"]
    provenance = await _run_provenance(session, page)
    snapshot = {
        "schema_version": 1,
        "captured_at": _timestamp(),
        "task_type": payload.task_type,
        "task_id": str(payload.task_id),
        "page_id": str(page.id),
        "album_id": str(page.album_id),
        "page": {
            "original_filename": page.original_filename,
            "storage_path": page.storage_path,
            "sha256": page.file_hash_sha256,
            "status": page.status.value,
            "photo_count_revision": (page.processing_metadata or {}).get("photo_count_revision"),
        },
        "provenance": provenance,
        "photos": [_photo_snapshot(photo, provenance) for photo in context.get("photos", [])],
    }
    if context.get("ocr") is not None:
        ocr: OCRResult = context["ocr"]
        snapshot["ocr"] = {
            "id": str(ocr.id),
            "text_content": ocr.text_content,
            "text_type": ocr.text_type.value if ocr.text_type else None,
            "bounding_box": deepcopy(ocr.bounding_box),
            "confidence": ocr.confidence,
            "is_verified": ocr.is_verified,
        }
    return snapshot


def _decision_record(event_id: UUID, payload: MobileReviewAnswer, *, state: str) -> dict:
    return {
        "event_id": str(event_id),
        "answer": payload.answer,
        "state": state,
        "reason_code": payload.reason_code,
        "reviewer": payload.reviewer,
        "recorded_at": _timestamp(),
    }


def _store_entity_decision(metadata: dict, task_type: str, decision: dict) -> dict:
    updated = deepcopy(metadata or {})
    decisions = deepcopy(updated.get("review_decisions") or {})
    history = list(decisions.get(task_type) or [])
    history.append(decision)
    decisions[task_type] = history
    updated["review_decisions"] = decisions
    return updated


def _validate_idempotent_replay(
    existing: ReviewEvent, payload: MobileReviewAnswer, principal: Principal | None
) -> None:
    existing_metadata = existing.event_metadata or {}
    existing_tenant = existing_metadata.get("tenant_id")
    existing_client = existing_metadata.get("client")
    if existing_client is None:
        # Schema-v1 events stored client fields at the top level. Preserve
        # replay compatibility while still requiring the original payload.
        existing_client = {key: value for key, value in existing_metadata.items() if key != "tenant_id"}
    if principal is not None and existing_tenant not in {None, str(principal.tenant_id)}:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Review event not found.")
    if (
        existing.task_type != payload.task_type
        or existing.task_id != payload.task_id
        or existing.answer != payload.answer
        or existing.reason_code != payload.reason_code
        or existing.reviewer != payload.reviewer
        or existing_client != payload.metadata
    ):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Idempotency key was already used for a different review decision.",
        )


def _validate_enhancement_decision(payload: MobileReviewAnswer, context: dict) -> None:
    if payload.task_type != "enhancement":
        return

    photo: ExtractedPhoto = context["photo"]
    variants = variants_from_metadata(photo.enhancement_applied)
    candidate_name = str(payload.metadata.get("candidate_name") or "")
    blind_choice = payload.metadata.get("blind_choice")
    source_names = {"original", "source_corrected", "source_corrected_untrimmed", "source_corrected_trimmed"}

    if candidate_name in source_names or not isinstance(variants.get(candidate_name), dict):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Enhancement review candidate is missing or is not an available derived variant.",
        )
    expected_choices = {"candidate", "original"} if payload.answer in {"left", "right"} else {payload.answer}
    if blind_choice not in expected_choices:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Enhancement review metadata does not match the submitted blind answer.",
        )


def _group_for_review(payload: MobileReviewAnswer, context: dict) -> list[ExtractedPhoto]:
    size, trace_count, mask_key, request_key, review_reason, name = GROUP_EXCLUSIONS[payload.answer]
    if name == "crossing" and payload.reason_code != "non_photo_object":
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="crossing_confirmation_required")
    submitted = payload.metadata.get(request_key)
    if not isinstance(submitted, list) or len(submitted) != size or not all(isinstance(photo_id, str) for photo_id in submitted):
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=f"{name}_group_selection_invalid")
    selected = [photo for photo in context["photos"] if str(photo.id) in submitted]
    evidence = (selected[0].segmentation_mask or {}).get(mask_key) if selected else None
    group = {
        photo.id for photo in context["photos"]
        if photo.segmentation_confidence is not None
        and (photo.segmentation_mask or {}).get(mask_key) == evidence
    }
    if (context["page"].status not in {PageStatus.completed, PageStatus.review_needed}
            or not isinstance(evidence, list) or len(evidence) != trace_count or len(selected) != size
            or group != {photo.id for photo in selected}
            or any(
                review_reason not in ((photo.enhancement_applied or {}).get("segmentation_review_reasons") or [])
                or variant_path_from_photo(photo, "full_print") or variant_path_from_photo(photo, "image_only")
                for photo in selected
            )):
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=f"{name}_review_group_stale")
    if name == "crossing":
        modes = ((selected[0].enhancement_applied or {}).get("correction") or {}).get("crop_modes") or {}
        if (any(not isinstance(trace, list) or len(trace) != 2 for trace in evidence)
                or any((modes.get(mode) or {}).get("status") != "review_needed" for mode in ("full_print", "image_only"))):
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="crossing_review_group_stale")
    return selected


def _confirmed_paper_grid_fragments(payload: MobileReviewAnswer, context: dict) -> list[ExtractedPhoto]:
    if payload.reason_code != "same_physical_print":
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="paper_grid_confirmation_required")
    parent_id = payload.metadata.get("paper_grid_parent_id")
    fragment_ids = payload.metadata.get("paper_grid_fragment_ids")
    if (not isinstance(parent_id, str) or not isinstance(fragment_ids, list) or not 1 <= len(fragment_ids) <= 2
            or not all(isinstance(photo_id, str) for photo_id in fragment_ids)
            or len(set(fragment_ids)) != len(fragment_ids) or parent_id in fragment_ids):
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="paper_grid_group_selection_invalid")

    active = {str(photo.id): photo for photo in context["photos"]}
    claims = {photo_id: mask for photo_id, photo in active.items() if photo.segmentation_confidence is not None
              and isinstance(mask := (photo.segmentation_mask or {}).get("paper_grid_review"), dict)}
    parent_claim = claims.get(parent_id) or {}
    outline = parent_claim.get("outline")
    group = {photo_id: claim for photo_id, claim in claims.items() if claim.get("outline") == outline}
    if (context["page"].status not in {PageStatus.completed, PageStatus.review_needed}
            or not isinstance(outline, list) or len(outline) != 4 or parent_claim.get("cut") is not None
            or set(group) != {parent_id, *fragment_ids}
            or any(not isinstance(group[photo_id].get("cut"), list) or len(group[photo_id]["cut"]) != 2
                   for photo_id in fragment_ids)):
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="paper_grid_review_group_stale")

    for photo_id in group:
        photo = active[photo_id]
        metadata = photo.enhancement_applied or {}
        modes = (metadata.get("correction") or {}).get("crop_modes") or {}
        if (PAPER_GRID_REVIEW not in (metadata.get("segmentation_review_reasons") or [])
                or (photo_id == parent_id and len((modes.get("full_print") or {}).get("suggested_warp_quad") or []) != 4)
                or any((modes.get(mode) or {}).get("status") != "review_needed" for mode in ("full_print", "image_only"))):
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="paper_grid_review_group_stale")
    return [active[photo_id] for photo_id in fragment_ids]


def _record_rejected_detection(photo: ExtractedPhoto, metadata: dict) -> None:
    metadata["review_rejected_detection"] = {
        "bounding_box": deepcopy(photo.bounding_box), "segmentation_mask": deepcopy(photo.segmentation_mask),
    }


def _apply_review_decision(event: ReviewEvent, payload: MobileReviewAnswer, context: dict) -> None:
    if payload.task_type == "geometry":
        photo: ExtractedPhoto = context["photo"]
        previously_rejected = _rejected_detection(photo)
        accepted = payload.answer == "correct"
        state = "accepted" if accepted else ("uncertain" if payload.answer == "unsure" else "rejected")
        metadata = _store_entity_decision(
            photo.enhancement_applied or {},
            "geometry",
            _decision_record(event.id, payload, state=state),
        )
        if payload.answer == "wrong_object":
            _record_rejected_detection(photo, metadata)
            if not previously_rejected:
                page: Page = context["page"]
                retire_source_fragment_proposals(page, fragment=(photo.segmentation_mask or {}).get("polygon"))
                page.processing_metadata = {**(page.processing_metadata or {}), "photo_count_revision": str(event.id)}
        elif not previously_rejected:
            metadata.pop("review_rejected_detection", None)
        reasons = [
            str(reason)
            for reason in metadata.get("review_reasons") or []
            if not str(reason).startswith("GEOMETRY_REVIEW_")
        ]
        if accepted:
            correction = metadata.get("correction") or {}
            photo.status = (
                PhotoStatus.completed
                if correction.get("status") != "rejected" and photo.is_duplicate_of is None
                else PhotoStatus.review_needed
            )
        else:
            reasons.append(f"GEOMETRY_REVIEW_{payload.answer.upper()}")
            photo.status = PhotoStatus.review_needed
        metadata["review_reasons"] = list(dict.fromkeys(reasons))
        photo.enhancement_applied = metadata
        return

    if payload.task_type == "segmentation":
        page: Page = context["page"]
        if payload.answer in COUNT_GROUP_ACTIONS:
            decision = {**_decision_record(event.id, payload, state="rejected"), "answer": "wrong_object"}
            for photo in context["excluded_group"]:
                metadata = _store_entity_decision(photo.enhancement_applied or {}, "geometry", decision)
                _record_rejected_detection(photo, metadata)
                reasons = [
                    reason for reason in metadata.get("review_reasons") or []
                    if not str(reason).startswith("GEOMETRY_REVIEW_")
                ]
                metadata["review_reasons"] = list(dict.fromkeys([*reasons, "GEOMETRY_REVIEW_WRONG_OBJECT"]))
                photo.enhancement_applied = metadata
                photo.status = PhotoStatus.review_needed
            page.processing_metadata = {**(page.processing_metadata or {}), "photo_count_revision": str(event.id)}
            return
        state = (
            "accepted" if payload.answer == "correct" else ("uncertain" if payload.answer == "unsure" else "rejected")
        )
        metadata = _store_entity_decision(
            page.processing_metadata or {},
            "segmentation",
            _decision_record(event.id, payload, state=state),
        )
        fusion = page_fusion_evidence(metadata)
        if payload.answer == "correct" and "deferred_print_proposals" in fusion:
            fusion.pop("deferred_print_proposals")
            photos = context["photos"]
            if (page.status == PageStatus.review_needed and not page.error_message
                    and page.blur_score is not None and page.blur_score >= get_settings().blur_threshold
                    and photos and all(photo.status == PhotoStatus.completed for photo in photos)):
                page.status = PageStatus.completed
        page.processing_metadata = metadata
        return

    if payload.task_type == "enhancement":
        photo = context["photo"]
        state = (
            "uncertain" if payload.answer == "unsure" else ("rejected" if payload.answer == "both_bad" else "selected")
        )
        metadata = _store_entity_decision(
            photo.enhancement_applied or {},
            "enhancement",
            _decision_record(event.id, payload, state=state)
            | {
                "blind_choice": payload.metadata.get("blind_choice"),
                "candidate_name": payload.metadata.get("candidate_name"),
            },
        )
        if payload.answer == "both_bad":
            reasons = list(metadata.get("review_reasons") or [])
            reasons.append("ENHANCEMENT_REVIEW_BOTH_BAD")
            metadata["review_reasons"] = list(dict.fromkeys(reasons))
            photo.status = PhotoStatus.review_needed
        variants = deepcopy(metadata.get("variants") or {})
        source_name = "source_corrected" if isinstance(variants.get("source_corrected"), dict) else "original"
        blind_choice = payload.metadata.get("blind_choice")
        candidate_name = str(payload.metadata.get("candidate_name") or "")
        selected_name = None
        if blind_choice == "candidate" and isinstance(variants.get(candidate_name), dict):
            selected_name = candidate_name
        elif blind_choice in {"original", "equal", "both_bad"} or payload.answer in {"equal", "both_bad"}:
            selected_name = source_name
        if selected_name is not None and isinstance(variants.get(selected_name), dict):
            for name, record in variants.items():
                if isinstance(record, dict):
                    record["selected"] = name == selected_name
            metadata["variants"] = variants
            metadata["selected_variant"] = selected_name
            selected_path = variants[selected_name].get("path")
            if selected_path:
                photo.storage_path = str(selected_path)
        photo.enhancement_applied = metadata
        return

    ocr: OCRResult = context["ocr"]
    if payload.answer == "correct":
        ocr.is_verified = True
    elif payload.answer == "incorrect":
        ocr.is_verified = False


@router.post("/events", status_code=status.HTTP_201_CREATED)
async def add_review_event(
    payload: MobileReviewAnswer,
    session: AsyncSession = Depends(get_async_session),
    principal: Principal | None = Depends(require_write_principal),
):
    if payload.answer not in ALLOWED_ANSWERS[payload.task_type]:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Unsupported answer for review task type."
        )
    context = await _task_context(
        session, payload.task_type, payload.task_id, principal, lock_photos=payload.answer in COUNT_GROUP_ACTIONS,
    )
    _validate_enhancement_decision(payload, context)
    existing = await session.scalar(select(ReviewEvent).where(ReviewEvent.idempotency_key == payload.idempotency_key))
    if existing is not None:
        _validate_idempotent_replay(existing, payload, principal)
        return {"id": existing.id, "saved": True, "duplicate": True}

    if payload.task_type == "segmentation":
        metadata = context["page"].processing_metadata or {}
        page_revision = metadata.get("photo_count_revision")
        if payload.metadata.get("photo_count_revision") != page_revision:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Die Abzüge auf dieser Seite haben sich geändert. Prüfe die aktuelle Anzahl.",
            )
        pending = page_fusion_evidence(metadata).get("deferred_print_proposals") or []
        if (payload.answer == "correct" and pending
                and payload.metadata.get("deferred_proposal_ids") != [candidate["id"] for candidate in pending]):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Prüfe die gestrichelten Konturen, bevor du die Anzahl bestätigst.",
            )
        if payload.answer in GROUP_EXCLUSIONS:
            context["excluded_group"] = _group_for_review(payload, context)
        elif payload.answer == CONFIRM_SHARED_PAPER_GRID:
            count_confirmed = await session.scalar(select(_completed_review_exists("segmentation", Page.id)).where(
                Page.id == payload.task_id,
            ))
            if count_confirmed:
                raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="paper_grid_review_group_stale")
            context["excluded_group"] = _confirmed_paper_grid_fragments(payload, context)

    if payload.task_type == "geometry" and payload.answer == "correct":
        photo = context["photo"]
        correction = (photo.enhancement_applied or {}).get("correction") or {}
        full_mode = (correction.get("crop_modes") or {}).get("full_print")
        mark = ((photo.enhancement_applied or {}).get("review_provenance") or {}).get("reviewed_album_mark")
        page = context["page"]
        if mark and (not get_settings().enable_protected_region_guard or page is None
                     or not album_mark_source_matches(mark, page.file_hash_sha256, page.processing_metadata)):
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="album_page_mark_review_stale")
        if full_mode and not variant_path_from_photo(photo, "full_print"):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Adjust the print corners or recapture the page before approving this crop.",
            )
        if motif_geometry_requires_review(correction):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="motif_geometry_review_required",
            )

    snapshot = await _snapshot_task(session, payload, context)
    client_metadata = deepcopy(payload.metadata)
    event_metadata = {
        "schema_version": 2,
        "tenant_id": str(principal.tenant_id) if principal else None,
        "client": client_metadata,
        "snapshot": snapshot,
    }
    # Preserve the small set of legacy keys consumed by existing local reports.
    for key in ("ui", "page_id", "blind_choice", "candidate_name", "difficulty_tags"):
        if key in client_metadata:
            event_metadata[key] = deepcopy(client_metadata[key])
    event = ReviewEvent(
        id=uuid4(),
        created_at=datetime.now(timezone.utc),
        task_type=payload.task_type,
        task_id=payload.task_id,
        answer=payload.answer,
        reason_code=payload.reason_code,
        idempotency_key=payload.idempotency_key,
        reviewer=payload.reviewer,
        event_metadata=event_metadata,
    )
    session.add(event)
    _apply_review_decision(event, payload, context)
    try:
        await session.commit()
    except IntegrityError:
        await session.rollback()
        existing = await session.scalar(
            select(ReviewEvent).where(ReviewEvent.idempotency_key == payload.idempotency_key)
        )
        if existing is not None:
            _validate_idempotent_replay(existing, payload, principal)
        return {"id": existing.id if existing else None, "saved": True, "duplicate": True}
    return {"id": event.id, "saved": True, "duplicate": False}


@router.get("/progress")
async def review_progress(
    session: AsyncSession = Depends(get_async_session),
    principal: Principal | None = Depends(get_principal),
):
    query = select(ReviewEvent)
    if principal is not None:
        query = query.where(ReviewEvent.event_metadata["tenant_id"].as_string() == str(principal.tenant_id))
    events = (await session.scalars(query)).all()
    by_type: dict[str, int] = {}
    for event in events:
        by_type[event.task_type] = by_type.get(event.task_type, 0) + 1
    return {"completed": len(events), "by_type": by_type}


def _completed_review_exists(task_type: str, task_id_column):
    query = select(ReviewEvent.answer).where(
        ReviewEvent.task_type == task_type,
        ReviewEvent.task_id == task_id_column,
    )
    if task_type == "segmentation":
        snapshot_revision = ReviewEvent.event_metadata["snapshot"]["page"]["photo_count_revision"].as_string()
        page_revision = Page.processing_metadata["photo_count_revision"].as_string()
        query = query.where(func.coalesce(snapshot_revision, "") == func.coalesce(page_revision, ""))
    latest_answer = (
        query
        .order_by(ReviewEvent.created_at.desc(), ReviewEvent.id.desc())
        .limit(1)
        .correlate_except(ReviewEvent)
        .scalar_subquery()
    )
    return func.coalesce(latest_answer, "").in_(COMPLETING_ANSWERS[task_type])


async def _geometry_review_is_current(session: AsyncSession, photo: ExtractedPhoto) -> bool:
    metadata = photo.enhancement_applied or {}
    correction = metadata.get("correction") or {}
    full_mode = (correction.get("crop_modes") or {}).get("full_print") or {}
    if (any(reason in PAPER_CLAIM_OVERLAP_CHECKS for reason in (full_mode.get("reason_codes") or []))
            or motif_geometry_requires_review(correction)):
        return False
    if paper_claims_need_overlap_check(photo.segmentation_mask or {}, metadata, automatic=photo.segmentation_confidence is not None):
        return False
    event = await session.scalar(
        select(ReviewEvent)
        .where(
            ReviewEvent.task_type == "geometry",
            ReviewEvent.task_id == photo.id,
        )
        .order_by(ReviewEvent.created_at.desc(), ReviewEvent.id.desc())
        .limit(1)
    )
    if event is None or event.answer != "correct":
        return False
    snapshot_photos = ((event.event_metadata or {}).get("snapshot") or {}).get("photos") or []
    if not snapshot_photos:
        # Legacy accepted events had no revision snapshot. Preserve their
        # completion semantics until the geometry is edited again.
        return not metadata.get("review_provenance", {}).get("correction_revisions")
    snap = next((item for item in snapshot_photos if item.get("photo_id") == str(photo.id)), snapshot_photos[0])
    geometry = snap.get("current_geometry") or {}
    return (
        geometry.get("bounding_box") == photo.bounding_box
        and geometry.get("segmentation_mask") == photo.segmentation_mask
        and not motif_geometry_requires_review(snap.get("current_correction"))
    )


def _first_url(urls: dict[str, str | None], names: tuple[str, ...]) -> str | None:
    return next((urls.get(name) for name in names if urls.get(name)), None)


def _photo_review_payload(photo: ExtractedPhoto | PhotoRead) -> dict:
    metadata = deepcopy(photo.enhancement_applied or {})
    mask = deepcopy(photo.segmentation_mask or {})
    correction = deepcopy(metadata.get("correction") or {})
    visible_polygon = deepcopy(mask.get("visible_polygon") or mask.get("polygon"))
    warp_quad = deepcopy(
        correction.get("warp_quad")
        or correction.get("source_quad")
        or mask.get("warp_quad")
        or (mask.get("polygon") if len(mask.get("polygon") or []) == 4 else None)
    )
    urls = variant_urls(photo)
    variants = variants_from_metadata(metadata)
    review_variants = {
        name: {
            "url": urls.get(name),
            "selected": bool(record.get("selected")),
            "metrics": deepcopy(record.get("metrics") or {}),
            "warnings": list(record.get("warnings") or []),
            "gates": deepcopy(record.get("gates") or []),
            "model_chain": deepcopy(record.get("model_chain") or []),
            "params": deepcopy(record.get("params") or {}),
        }
        for name, record in variants.items()
        if isinstance(record, dict)
    }
    confidence_components = deepcopy(metadata.get("confidence_components") or {})
    confidence_components.setdefault("detection_confidence", photo.segmentation_confidence)
    for name in (
        "boundary_confidence",
        "geometry_confidence",
        "perspective_confidence",
        "orientation_confidence",
        "border_trim_confidence",
    ):
        if name not in confidence_components and correction.get(name) is not None:
            confidence_components[name] = correction[name]
    return {
        "photo_id": str(photo.id),
        "bounding_box": deepcopy(photo.bounding_box),
        "segmentation_mask": mask,
        "polygon": visible_polygon,
        "visible_polygon": visible_polygon,
        "warp_quad": warp_quad,
        "image_quad": deepcopy((correction.get("crop_modes") or {}).get("image_only", {}).get("warp_quad") or mask.get("image_quad")),
        "print_size_mm": deepcopy(mask.get("print_size_mm")),
        "crop_modes": deepcopy(correction.get("crop_modes") or {}),
        "motif_review_pending": motif_geometry_requires_review(correction),
        "confidence": photo.segmentation_confidence,
        "confidence_components": confidence_components,
        "geometry_valid": photo.geometry_valid,
        "correction": correction,
        "orientation": deepcopy(correction.get("orientation") or metadata.get("orientation") or {}),
        "review_reasons": list(metadata.get("review_reasons") or []),
        "source_corrected_url": _first_url(urls, ("source_corrected", "original")),
        "untrimmed_url": _first_url(urls, ("source_corrected_untrimmed", "untrimmed", "original_untrimmed")),
        "trimmed_url": _first_url(urls, ("source_corrected_trimmed", "trimmed", "source_corrected", "original")),
        "variant_urls": urls,
        "variants": review_variants,
    }


@router.get("/queue", response_model=list[ReviewItem])
async def get_review_queue(
    album_id: UUID | None = None,
    photo_id: UUID | None = None,
    review_type: str = Query(default="segmentation", alias="type", pattern="^(segmentation|ocr|geometry|enhancement)$"),
    limit: int = Query(default=25, ge=1, le=100),
    include_rejected: bool = True,
    session: AsyncSession = Depends(get_async_session),
    principal: Principal | None = Depends(get_principal),
) -> list[ReviewItem]:
    items: list[ReviewItem] = []
    if review_type == "segmentation":
        reviewed = _completed_review_exists("segmentation", Page.id)
        query = select(Page).where(Page.status.in_([PageStatus.review_needed, PageStatus.completed]), ~reviewed)
        if album_id:
            query = query.where(Page.album_id == album_id)
        if principal is not None:
            query = query.where(Page.album.has(tenant_id=principal.tenant_id))
        pages = (
            await session.scalars(
                query.order_by(
                    case((Page.status == PageStatus.review_needed, 0), else_=1), Page.created_at.asc()
                ).limit(limit)
            )
        ).all()
        for page in pages:
            photos = (
                await session.scalars(
                    select(ExtractedPhoto)
                    .where(ExtractedPhoto.page_id == page.id)
                    .order_by(ExtractedPhoto.created_at.asc())
                )
            ).all()
            detections = [_photo_review_payload(photo) for photo in photos if not _rejected_detection(photo)]
            segmentation = ((page.processing_metadata or {}).get("steps") or {}).get("segmentation") or {}
            items.append(
                ReviewItem(
                    item_type="segmentation",
                    id=page.id,
                    album_id=page.album_id,
                    page_id=page.id,
                    status=page.status.value,
                    reason=page.error_message,
                    preview_url=f"/api/v1/pages/{page.id}/image",
                    payload={
                        "detection_count": len(detections), "detections": detections,
                        "photo_count_revision": (page.processing_metadata or {}).get("photo_count_revision"),
                        "missing_print_edge_traces": segmentation.get("missing_print_edge_traces", []),
                        "deferred_print_proposals": page_fusion_evidence(page.processing_metadata).get("deferred_print_proposals", []),
                    },
                )
            )
    elif review_type == "geometry":
        query = select(ExtractedPhoto)
        if album_id:
            query = query.where(ExtractedPhoto.album_id == album_id)
        if photo_id:
            query = query.where(ExtractedPhoto.id == photo_id)
        if principal is not None:
            query = query.where(ExtractedPhoto.album.has(tenant_id=principal.tenant_id))
        priority = case(
            (ExtractedPhoto.status == PhotoStatus.review_needed, 0),
            (ExtractedPhoto.geometry_valid.is_(False), 0),
            else_=1,
        )
        candidate_photos = (
            await session.scalars(
                query.order_by(
                    priority, ExtractedPhoto.segmentation_confidence.asc().nullsfirst(), ExtractedPhoto.created_at.asc()
                )
            )
        ).all()
        views = await photo_views_for_current_album_marks(session, list(candidate_photos))
        for photo, view in zip(candidate_photos, views, strict=True):
            if not include_rejected and _rejected_detection(photo):
                continue
            if view is photo and await _geometry_review_is_current(session, photo):
                continue
            payload = _photo_review_payload(view)
            review_status = PhotoStatus.review_needed if payload["motif_review_pending"] else view.status
            items.append(
                ReviewItem(
                    item_type="geometry",
                    id=photo.id,
                    album_id=photo.album_id,
                    page_id=photo.page_id,
                    status=review_status.value,
                    reason="crop requires confirmation" if review_status == PhotoStatus.review_needed else "quality-control sample",
                    preview_url=f"/api/v1/pages/{photo.page_id}/image",
                    payload=payload
                    | {"crop_url": payload["source_corrected_url"] if photo.original_storage_path else None},
                )
            )
            if len(items) >= limit:
                break
    elif review_type == "ocr":
        reviewed = _completed_review_exists("ocr", OCRResult.id)
        query = (
            select(OCRResult, Page.album_id)
            .join(Page, Page.id == OCRResult.page_id)
            .where(OCRResult.is_verified.is_(False), ~reviewed)
        )
        if album_id:
            query = query.where(Page.album_id == album_id)
        if principal is not None:
            query = query.where(Page.album.has(tenant_id=principal.tenant_id))
        rows = (await session.execute(query.order_by(OCRResult.created_at.asc()).limit(limit))).all()
        for ocr, row_album_id in rows:
            items.append(
                ReviewItem(
                    item_type="ocr",
                    id=ocr.id,
                    album_id=row_album_id,
                    page_id=ocr.page_id,
                    status="unverified",
                    reason=ocr.spatial_classification_reason,
                    preview_url=f"/api/v1/pages/{ocr.page_id}/image",
                    payload={"bounding_box": deepcopy(ocr.bounding_box), "confidence": ocr.confidence},
                )
            )
    else:
        reviewed = _completed_review_exists("enhancement", ExtractedPhoto.id)
        query = select(ExtractedPhoto).where(ExtractedPhoto.original_storage_path.is_not(None), ~reviewed)
        if album_id:
            query = query.where(ExtractedPhoto.album_id == album_id)
        if principal is not None:
            query = query.where(ExtractedPhoto.album.has(tenant_id=principal.tenant_id))
        photos = (await session.scalars(query.order_by(ExtractedPhoto.created_at.asc()).limit(limit * 4))).all()
        for photo in photos:
            if _rejected_detection(photo):
                continue
            payload = _photo_review_payload(photo)
            candidate_name = next(
                (
                    name
                    for name in (
                        "conservative_enhanced",
                        "restoration_candidate",
                        "face_restoration_candidate",
                        "premium",
                        "conservative",
                        "enhanced",
                    )
                    if payload["variants"].get(name)
                ),
                None,
            )
            if candidate_name is None:
                continue
            items.append(
                ReviewItem(
                    item_type="enhancement",
                    id=photo.id,
                    album_id=photo.album_id,
                    page_id=photo.page_id,
                    status=photo.status.value,
                    reason="blind comparison",
                    preview_url=payload["source_corrected_url"],
                    payload=payload
                    | {
                        "original_url": payload["source_corrected_url"],
                        "candidate_url": payload["variant_urls"].get(candidate_name),
                        "candidate_name": candidate_name,
                        "degradation": deepcopy((photo.enhancement_applied or {}).get("degradation", {})),
                    },
                )
            )
            if len(items) >= limit:
                break
    return items
