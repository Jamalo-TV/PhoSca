from uuid import UUID

from app.config import get_settings
from app.services.pipeline import (
    deduplication_task_logic,
    enhancement_task_logic,
    ingestion_task_logic,
    ocr_task_logic,
    persistence_task_logic,
    perspective_correction_task_logic,
    process_pages_pipeline,
    process_page_pipeline,
    run_async,
    segmentation_task_logic,
)
from app.database import get_session_factory
from app.tasks.celery_app import celery_app


@celery_app.task(name="pipeline.ingestion")
def ingestion_task(page_id: str) -> dict:
    async def _run() -> dict:
        async with get_session_factory()() as session:
            return await ingestion_task_logic(session, UUID(page_id), get_settings())

    return run_async(_run())


@celery_app.task(name="pipeline.segmentation")
def segmentation_task(page_id: str) -> dict:
    async def _run() -> dict:
        async with get_session_factory()() as session:
            return await segmentation_task_logic(session, UUID(page_id), get_settings())

    return run_async(_run())


@celery_app.task(name="pipeline.perspective_correction")
def perspective_correction_task(photo_id: str) -> dict:
    async def _run() -> dict:
        async with get_session_factory()() as session:
            return await perspective_correction_task_logic(session, UUID(photo_id), get_settings())

    return run_async(_run())


@celery_app.task(name="pipeline.ocr")
def ocr_task(page_id: str) -> dict:
    async def _run() -> dict:
        async with get_session_factory()() as session:
            return await ocr_task_logic(session, UUID(page_id), get_settings())

    return run_async(_run())


@celery_app.task(name="pipeline.enhancement")
def enhancement_task(photo_id: str) -> dict:
    async def _run() -> dict:
        async with get_session_factory()() as session:
            return await enhancement_task_logic(session, UUID(photo_id), get_settings())

    return run_async(_run())


@celery_app.task(name="pipeline.deduplication")
def deduplication_task(photo_id: str) -> dict:
    async def _run() -> dict:
        async with get_session_factory()() as session:
            return await deduplication_task_logic(session, UUID(photo_id))

    return run_async(_run())


@celery_app.task(name="pipeline.persistence")
def persistence_task(photo_id: str) -> dict:
    async def _run() -> dict:
        async with get_session_factory()() as session:
            return await persistence_task_logic(session, UUID(photo_id))

    return run_async(_run())


@celery_app.task(name="pipeline.process_page")
def process_page_task(page_id: str) -> dict:
    return run_async(process_page_pipeline(UUID(page_id), get_settings()))


@celery_app.task(name="pipeline.process_pages")
def process_pages_task(page_ids: list[str]) -> dict:
    return run_async(process_pages_pipeline([UUID(page_id) for page_id in page_ids], get_settings()))

