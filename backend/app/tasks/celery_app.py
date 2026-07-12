from celery import Celery

from app.config import get_settings

settings = get_settings()

celery_app = Celery(
    "album_digitizer",
    broker=settings.redis_url,
    backend=settings.redis_url,
    include=["app.tasks.health", "app.tasks.pipeline"],
)
celery_app.conf.update(
    task_track_started=True,
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    task_always_eager=settings.celery_task_always_eager,
    task_store_eager_result=settings.celery_task_always_eager,
    task_routes={
        "pipeline.ingestion": {"queue": "cpu"},
        "pipeline.perspective_correction": {"queue": "cpu"},
        "pipeline.ocr": {"queue": "analysis_gpu"},
        "pipeline.segmentation": {"queue": "analysis_gpu"},
        "pipeline.enhancement": {"queue": "restoration_gpu"},
        "pipeline.deduplication": {"queue": "cpu"},
        "pipeline.persistence": {"queue": "cpu"},
        "pipeline.process_page": {"queue": "cpu"},
        "pipeline.process_pages": {"queue": "cpu"},
    },
    task_default_queue="cpu",
    worker_prefetch_multiplier=1,
    task_acks_late=True,
    task_reject_on_worker_lost=True,
)
