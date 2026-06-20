from app.tasks.celery_app import celery_app


@celery_app.task(name="health.check")
def health_check() -> dict[str, str]:
    return {"status": "ok"}

