from fastapi import APIRouter

from app.tasks.celery_app import celery_app

router = APIRouter(prefix="/api/v1/jobs", tags=["jobs"])


@router.get("/{job_id}")
async def get_job(job_id: str) -> dict:
    result = celery_app.AsyncResult(job_id)
    payload: dict = {"job_id": job_id, "status": result.status.lower()}
    if result.ready():
        if result.successful():
            payload["result"] = result.result
        else:
            payload["error"] = str(result.result)
    return payload

