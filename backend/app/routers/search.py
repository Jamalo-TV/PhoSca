from fastapi import APIRouter, Depends, Query
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_async_session
from app.models import OCRResult
from app.schemas import SearchResult

router = APIRouter(prefix="/api/v1/search", tags=["search"])


def _highlight(text: str, query: str) -> str:
    lower = text.lower()
    index = lower.find(query.lower())
    if index == -1:
        return text[:160]
    start = max(0, index - 60)
    end = min(len(text), index + len(query) + 60)
    return text[start:end]


@router.get("", response_model=list[SearchResult])
async def search_ocr(
    q: str = Query(min_length=1, max_length=200),
    limit: int = Query(default=50, ge=1, le=100),
    session: AsyncSession = Depends(get_async_session),
) -> list[SearchResult]:
    dialect = session.get_bind().dialect.name
    query = select(OCRResult)
    if dialect == "postgresql":
        vector = func.to_tsvector("english", OCRResult.text_content)
        ts_query = func.plainto_tsquery("english", q)
        query = query.where(vector.op("@@")(ts_query)).order_by(func.ts_rank(vector, ts_query).desc())
    else:
        query = query.where(OCRResult.text_content.ilike(f"%{q}%")).order_by(OCRResult.created_at.desc())
    rows = (await session.scalars(query.limit(limit))).all()
    return [
        SearchResult(
            photo_id=row.photo_id,
            page_id=row.page_id,
            text_content=row.text_content,
            text_type=row.text_type,
            confidence=row.confidence,
            highlight=_highlight(row.text_content, q),
        )
        for row in rows
    ]
