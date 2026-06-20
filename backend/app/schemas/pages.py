from uuid import UUID

from pydantic import BaseModel, ConfigDict

from app.models import PageStatus


class PageUploadItem(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    page_id: UUID
    filename: str
    status: PageStatus


class PageUploadResponse(BaseModel):
    pages: list[PageUploadItem]

