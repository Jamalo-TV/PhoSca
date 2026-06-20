from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from app.models import AlbumStatus
from app.schemas.common import StrictSchema


class AlbumCreate(StrictSchema):
    name: str = Field(min_length=1, max_length=255)
    description: str | None = Field(default=None, max_length=10_000)


class AlbumRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    name: str
    description: str | None
    directory_path: str | None
    status: AlbumStatus
    total_pages: int
    processed_pages: int
