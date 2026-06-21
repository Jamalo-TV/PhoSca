from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class StrictSchema(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class BoundingBox(StrictSchema):
    x1: float = Field(ge=0.0, le=1.0)
    y1: float = Field(ge=0.0, le=1.0)
    x2: float = Field(ge=0.0, le=1.0)
    y2: float = Field(ge=0.0, le=1.0)

    def as_dict(self) -> dict[str, float]:
        return self.model_dump()


class NormalizedPoint(StrictSchema):
    x: float = Field(ge=0.0, le=1.0)
    y: float = Field(ge=0.0, le=1.0)


class SegmentationMask(StrictSchema):
    polygon: list[NormalizedPoint] = Field(min_length=4)
    source: str | None = Field(default=None, max_length=80)

    def as_dict(self) -> dict:
        return self.model_dump(exclude_none=True)


class JobQueued(BaseModel):
    job_id: UUID | str
    status: str
    estimated_pages: int
