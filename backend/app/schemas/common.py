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


class JobQueued(BaseModel):
    job_id: UUID | str
    status: str
    estimated_pages: int

