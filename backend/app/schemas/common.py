from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator


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


class PrintSizeMM(StrictSchema):
    """Physical lengths of the numbered width and height edges, in millimetres."""

    width: float = Field(gt=0.0, allow_inf_nan=False)
    height: float = Field(gt=0.0, allow_inf_nan=False)

    @property
    def aspect_ratio(self) -> float:
        return self.width / self.height

    @model_validator(mode="after")
    def validate_aspect_ratio(self) -> "PrintSizeMM":
        if not 0.1 <= self.aspect_ratio <= 10:
            raise ValueError("Print width to height ratio must be between 0.1 and 10.")
        return self


class SegmentationMask(StrictSchema):
    polygon: list[NormalizedPoint] = Field(min_length=4)
    source: str | None = Field(default=None, max_length=80)

    def as_dict(self) -> dict:
        return self.model_dump(exclude_none=True)


class JobQueued(BaseModel):
    job_id: UUID | str
    status: str
    estimated_pages: int
