"""HTTP request/response models (the public API contract)."""

from __future__ import annotations

from typing import Annotated, Any, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.domain.models import AnalysisResult, ReadingDirection


class ImageSource(BaseModel):
    """Exactly one of `url` or `base64` must be provided."""

    model_config = ConfigDict(extra="forbid")

    url: str | None = Field(default=None, description="http(s) URL the service can reach.")
    base64: str | None = Field(default=None, description="Base64 image, or a `data:` URI.")

    @model_validator(mode="after")
    def _exactly_one(self) -> Self:
        if (self.url is None) == (self.base64 is None):
            msg = "Provide exactly one of 'url' or 'base64'."
            raise ValueError(msg)
        return self


class AnalyzeOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reading_direction: ReadingDirection = ReadingDirection.LTR


class AnalyzeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    image: ImageSource
    options: AnalyzeOptions = Field(default_factory=AnalyzeOptions)


class ImageSizeOut(BaseModel):
    width: int
    height: int


class BoxOut(BaseModel):
    x: int
    y: int
    width: int
    height: int


class NormalizedBoxOut(BaseModel):
    x: float
    y: float
    width: float
    height: float


UnitInterval = Annotated[float, Field(ge=0, le=1)]


class PanelOut(BaseModel):
    id: int
    order: int
    bbox: BoxOut
    bbox_normalized: NormalizedBoxOut
    confidence: UnitInterval


class AnalyzeResponse(BaseModel):
    image: ImageSizeOut
    reading_direction: ReadingDirection
    panels: list[PanelOut]
    sequence_confidence: UnitInterval
    method: str
    warnings: list[str]

    @classmethod
    def from_result(cls, result: AnalysisResult) -> AnalyzeResponse:
        return cls(
            image=ImageSizeOut(width=result.image_size.width, height=result.image_size.height),
            reading_direction=result.reading_direction,
            panels=[
                PanelOut(
                    id=p.id,
                    order=p.order,
                    bbox=BoxOut(x=p.bbox.x, y=p.bbox.y, width=p.bbox.width, height=p.bbox.height),
                    bbox_normalized=NormalizedBoxOut(
                        x=p.normalized_bbox.x,
                        y=p.normalized_bbox.y,
                        width=p.normalized_bbox.width,
                        height=p.normalized_bbox.height,
                    ),
                    confidence=p.confidence,
                )
                for p in result.panels
            ],
            sequence_confidence=result.sequence_confidence,
            method=result.method,
            warnings=list(result.warnings),
        )


class ErrorBody(BaseModel):
    code: str
    message: str
    details: dict[str, Any]
    request_id: str | None


class HealthResponse(BaseModel):
    status: str
