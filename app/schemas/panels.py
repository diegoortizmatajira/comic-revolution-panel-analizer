"""HTTP request/response models (the public API contract)."""

from __future__ import annotations

from typing import Annotated, Any, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.core import options as opt  # noqa: TC001  (pydantic needs them at runtime)
from app.domain.models import AnalysisResult, ReadingDirection  # noqa: TC001


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
    """Per-request processing options. Leave one out (or null) to use the server's default.

    `GET /v1/panels/options` lists every option with its default, range and description.
    """

    model_config = ConfigDict(extra="forbid")

    reading_direction: opt.ReadingDirectionName | None = None
    detector: opt.DetectorName | None = None
    refine_edges: opt.RefineEdges | None = None
    refine_max_expand: opt.RefineMaxExpand | None = None
    claim_captions: opt.ClaimCaptions | None = None
    color_tolerance: opt.ColorTolerance | None = None
    gutter_content_ratio: opt.GutterContentRatio | None = None
    min_gutter_ratio: opt.MinGutterRatio | None = None
    min_area_ratio: opt.MinAreaRatio | None = None
    hybrid_min_panels: opt.HybridMinPanels | None = None
    hybrid_max_aspect: opt.HybridMaxAspect | None = None
    ml_confidence: opt.MlConfidence | None = None
    ml_nms_iou: opt.MlNmsIou | None = None
    order_gap_tolerance: opt.OrderGapTolerance | None = None

    def overrides(self) -> dict[str, Any]:
        """The options the caller actually set."""
        return self.model_dump(exclude_none=True)


class UploadOptions(AnalyzeOptions):
    """The same options as form fields. Other form parts (the file) are not options, so ignored."""

    model_config = ConfigDict(extra="ignore")


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
    version: str = Field(
        description="Opaque analysis version (detector + ordering). Results are comparable, and "
        "safe to cache, only while this value is unchanged."
    )
    whole_page_fallback: bool = Field(
        description="True when nothing was detected and `panels` holds one whole-page stand-in."
    )
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
            version=result.version,
            whole_page_fallback=result.whole_page_fallback,
            warnings=list(result.warnings),
        )


class OptionInfo(BaseModel):
    type: str
    default: Any
    description: str
    minimum: float | None = None
    maximum: float | None = None
    exclusive_minimum: float | None = None
    enum: list[str] | None = None


class OptionsResponse(BaseModel):
    options: dict[str, OptionInfo]


class ErrorBody(BaseModel):
    code: str
    message: str
    details: dict[str, Any]
    request_id: str | None


class HealthResponse(BaseModel):
    status: str


def describe_options(defaults: dict[str, Any]) -> dict[str, OptionInfo]:
    """Every request option with its default, range and description, from the model's schema."""
    properties = AnalyzeOptions.model_json_schema()["properties"]
    described: dict[str, OptionInfo] = {}
    for name, prop in properties.items():
        spec = next((p for p in prop.get("anyOf", [prop]) if p.get("type") != "null"), prop)
        described[name] = OptionInfo(
            type=spec["type"],
            default=defaults[name],
            description=spec.get("description", ""),
            minimum=spec.get("minimum"),
            maximum=spec.get("maximum"),
            exclusive_minimum=spec.get("exclusiveMinimum"),
            enum=spec.get("enum"),
        )
    return described
