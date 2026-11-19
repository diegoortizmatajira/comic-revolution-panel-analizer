"""Immutable domain value objects. No framework or I/O dependencies."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class ReadingDirection(StrEnum):
    LTR = "ltr"
    RTL = "rtl"


@dataclass(frozen=True, slots=True)
class ImageSize:
    width: int
    height: int

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            msg = "image dimensions must be positive"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class BoundingBox:
    """Axis-aligned box in pixel coordinates (origin top-left)."""

    x: int
    y: int
    width: int
    height: int

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            msg = "bounding box dimensions must be positive"
            raise ValueError(msg)

    @property
    def right(self) -> int:
        return self.x + self.width

    @property
    def bottom(self) -> int:
        return self.y + self.height

    @property
    def area(self) -> int:
        return self.width * self.height


@dataclass(frozen=True, slots=True)
class NormalizedBox:
    """Bounding box relative to image size; every value lies within 0-1."""

    x: float
    y: float
    width: float
    height: float

    @classmethod
    def from_box(cls, box: BoundingBox, size: ImageSize) -> NormalizedBox:
        return cls(
            x=box.x / size.width,
            y=box.y / size.height,
            width=box.width / size.width,
            height=box.height / size.height,
        )


@dataclass(frozen=True, slots=True)
class DetectedPanel:
    bbox: BoundingBox
    confidence: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            msg = "confidence must be within [0, 1]"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class Detection:
    """What a detector found on one page and which method actually produced it.

    A composite detector (hybrid) reports the component that answered, so `method` is per page,
    not per detector instance.
    """

    panels: tuple[DetectedPanel, ...]
    method: str


@dataclass(frozen=True, slots=True)
class OrderedPanel:
    id: int
    order: int
    bbox: BoundingBox
    normalized_bbox: NormalizedBox
    confidence: float


@dataclass(frozen=True, slots=True)
class OrderingResult:
    panels: tuple[DetectedPanel, ...]
    confidence: float


@dataclass(frozen=True, slots=True)
class AnalysisResult:
    image_size: ImageSize
    reading_direction: ReadingDirection
    panels: tuple[OrderedPanel, ...]
    sequence_confidence: float
    method: str
    version: str
    whole_page_fallback: bool = False
    warnings: tuple[str, ...] = field(default_factory=tuple)
