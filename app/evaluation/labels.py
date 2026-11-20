"""Ground-truth label format and corpus discovery.

A labelled page is an image plus a sibling ``<name>.labels.json``:

    {
      "version": 1,
      "reading_direction": "ltr",
      "verified": true,
      "tags": ["borderless"],
      "notes": "free text",
      "panels": [{"x": 0.05, "y": 0.03, "width": 0.9, "height": 0.3}, ...]
    }

``caption_panels`` (optional) is the same list with the lettering above each frame included.
``panels`` are boxes normalized to the image (0-1, origin top-left), listed in reading order
for ``reading_direction``. Draft labels written by ``--propose`` carry ``"verified": false``
and are ignored until a person has checked and corrected them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.domain.models import NormalizedBox, ReadingDirection
from app.services.image.decoder import DecodedImage, ImageLimits, decode_image

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence
    from pathlib import Path

IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".webp"})
LABELS_SUFFIX = ".labels.json"
LIMITS = ImageLimits(max_bytes=50_000_000, max_pixels=100_000_000)
_EPSILON = 1e-6


class PanelLabel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    x: float = Field(ge=0, le=1)
    y: float = Field(ge=0, le=1)
    width: float = Field(gt=0, le=1)
    height: float = Field(gt=0, le=1)

    @model_validator(mode="after")
    def _inside_page(self) -> Self:
        if self.x + self.width > 1 + _EPSILON or self.y + self.height > 1 + _EPSILON:
            msg = "panel extends beyond the page"
            raise ValueError(msg)
        return self

    def to_box(self) -> NormalizedBox:
        return NormalizedBox(x=self.x, y=self.y, width=self.width, height=self.height)


class PageLabels(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: Literal[1] = 1
    reading_direction: ReadingDirection = ReadingDirection.LTR
    verified: bool = False
    tags: list[str] = Field(default_factory=list)
    notes: str = ""
    panels: list[PanelLabel]
    # Optional: the same panels, in the same order, with the lettering that hangs above each frame
    # included. Scored instead of ``panels`` when caption claiming is on.
    caption_panels: list[PanelLabel] | None = None

    @model_validator(mode="after")
    def _same_panels_with_captions(self) -> Self:
        if self.caption_panels is not None and len(self.caption_panels) != len(self.panels):
            msg = "caption_panels must list the same panels as panels"
            raise ValueError(msg)
        return self


@dataclass(frozen=True, slots=True)
class LabeledPage:
    id: str
    image: DecodedImage
    direction: ReadingDirection
    panels: tuple[NormalizedBox, ...]  # ground truth, in reading order
    tags: frozenset[str]
    source: str  # "file" or "synthetic"
    caption_panels: tuple[NormalizedBox, ...] | None = None  # ...with top captions, if labelled


def labels_path(image: Path) -> Path:
    return image.with_name(image.stem + LABELS_SUFFIX)


def read_labels(path: Path) -> PageLabels:
    return PageLabels.model_validate_json(path.read_text())


def write_labels(path: Path, labels: PageLabels) -> None:
    path.write_text(labels.model_dump_json(indent=2) + "\n")


def image_files(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


def discover(corpus_dirs: Iterable[Path], *, include_unverified: bool = False) -> list[LabeledPage]:
    """Load every labelled page found in the given folders (missing folders are skipped)."""
    pages: list[LabeledPage] = []
    for folder in corpus_dirs:
        if not folder.is_dir():
            continue
        for image in image_files(folder):
            label_file = labels_path(image)
            if not label_file.exists():
                continue
            labels = read_labels(label_file)
            if not labels.verified and not include_unverified:
                continue
            pages.append(_load(folder, image, labels))
    return pages


def _load(folder: Path, image: Path, labels: PageLabels) -> LabeledPage:
    decoded = decode_image(image.read_bytes(), LIMITS)
    return LabeledPage(
        id=f"{folder.name}/{image.stem}",
        image=decoded,
        direction=labels.reading_direction,
        panels=tuple(p.to_box() for p in labels.panels),
        caption_panels=(
            None
            if labels.caption_panels is None
            else tuple(p.to_box() for p in labels.caption_panels)
        ),
        tags=frozenset(labels.tags),
        source="file",
    )


def to_labels(
    boxes: Sequence[NormalizedBox], direction: ReadingDirection, *, verified: bool
) -> PageLabels:
    return PageLabels(
        reading_direction=direction,
        verified=verified,
        panels=[PanelLabel(x=b.x, y=b.y, width=b.width, height=b.height) for b in boxes],
    )
