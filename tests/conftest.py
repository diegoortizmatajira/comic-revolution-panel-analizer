from __future__ import annotations

import io
from typing import TYPE_CHECKING

import pytest
from app.domain.models import BoundingBox, DetectedPanel
from PIL import Image, ImageDraw

if TYPE_CHECKING:
    from collections.abc import Sequence

Rect = tuple[int, int, int, int]  # x, y, width, height


def make_page(
    rects: Sequence[Rect], size: tuple[int, int] = (800, 1200), fmt: str = "PNG"
) -> bytes:
    """White page with dark-filled rectangles standing in for panels."""
    img = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(img)
    for x, y, w, h in rects:
        draw.rectangle([x, y, x + w - 1, y + h - 1], fill=(60, 60, 60))
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


def panel(x: int, y: int, w: int, h: int, confidence: float = 1.0) -> DetectedPanel:
    return DetectedPanel(BoundingBox(x, y, w, h), confidence)


@pytest.fixture
def page_2x2() -> bytes:
    return make_page(
        [(40, 40, 340, 540), (420, 40, 340, 540), (40, 620, 340, 540), (420, 620, 340, 540)]
    )
