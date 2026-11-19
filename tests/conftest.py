from __future__ import annotations

import io
from typing import TYPE_CHECKING

import numpy as np
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


def make_comic_page(
    rects: Sequence[Rect],
    size: tuple[int, int] = (800, 1200),
    background: tuple[int, int, int] = (255, 255, 255),
    noise: float = 0.0,
) -> np.ndarray:
    """RGB page with black-bordered panels filled with a vertical colour gradient."""
    width, height = size
    page = np.zeros((height, width, 3), dtype=np.float32)
    page[:] = background
    for x, y, w, h in rects:
        gradient = np.linspace(90, 230, h, dtype=np.float32)[:, None]
        page[y : y + h, x : x + w] = np.stack(
            [gradient * 1.0, gradient * 0.6, gradient * 0.3], axis=2
        )
        page[y : y + 6, x : x + w] = 0  # black frame
        page[y + h - 6 : y + h, x : x + w] = 0
        page[y : y + h, x : x + 6] = 0
        page[y : y + h, x + w - 6 : x + w] = 0
    if noise:
        rng = np.random.default_rng(0)
        page += rng.normal(0, noise, page.shape).astype(np.float32)
    return np.clip(page, 0, 255).astype(np.uint8)


def panel(x: int, y: int, w: int, h: int, confidence: float = 1.0) -> DetectedPanel:
    return DetectedPanel(BoundingBox(x, y, w, h), confidence)


@pytest.fixture
def page_2x2() -> bytes:
    return make_page(
        [(40, 40, 340, 540), (420, 40, 340, 540), (40, 620, 340, 540), (420, 620, 340, 540)]
    )
