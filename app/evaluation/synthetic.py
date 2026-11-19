"""Generated pages with exact ground truth, covering layouts that are hard for detectors.

Real pages are labelled by hand; these complement them with cases whose truth is known by
construction, so a failure here is always a detector failure, never a labelling doubt.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np

from app.domain.models import ImageSize, NormalizedBox, ReadingDirection
from app.evaluation.labels import LabeledPage
from app.services.image.decoder import DecodedImage

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

Rect = tuple[int, int, int, int]  # x, y, width, height in pixels
RGB = tuple[int, int, int]

WHITE: RGB = (255, 255, 255)
MAROON: RGB = (79, 18, 18)
DEFAULT_SIZE = (800, 1200)
FRAME = 6


def draw_comic_page(
    rects: Sequence[Rect],
    size: tuple[int, int] = DEFAULT_SIZE,
    background: RGB = WHITE,
    noise: float = 0.0,
    *,
    frames: bool = True,
) -> NDArray[np.uint8]:
    """RGB page whose panels are gradient-filled, optionally with black frames."""
    width, height = size
    page = np.zeros((height, width, 3), dtype=np.float32)
    page[:] = background
    for x, y, w, h in rects:
        gradient = np.linspace(90, 230, h, dtype=np.float32)[:, None]
        page[y : y + h, x : x + w] = np.stack(
            [gradient * 1.0, gradient * 0.6, gradient * 0.3], axis=2
        )
        if frames:
            page[y : y + FRAME, x : x + w] = 0
            page[y + h - FRAME : y + h, x : x + w] = 0
            page[y : y + h, x : x + FRAME] = 0
            page[y : y + h, x + w - FRAME : x + w] = 0
    if noise:
        rng = np.random.default_rng(0)
        page += rng.normal(0, noise, page.shape).astype(np.float32)
    return np.clip(page, 0, 255).astype(np.uint8)


def _page(
    page_id: str,
    pixels: NDArray[np.uint8],
    truth: Sequence[Rect],
    direction: ReadingDirection = ReadingDirection.LTR,
    *tags: str,
) -> LabeledPage:
    height, width = pixels.shape[:2]
    size = ImageSize(width=width, height=height)
    return LabeledPage(
        id=f"synthetic/{page_id}",
        image=DecodedImage(pixels=pixels, size=size),
        direction=direction,
        panels=tuple(
            NormalizedBox(x / width, y / height, w / width, h / height) for x, y, w, h in truth
        ),
        tags=frozenset({"synthetic", *tags}),
        source="synthetic",
    )


GRID: list[Rect] = [
    (40, 40, 340, 540),
    (420, 40, 340, 540),
    (40, 620, 340, 540),
    (420, 620, 340, 540),
]
MIXED: list[Rect] = [
    (40, 40, 220, 480),
    (290, 40, 470, 220),
    (290, 290, 470, 230),
    (40, 560, 340, 420),
    (410, 560, 350, 420),
]


def _diagonal() -> NDArray[np.uint8]:
    page = np.full((1200, 800, 3), 255, dtype=np.uint8)
    cv2.fillPoly(page, [np.array([[40, 40], [760, 40], [760, 420], [40, 700]])], (60, 90, 200))
    cv2.fillPoly(page, [np.array([[40, 760], [760, 480], [760, 1160], [40, 1160]])], (200, 90, 60))
    return page


def _splash() -> NDArray[np.uint8]:
    y, x = np.mgrid[0:1200, 0:800]
    return np.stack([x / 800 * 255, y / 1200 * 255, np.full_like(x, 120)], axis=2).astype(np.uint8)


def _borderless() -> NDArray[np.uint8]:
    page = np.full((1200, 800, 3), 255, dtype=np.uint8)
    for (x, y, w, h), colour in [
        ((40, 40, 720, 500), (200, 90, 60)),
        ((40, 580, 340, 580), (60, 140, 200)),
        ((420, 580, 340, 580), (90, 180, 90)),
    ]:
        page[y : y + h, x : x + w] = colour
    return page


def synthetic_pages() -> list[LabeledPage]:
    rtl = ReadingDirection.RTL
    spread_left: list[Rect] = [(30, 40, 470, 520), (30, 600, 470, 540)]
    spread_right: list[Rect] = [(560, 40, 470, 520), (560, 600, 470, 540)]
    manga: list[Rect] = [
        (430, 40, 330, 560),
        (40, 40, 350, 270),
        (40, 340, 350, 260),
        (40, 640, 720, 520),
    ]
    return [
        _page("grid_white", draw_comic_page(GRID), GRID, ReadingDirection.LTR, "clean"),
        _page(
            "mixed_dark",
            draw_comic_page(MIXED, background=MAROON),
            MIXED,
            ReadingDirection.LTR,
            "dark",
        ),
        _page(
            "mixed_dark_noisy",
            draw_comic_page(MIXED, background=MAROON, noise=8.0),
            MIXED,
            ReadingDirection.LTR,
            "dark",
            "noisy",
        ),
        _page(
            "bleed_top",
            draw_comic_page(
                [(0, 0, 800, 500), (40, 540, 340, 300), (420, 540, 340, 300)], size=(800, 900)
            ),
            [(0, 0, 800, 500), (40, 540, 340, 300), (420, 540, 340, 300)],
            ReadingDirection.LTR,
            "bleed",
        ),
        _page(
            "borderless",
            _borderless(),
            [(40, 40, 720, 500), (40, 580, 340, 580), (420, 580, 340, 580)],
            ReadingDirection.LTR,
            "borderless",
        ),
        _page("splash", _splash(), [(0, 0, 800, 1200)], ReadingDirection.LTR, "splash"),
        _page(
            "diagonal_gutter",
            _diagonal(),
            [(40, 40, 720, 660), (40, 480, 720, 680)],
            ReadingDirection.LTR,
            "diagonal",
        ),
        _page(
            "no_gutter",
            draw_comic_page([(40, 40, 340, 500), (380, 40, 340, 500)]),
            [(40, 40, 340, 500), (380, 40, 340, 500)],
            ReadingDirection.LTR,
            "no-gutter",
        ),
        _page(
            "manga_rtl",
            draw_comic_page(manga),
            [manga[0], manga[1], manga[2], manga[3]],
            rtl,
            "rtl",
        ),
        _page(
            "spread_ltr",
            draw_comic_page([*spread_left, *spread_right], size=(1060, 1180)),
            [*spread_left, *spread_right],
            ReadingDirection.LTR,
            "spread",
        ),
        _page(
            "spread_rtl",
            draw_comic_page([*spread_left, *spread_right], size=(1060, 1180)),
            [*spread_right, *spread_left],
            rtl,
            "spread",
            "rtl",
        ),
    ]
