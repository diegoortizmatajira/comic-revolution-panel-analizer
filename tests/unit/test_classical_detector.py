import numpy as np
from app.services.detection.classical import ClassicalPanelDetector
from app.services.image.decoder import ImageLimits, decode_image

from tests.conftest import make_page

LIMITS = ImageLimits(max_bytes=10_000_000, max_pixels=10_000_000)


def test_detects_four_panels(page_2x2: bytes) -> None:
    gray = decode_image(page_2x2, LIMITS).pixels
    panels = ClassicalPanelDetector().detect(gray).panels
    boxes = sorted((p.bbox.x, p.bbox.y, p.bbox.width, p.bbox.height) for p in panels)
    assert boxes == [
        (40, 40, 340, 540),
        (40, 620, 340, 540),
        (420, 40, 340, 540),
        (420, 620, 340, 540),
    ]
    assert all(p.confidence > 0.95 for p in panels)


def test_blank_page_has_no_panels() -> None:
    assert ClassicalPanelDetector().detect(np.full((200, 100, 3), 255, dtype=np.uint8)).panels == ()


def test_ignores_specks_below_min_area() -> None:
    gray = decode_image(make_page([(10, 10, 20, 20)]), LIMITS).pixels
    assert ClassicalPanelDetector().detect(gray).panels == ()
