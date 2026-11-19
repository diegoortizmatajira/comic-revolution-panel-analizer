from pathlib import Path

import numpy as np
import pytest
from app.domain.models import ReadingDirection
from app.services.detection.projection import ProjectionPanelDetector
from app.services.image.decoder import ImageLimits, decode_image
from app.services.ordering.xy_cut import XYCutOrdering

from tests.conftest import make_comic_page

detector = ProjectionPanelDetector()
MAROON = (79, 18, 18)

# Layout modelled on a real page: tall left panel, two stacked right, two across the bottom.
REAL_LAYOUT = [
    (40, 40, 220, 480),
    (290, 40, 470, 220),
    (290, 290, 470, 230),
    (40, 560, 340, 420),
    (410, 560, 350, 420),
]


def boxes(page: np.ndarray) -> list[tuple[int, int, int, int]]:
    return sorted((p.bbox.x, p.bbox.y, p.bbox.width, p.bbox.height) for p in detector.detect(page))


def assert_close(actual: tuple[int, ...], expected: tuple[int, ...], tol: int = 3) -> None:
    assert all(abs(a - e) <= tol for a, e in zip(actual, expected, strict=True)), (actual, expected)


def test_grid_on_white_page() -> None:
    rects = [(40, 40, 340, 540), (420, 40, 340, 540), (40, 620, 340, 540), (420, 620, 340, 540)]
    found = boxes(make_comic_page(rects))
    assert len(found) == 4
    for actual, expected in zip(found, sorted(rects), strict=True):
        assert_close(actual, expected)


def test_dark_background_with_black_frames() -> None:
    # Black frames are close to a dark page in grayscale; colour distance still separates them.
    assert len(boxes(make_comic_page(REAL_LAYOUT, background=MAROON))) == 5


def test_noisy_background_and_art() -> None:
    assert len(boxes(make_comic_page(REAL_LAYOUT, background=MAROON, noise=8.0))) == 5


def test_real_layout_reading_order_ltr_and_rtl() -> None:
    panels = detector.detect(make_comic_page(REAL_LAYOUT, background=MAROON))
    ltr = XYCutOrdering().order(panels, ReadingDirection.LTR)
    got = [(p.bbox.x, p.bbox.y) for p in ltr.panels]
    expected = [(40, 40), (290, 40), (290, 290), (40, 560), (410, 560)]
    for actual, exp in zip(got, expected, strict=True):
        assert_close(actual, exp)
    assert ltr.confidence == 1.0
    rtl = XYCutOrdering().order(panels, ReadingDirection.RTL)
    assert rtl.panels[0].bbox.x > 200  # top-right block first in manga order


def test_art_bleeding_off_the_page_edge_does_not_break_background_detection() -> None:
    rects = [(0, 0, 800, 500), (40, 540, 340, 300), (420, 540, 340, 300)]
    found = boxes(make_comic_page(rects, size=(800, 900)))
    assert len(found) == 3


def test_blank_page_has_no_panels() -> None:
    assert detector.detect(np.full((300, 200, 3), 255, dtype=np.uint8)) == []


def test_ignores_specks_below_min_area() -> None:
    assert boxes(make_comic_page([(20, 20, 30, 30)])) == []


def test_panels_with_no_gutter_stay_merged() -> None:
    # Documented limitation: without a gutter there is nothing to cut along.
    found = boxes(make_comic_page([(40, 40, 340, 500), (380, 40, 340, 500)]))
    assert len(found) == 1


def test_confidence_is_within_range() -> None:
    for p in detector.detect(make_comic_page(REAL_LAYOUT, background=MAROON)):
        assert 0.0 <= p.confidence <= 1.0


SAMPLES = Path(__file__).resolve().parents[2] / "samples"


@pytest.mark.skipif(not (SAMPLES / "001.webp").exists(), reason="local sample not present")
def test_real_sample_001_is_split_into_separate_panels() -> None:
    # Local, git-ignored sample (dark page, jagged hand-drawn frames). The contour
    # detector found 2 panels here; the projection detector separates at least 4.
    image = decode_image((SAMPLES / "001.webp").read_bytes(), ImageLimits(10**8, 10**8))
    assert len(detector.detect(image.pixels)) >= 4
