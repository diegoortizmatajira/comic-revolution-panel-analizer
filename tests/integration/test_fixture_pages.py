"""End-to-end detection on committed pages (public-domain scans plus synthetic edge cases).

Where the detector is correct the tests assert it. Where it is known to fall short the test is a
strict xfail, so the suite stays green today and fails loudly (XPASS) when detection improves.
"""

from pathlib import Path

import cv2
import numpy as np
import pytest
from app.domain.models import AnalysisResult, ImageSize, ReadingDirection
from app.services.analyzer import PanelAnalyzer
from app.services.detection.projection import ProjectionPanelDetector
from app.services.image.decoder import DecodedImage, ImageLimits, decode_image
from app.services.ordering.xy_cut import XYCutOrdering

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
LIMITS = ImageLimits(max_bytes=10_000_000, max_pixels=40_000_000)
analyzer = PanelAnalyzer(ProjectionPanelDetector(), XYCutOrdering())

PUBLIC_DOMAIN_PAGES = ["nature_boy_4.jpg", "western_fighters_6.jpg", "western_fighters_8.jpg"]


def analyze_file(name: str, direction: ReadingDirection = ReadingDirection.LTR) -> AnalysisResult:
    image = decode_image((FIXTURES / name).read_bytes(), LIMITS)
    return analyzer.analyze(image, direction)


def analyze_array(
    page: np.ndarray, direction: ReadingDirection = ReadingDirection.LTR
) -> AnalysisResult:
    height, width = page.shape[:2]
    return analyzer.analyze(DecodedImage(page, ImageSize(width, height)), direction)


def assert_well_formed(result: AnalysisResult) -> None:
    assert result.panels, "at least one panel (or the whole-page fallback) is always returned"
    assert [p.order for p in result.panels] == list(range(1, len(result.panels) + 1))
    assert 0.0 <= result.sequence_confidence <= 1.0
    for p in result.panels:
        assert 0.0 <= p.confidence <= 1.0
        n = p.normalized_bbox
        assert 0.0 <= n.x <= 1.0
        assert 0.0 <= n.y <= 1.0
        assert n.x + n.width <= 1.0 + 1e-9
        assert n.y + n.height <= 1.0 + 1e-9


@pytest.mark.parametrize("name", PUBLIC_DOMAIN_PAGES)
@pytest.mark.parametrize("direction", list(ReadingDirection))
def test_public_domain_pages_yield_well_formed_results(
    name: str, direction: ReadingDirection
) -> None:
    assert_well_formed(analyze_file(name, direction))


def test_western_fighters_8_separates_top_and_both_bottom_panels() -> None:
    result = analyze_file("western_fighters_8.jpg")
    top, bottom_left, bottom_right = result.panels
    assert top.bbox.y < 100
    assert bottom_left.bbox.x < 50
    assert bottom_right.bbox.x > 400
    assert abs(bottom_left.bbox.y - bottom_right.bbox.y) < 20
    assert result.sequence_confidence == 1.0


def test_western_fighters_8_rtl_reads_bottom_right_before_bottom_left() -> None:
    result = analyze_file("western_fighters_8.jpg", ReadingDirection.RTL)
    top, first_bottom, second_bottom = result.panels
    assert top.bbox.y < 100
    assert first_bottom.bbox.x > second_bottom.bbox.x


@pytest.mark.xfail(strict=True, reason="two bottom panels separated by a narrow gutter stay merged")
def test_nature_boy_4_finds_the_three_panels() -> None:
    assert len(analyze_file("nature_boy_4.jpg").panels) >= 3


@pytest.mark.xfail(
    strict=True, reason="borderless panel fading into the paper defeats gutter cutting"
)
def test_western_fighters_6_finds_the_three_panels() -> None:
    assert len(analyze_file("western_fighters_6.jpg").panels) >= 3


# --- synthetic edge cases ---------------------------------------------------------------------


def splash_page() -> np.ndarray:
    """Full-bleed art with no margin at all."""
    y, x = np.mgrid[0:1200, 0:800]
    return np.stack([(x / 800 * 255), (y / 1200 * 255), np.full_like(x, 120)], axis=2).astype(
        np.uint8
    )


def borderless_page() -> np.ndarray:
    """Coloured panels with no frame lines, separated only by white gutters."""
    page = np.full((1200, 800, 3), 255, dtype=np.uint8)
    for (x, y, w, h), colour in [
        ((40, 40, 720, 500), (200, 90, 60)),
        ((40, 580, 340, 580), (60, 140, 200)),
        ((420, 580, 340, 580), (90, 180, 90)),
    ]:
        page[y : y + h, x : x + w] = colour
    return page


def diagonal_page() -> np.ndarray:
    """Two panels split by a diagonal gutter: no clean row or column exists."""
    page = np.full((1200, 800, 3), 255, dtype=np.uint8)
    cv2.fillPoly(page, [np.array([[40, 40], [760, 40], [760, 420], [40, 700]])], (60, 90, 200))
    cv2.fillPoly(page, [np.array([[40, 760], [760, 480], [760, 1160], [40, 1160]])], (200, 90, 60))
    return page


def test_splash_page_yields_a_single_full_page_panel() -> None:
    result = analyze_array(splash_page())
    assert_well_formed(result)
    assert len(result.panels) == 1
    assert (result.panels[0].normalized_bbox.width, result.panels[0].normalized_bbox.height) == (
        1.0,
        1.0,
    )


def test_borderless_panels_are_separated_by_white_gutters() -> None:
    result = analyze_array(borderless_page())
    assert_well_formed(result)
    assert [(p.bbox.x, p.bbox.y) for p in result.panels] == [(40, 40), (40, 580), (420, 580)]


@pytest.mark.xfail(strict=True, reason="diagonal gutters cannot be cut by rows/columns")
def test_diagonal_gutter_panels_are_separated() -> None:
    assert len(analyze_array(diagonal_page()).panels) == 2


def test_diagonal_gutter_degrades_gracefully() -> None:
    result = analyze_array(diagonal_page())
    assert_well_formed(result)  # never an error: worst case is one larger panel
