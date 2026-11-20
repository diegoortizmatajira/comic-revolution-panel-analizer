from pathlib import Path

import cv2
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
    return sorted(
        (p.bbox.x, p.bbox.y, p.bbox.width, p.bbox.height) for p in detector.detect(page).panels
    )


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
    panels = detector.detect(make_comic_page(REAL_LAYOUT, background=MAROON)).panels
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
    assert detector.detect(np.full((300, 200, 3), 255, dtype=np.uint8)).panels == ()


def test_ignores_specks_below_min_area() -> None:
    assert boxes(make_comic_page([(20, 20, 30, 30)])) == []


def test_panels_with_no_gutter_stay_merged() -> None:
    # Documented limitation: without a gutter there is nothing to cut along.
    found = boxes(make_comic_page([(40, 40, 340, 500), (380, 40, 340, 500)]))
    assert len(found) == 1


def test_confidence_is_within_range() -> None:
    for p in detector.detect(make_comic_page(REAL_LAYOUT, background=MAROON)).panels:
        assert 0.0 <= p.confidence <= 1.0


SAMPLES = Path(__file__).resolve().parents[2] / "samples"


@pytest.mark.skipif(not (SAMPLES / "001.webp").exists(), reason="local sample not present")
def test_real_sample_001_is_split_into_separate_panels() -> None:
    # Local, git-ignored sample (dark page, jagged hand-drawn frames). The contour
    # detector found 2 panels here; the projection detector separates at least 4.
    image = decode_image((SAMPLES / "001.webp").read_bytes(), ImageLimits(10**8, 10**8))
    assert len(detector.detect(image.pixels).panels) >= 4


def test_slanted_gutter_is_cut_along_its_slope() -> None:
    page = np.full((1200, 800, 3), 255, dtype=np.uint8)
    cv2.fillPoly(page, [np.array([[40, 40], [760, 40], [760, 420], [40, 700]])], (60, 90, 200))
    cv2.fillPoly(page, [np.array([[40, 760], [760, 480], [760, 1160], [40, 1160]])], (200, 90, 60))
    assert len(detector.detect(page).panels) == 2


def wobbly_gutter_page(*, bridged: bool) -> np.ndarray:
    """Two dark-framed panels split by a thin, slightly slanted, jagged gutter."""
    height, width = 700, 1100
    page = np.full((height, width, 3), MAROON, dtype=np.uint8)
    rng = np.random.default_rng(1)
    rows = np.arange(height)
    centre = 520 + 0.015 * rows + np.cumsum(rng.integers(-1, 2, height)) * 0.3
    for y in rows:
        left, right = int(centre[y]) - 5, int(centre[y]) + 5
        page[y, 20:left] = (200, 120, 60)
        page[y, right : width - 20] = (60, 140, 200)
    page[:20] = page[-20:] = MAROON
    if bridged:  # art from both panels touches across the gutter for a few rows
        for y in range(300, 310):
            page[y, int(centre[y]) - 6 : int(centre[y]) + 6] = (20, 20, 20)
    return page


@pytest.mark.parametrize("bridged", [False, True])
def test_wobbly_gutter_splits_even_when_art_bridges_it(bridged: bool) -> None:
    assert len(detector.detect(wobbly_gutter_page(bridged=bridged)).panels) == 2


def test_wobbly_gutter_does_not_split_a_single_panel() -> None:
    page = np.full((700, 1100, 3), MAROON, dtype=np.uint8)
    page[20:680, 20:1080] = (200, 120, 60)
    assert len(detector.detect(page).panels) == 1


def banner_page() -> np.ndarray:
    """Title lettering above two framed panels that touch the lettering's baseline."""
    page = np.full((1200, 800, 3), 240, dtype=np.uint8)
    rng = np.random.default_rng(3)
    for x in range(70, 650, 40):  # overlapping letters of uneven height
        page[int(rng.integers(80, 150)) : 230, x : x + 60] = (30, 30, 200)
    page[230:236, 40:760] = 0  # the top frame line sits right under the lettering
    page[236:1160, 40:760] = (200, 170, 120)
    page[700:706, 40:760] = 0  # a frame line between the two stacked panels
    page[236:1160, 40:46] = 0
    page[236:1160, 754:760] = 0
    page[1154:1160, 40:760] = 0
    return page


def test_title_lettering_above_a_frame_line_is_not_a_panel() -> None:
    found = boxes(banner_page())
    assert found, "the framed panels are still found"
    assert all(y >= 225 for _, y, _, _ in found)


def test_a_strip_between_two_panels_is_shared_between_them() -> None:
    kept = [(0, 100, 0, 100), (200, 300, 0, 100)]  # y0, y1, x0, x1
    strip = (100, 200, 0, 100)
    above, below = ProjectionPanelDetector._attach_strips(kept, [strip], 10, 5_000)
    assert above == (0, 140, 0, 100)
    assert below == (140, 300, 0, 100)


def test_a_small_strip_at_the_top_goes_to_the_panel_below_but_a_banner_is_dropped() -> None:
    panel = [(200, 300, 0, 100)]
    small, banner = (150, 200, 0, 100), (0, 200, 0, 100)
    assert ProjectionPanelDetector._attach_strips(panel, [small], 10, 20_000) == [
        (150, 300, 0, 100)
    ]
    assert ProjectionPanelDetector._attach_strips(panel, [banner], 10, 20_000) == panel


def test_full_bleed_art_touching_the_next_row_is_a_panel_not_a_banner() -> None:
    page = np.full((1200, 800, 3), 255, dtype=np.uint8)
    ramp = np.linspace(60, 200, 420, dtype=np.uint8)[:, None]
    page[0:420] = np.stack([ramp, ramp // 2, np.full_like(ramp, 90)], axis=2)  # art off the page
    page[420:426, 60:740] = 0  # the next row's top frame, with no gutter above it
    page[426:1140, 60:740] = (200, 170, 120)
    page[426:1140, 60:66] = page[426:1140, 734:740] = 0
    page[1134:1140, 60:740] = 0
    found = boxes(page)
    assert any(y == 0 and w >= 780 for _, y, w, _ in found), found
