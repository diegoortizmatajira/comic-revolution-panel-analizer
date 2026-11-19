import numpy as np
from app.domain.models import ImageSize, ReadingDirection
from app.services.analyzer import PanelAnalyzer
from app.services.detection.refine import EdgeSnapRefiner
from app.services.image.decoder import DecodedImage
from app.services.ordering.xy_cut import XYCutOrdering
from numpy.typing import NDArray

from tests.conftest import panel
from tests.unit.test_analyzer import FakeDetector

WIDTH, HEIGHT = 600, 800


def blank() -> NDArray[np.uint8]:
    return np.full((HEIGHT, WIDTH, 3), 255, dtype=np.uint8)


def ink(page: NDArray[np.uint8], x0: int, y0: int, x1: int, y1: int) -> None:
    page[y0:y1, x0:x1] = 20


def balloon(page: NDArray[np.uint8], x0: int, y0: int, x1: int, y1: int) -> None:
    """Hollow rectangle with a 3 px outline: a white balloon on a white page."""
    ink(page, x0, y0, x1, y0 + 3)
    ink(page, x0, y1 - 3, x1, y1)
    ink(page, x0, y0, x0 + 3, y1)
    ink(page, x1 - 3, y0, x1, y1)


def bounds(panels: tuple) -> tuple[int, int, int, int]:
    box = panels[0].bbox
    return box.x, box.y, box.right, box.bottom


def test_grows_an_edge_that_cuts_through_a_balloon() -> None:
    page = blank()
    ink(page, 100, 100, 500, 400)  # the panel art
    balloon(page, 200, 380, 400, 460)  # a balloon hanging out of the panel's bottom
    refined = EdgeSnapRefiner().refine(page, [panel(100, 100, 400, 330)])  # cut at y=430
    assert bounds(refined) == (100, 100, 500, 460)


def test_grows_in_every_direction_a_balloon_pokes_out() -> None:
    page = blank()
    ink(page, 100, 200, 500, 500)
    balloon(page, 75, 300, 140, 360)  # sticks out left
    balloon(page, 450, 170, 520, 230)  # sticks out top-right
    refined = EdgeSnapRefiner().refine(page, [panel(100, 200, 400, 300)])
    assert bounds(refined) == (75, 170, 520, 500)


def test_leaves_an_edge_that_already_sits_in_a_gutter() -> None:
    page = blank()
    ink(page, 100, 100, 500, 400)
    refined = EdgeSnapRefiner().refine(page, [panel(100, 100, 400, 300)])
    assert bounds(refined) == (100, 100, 500, 400)


def test_never_shrinks_a_box_that_has_slack() -> None:
    page = blank()
    ink(page, 100, 100, 500, 400)
    refined = EdgeSnapRefiner().refine(page, [panel(60, 60, 480, 380)])
    assert bounds(refined) == (60, 60, 540, 440)


def test_does_not_grow_when_no_gutter_is_found_within_reach() -> None:
    page = blank()
    ink(page, 100, 100, 500, 400)
    ink(page, 100, 400, 500, 800)  # art that bleeds on, far past the 6% allowance
    refined = EdgeSnapRefiner().refine(page, [panel(100, 100, 400, 300)])
    assert bounds(refined) == (100, 100, 500, 400)


def test_does_not_run_into_a_neighbouring_panel() -> None:
    page = blank()
    ink(page, 100, 100, 500, 400)
    ink(page, 100, 412, 500, 700)  # neighbour, 12 px below
    refined = EdgeSnapRefiner().refine(page, [panel(100, 100, 400, 290), panel(100, 412, 400, 288)])
    assert refined[0].bbox.bottom <= 412
    assert refined[1].bbox.y == 412


def test_content_that_runs_off_the_page_grows_to_the_border() -> None:
    page = blank()
    ink(page, 100, 100, 500, 400)
    balloon(page, 150, 0, 350, 130)  # a balloon from the page top down into the panel
    refined = EdgeSnapRefiner().refine(page, [panel(100, 100, 400, 300)])
    assert refined[0].bbox.y == 0


def test_the_reach_is_configurable() -> None:
    page = blank()
    ink(page, 100, 300, 500, 600)
    balloon(page, 150, 150, 350, 330)  # 150 px above the panel, i.e. ~19% of the page
    assert EdgeSnapRefiner().refine(page, [panel(100, 300, 400, 300)])[0].bbox.y == 300
    wide = EdgeSnapRefiner(max_expand_ratio=0.25).refine(page, [panel(100, 300, 400, 300)])
    assert wide[0].bbox.y == 150


def test_claims_a_balloon_in_the_gap_that_touches_only_its_own_panel() -> None:
    page = blank()
    ink(page, 100, 100, 500, 300)  # upper panel, with a frame that trails 10 px past its box
    ink(page, 100, 300, 500, 310)
    ink(page, 100, 500, 500, 700)  # lower panel
    balloon(page, 150, 400, 350, 520)  # hangs into the lower panel, 90 px below the frame
    upper, lower = panel(100, 100, 400, 200), panel(100, 500, 400, 200)
    refined = EdgeSnapRefiner().refine(page, [upper, lower])
    assert refined[0].bbox.bottom == 310  # the upper panel only takes its own trailing frame
    assert refined[1].bbox.y == 400  # the lower one does


def test_leaves_a_blob_that_touches_both_neighbours() -> None:
    page = blank()
    ink(page, 100, 100, 500, 300)
    ink(page, 100, 500, 500, 700)
    balloon(page, 150, 300, 350, 500)  # a balloon joining the two panels
    refined = EdgeSnapRefiner().refine(page, [panel(100, 100, 400, 200), panel(100, 500, 400, 200)])
    assert (refined[0].bbox.bottom, refined[1].bbox.y) == (300, 500)


def test_stays_inside_the_page() -> None:
    page = blank()
    ink(page, 0, 0, 300, 300)
    refined = EdgeSnapRefiner().refine(page, [panel(0, 0, 300, 290)])
    assert refined[0].bbox.y == 0
    assert refined[0].bbox.x == 0


def test_no_panels_is_a_no_op() -> None:
    assert EdgeSnapRefiner().refine(blank(), []) == ()


def test_keeps_confidence() -> None:
    page = blank()
    ink(page, 100, 100, 500, 400)
    refined = EdgeSnapRefiner().refine(page, [panel(100, 100, 400, 300, 0.42)])
    assert refined[0].confidence == 0.42


def test_the_analyzer_applies_the_refiner_and_reports_it_in_the_version() -> None:
    page = blank()
    ink(page, 100, 100, 500, 400)
    balloon(page, 200, 380, 400, 460)
    image = DecodedImage(page, ImageSize(WIDTH, HEIGHT))
    detector = FakeDetector([panel(100, 100, 400, 330)])

    plain = PanelAnalyzer(detector, XYCutOrdering()).analyze(image, ReadingDirection.LTR)
    refined = PanelAnalyzer(detector, XYCutOrdering(), EdgeSnapRefiner()).analyze(
        image, ReadingDirection.LTR
    )

    assert plain.panels[0].bbox.bottom == 430
    assert refined.panels[0].bbox.bottom == 460
    assert plain.version == "fake@7+xycut@2"
    assert refined.version == "fake@7+edge-snap@1+xycut@2"
