from app.services.detection.frames import FrameLines
from app.services.detection.margin import margin_box, normalize_unframed

WIDTH, HEIGHT = 1000, 1400


def framed_page(rects: list[tuple[int, int, int, int]]) -> FrameLines:
    """Frame lines on all four sides of every rect given as (x0, y0, x1, y1)."""
    horizontal = [(y, x0, x1) for x0, y0, x1, y1 in rects for y in (y0, y1)]
    vertical = [(x, y0, y1) for x0, y0, x1, y1 in rects for x in (x0, x1)]
    return FrameLines(horizontal, vertical, HEIGHT, WIDTH)


BODY = (40, 300, 960, 800)
LEFT, RIGHT = (30, 850, 480, 1350), (520, 850, 970, 1350)  # the margin box is x 30..970
TITLE = (60, 60, 900, 250)  # no frame lines around it


def test_margin_box_is_the_bounding_box_of_the_framed_panels() -> None:
    frames = framed_page([BODY, LEFT, RIGHT])
    assert margin_box([TITLE, BODY, LEFT, RIGHT], frames) == (30, 300, 970, 1350)


def test_a_title_takes_the_margin_edges_and_ends_where_the_first_panel_begins() -> None:
    frames = framed_page([BODY, LEFT, RIGHT])
    title = normalize_unframed([TITLE, BODY, LEFT, RIGHT], frames)[0]
    assert title == (30, 60, 970, 300)


def test_a_footer_begins_where_the_last_panel_ends() -> None:
    frames = framed_page([BODY, LEFT, RIGHT])
    footer = (50, 1370, 950, 1395)
    normalized = normalize_unframed([BODY, LEFT, RIGHT, footer], frames)[3]
    assert normalized == (30, 1350, 970, 1395)


def test_framed_panels_and_panels_that_are_not_banners_are_left_alone() -> None:
    frames = framed_page([BODY, LEFT, RIGHT])
    tall = (60, 60, 300, 280)  # unframed, but not wide and short: just a panel with weak frames
    rects = [tall, BODY, LEFT, RIGHT]
    assert normalize_unframed(rects, frames) == rects


def test_no_margin_box_without_two_framed_panels() -> None:
    frames = framed_page([BODY])  # one framed panel is no basis for a layout
    assert margin_box([TITLE, BODY], frames) is None
    assert normalize_unframed([TITLE, BODY], frames) == [TITLE, BODY]


def test_an_edge_far_from_the_margin_box_is_not_stretched() -> None:
    frames = framed_page([BODY, LEFT, RIGHT])
    narrow = (400, 60, 600, 100)  # a 200-wide banner far from both margin edges
    assert normalize_unframed([narrow, BODY, LEFT, RIGHT], frames)[0][0:3:2] == (400, 600)
