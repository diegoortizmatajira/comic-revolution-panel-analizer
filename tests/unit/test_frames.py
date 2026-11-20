import numpy as np
from app.services.detection.frames import FrameLines, _merge


def lines(
    horizontal: list[tuple[int, int, int]], height: int = 1000, width: int = 800
) -> FrameLines:
    return FrameLines(horizontal=horizontal, vertical=[], height=height, width=width)


def test_merge_joins_collinear_fragments_across_a_gap() -> None:
    assert _merge([(100, 0, 200), (102, 260, 500)], max_gap=80, tolerance=5) == [(101, 0, 500)]


def test_merge_keeps_lines_at_different_positions_apart() -> None:
    assert len(_merge([(100, 0, 200), (140, 0, 200)], max_gap=80, tolerance=5)) == 2


def test_merge_joins_groups_that_only_touch_once_both_have_grown() -> None:
    fragments = [(100, 0, 100), (110, 400, 500), (104, 90, 410)]
    assert _merge(fragments, max_gap=10, tolerance=12) == [(106, 0, 500)]


def test_cut_rows_lists_lines_that_cover_enough_of_the_region_longest_first() -> None:
    frames = lines([(300, 0, 400), (500, 0, 700), (520, 0, 100)])
    assert frames.cut_rows((0, 1000, 0, 800)) == [500, 300]


def test_cut_rows_ignores_lines_near_the_region_edge() -> None:
    assert lines([(20, 0, 800)]).cut_rows((0, 1000, 0, 800)) == []


def test_sides_counts_framed_sides() -> None:
    frames = FrameLines(
        horizontal=[(100, 0, 800), (600, 0, 800)],
        vertical=[(0, 100, 600)],
        height=1000,
        width=800,
    )
    assert frames.sides((100, 600, 0, 800)) == 3
    assert frames.sides((700, 900, 0, 800)) == 0


def test_find_reports_straight_frame_lines_of_a_drawn_page() -> None:
    page = np.full((600, 800, 3), 255, dtype=np.uint8)
    page[300:306, 40:760] = 0  # a horizontal frame line
    page[100:500, 40:46] = 0  # a vertical one
    found = FrameLines.find(page)
    assert any(abs(y - 303) <= 4 and x1 - x0 > 600 for y, x0, x1 in found.horizontal)
    assert any(abs(x - 43) <= 4 and y1 - y0 > 300 for x, y0, y1 in found.vertical)
