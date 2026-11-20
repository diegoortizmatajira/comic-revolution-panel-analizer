from app.domain.models import DetectedPanel, ReadingDirection
from app.services.ordering.xy_cut import XYCutOrdering
from hypothesis import given
from hypothesis import strategies as st

from tests.conftest import panel

ordering = XYCutOrdering()


def positions(panels: tuple[DetectedPanel, ...]) -> list[tuple[int, int]]:
    return [(p.bbox.x, p.bbox.y) for p in panels]


def test_empty_page() -> None:
    result = ordering.order([], ReadingDirection.LTR)
    assert result.panels == ()
    assert result.confidence == 1.0


def test_grid_ltr() -> None:
    grid = [
        panel(420, 620, 340, 540),
        panel(40, 40, 340, 540),
        panel(420, 40, 340, 540),
        panel(40, 620, 340, 540),
    ]
    result = ordering.order(grid, ReadingDirection.LTR)
    assert positions(result.panels) == [(40, 40), (420, 40), (40, 620), (420, 620)]
    assert result.confidence == 1.0


def test_grid_rtl_reverses_columns_not_rows() -> None:
    grid = [
        panel(40, 40, 340, 540),
        panel(420, 40, 340, 540),
        panel(40, 620, 340, 540),
        panel(420, 620, 340, 540),
    ]
    result = ordering.order(grid, ReadingDirection.RTL)
    assert positions(result.panels) == [(420, 40), (40, 40), (420, 620), (40, 620)]


def test_tall_panel_left_then_stacked_right() -> None:
    layout = [panel(420, 340, 340, 300), panel(40, 40, 340, 600), panel(420, 40, 340, 260)]
    result = ordering.order(layout, ReadingDirection.LTR)
    assert positions(result.panels) == [(40, 40), (420, 40), (420, 340)]
    assert result.confidence == 1.0


def test_interlocking_layout_is_ambiguous() -> None:
    # Overlapping extents on both axes cannot be cut cleanly.
    layout = [panel(0, 0, 60, 60), panel(40, 40, 60, 60)]
    result = ordering.order(layout, ReadingDirection.LTR)
    assert positions(result.panels) == [(0, 0), (40, 40)]
    assert result.confidence == 0.5


def test_ambiguous_fallback_rtl() -> None:
    layout = [panel(0, 0, 60, 60), panel(40, 0, 60, 60)]
    rtl = ordering.order(layout, ReadingDirection.RTL)
    assert positions(rtl.panels) == [(40, 0), (0, 0)]


def test_uncuttable_group_reads_panels_with_near_equal_tops_left_to_right() -> None:
    # Diagonal gutters make the boxes overlap on both axes, so no clean cut exists. The
    # bottom-right panel's top is 75px above the bottom-left's: still the same row.
    layout = [
        panel(0, 0, 1988, 1188),
        panel(0, 1190, 1641, 939),
        panel(1642, 1297, 346, 724),
        panel(1284, 1823, 704, 1233),
        panel(0, 1898, 1283, 1158),
    ]
    result = ordering.order(layout, ReadingDirection.LTR)
    assert positions(result.panels) == [(0, 0), (0, 1190), (1642, 1297), (0, 1898), (1284, 1823)]
    rtl = ordering.order(layout[3:], ReadingDirection.RTL)
    assert positions(rtl.panels) == [(1284, 1823), (0, 1898)]


rect = st.builds(
    panel,
    st.integers(0, 500),
    st.integers(0, 500),
    st.integers(1, 200),
    st.integers(1, 200),
)


@given(st.lists(rect, max_size=12), st.sampled_from(ReadingDirection))
def test_ordering_is_a_permutation_with_valid_confidence(
    panels: list[DetectedPanel], direction: ReadingDirection
) -> None:
    result = ordering.order(panels, direction)
    assert sorted(map(id, result.panels)) == sorted(map(id, panels))
    assert 0.0 <= result.confidence <= 1.0


def test_small_overlaps_still_separate_panels_with_the_default_tolerance() -> None:
    # Ragged ML boxes overlap by a few pixels; the order must still be found without guessing.
    row = [
        panel(0, 0, 105, 100),
        panel(100, 0, 105, 100),
        panel(0, 102, 105, 100),
        panel(100, 98, 105, 100),
    ]
    result = ordering.order(row, ReadingDirection.LTR)
    assert positions(result.panels) == [(0, 0), (100, 0), (0, 102), (100, 98)]
    assert result.confidence == 1.0


def test_tolerance_zero_keeps_the_strict_behaviour() -> None:
    row = [panel(0, 0, 105, 100), panel(100, 0, 105, 100)]
    strict = XYCutOrdering(gap_tolerance=0.0).order(row, ReadingDirection.LTR)
    assert strict.confidence == 0.5  # overlapping extents: no clean cut, order is a guess


def test_large_overlaps_are_still_ambiguous() -> None:
    result = ordering.order([panel(0, 0, 60, 60), panel(40, 40, 60, 60)], ReadingDirection.LTR)
    assert result.confidence == 0.5
