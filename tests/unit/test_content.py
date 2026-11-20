import numpy as np
from app.services.detection.content import border_paper, content_mask, local_content_mask


def stained_page() -> np.ndarray:
    """Two framed panels on paper whose tone drifts from cream (left) to deep tan (right)."""
    height, width = 700, 800
    ramp = np.linspace(0, 1, width, dtype=np.float32)[None, :, None]
    left, right = np.array([245, 235, 225], np.float32), np.array([222, 180, 120], np.float32)
    page = np.broadcast_to(left + (right - left) * ramp, (height, width, 3)).astype(np.uint8).copy()
    for x0, x1 in ((40, 380), (420, 760)):
        page[40:660, x0:x1] = (90, 140, 190)
        page[40:46, x0:x1] = page[654:660, x0:x1] = 0
        page[40:660, x0 : x0 + 6] = page[40:660, x1 - 6 : x1] = 0
    return page


def test_border_paper_reaches_margins_and_gutters_but_not_panel_interiors() -> None:
    paper = border_paper(stained_page())
    assert paper[10, 10]  # margin
    assert paper[350, 400]  # the gutter between the panels
    assert not paper[350, 200]  # inside a framed panel


def test_one_page_colour_misreads_stained_paper_but_the_local_colour_does_not() -> None:
    page = stained_page()
    gutter = (slice(100, 600), slice(386, 414))
    assert content_mask(page)[gutter].mean() > 0.5  # the single page colour calls it content
    assert local_content_mask(page)[gutter].mean() < 0.05  # the local paper colour does not


def test_local_mask_falls_back_to_the_page_colour_without_reachable_paper() -> None:
    page = np.full((100, 100, 3), 255, dtype=np.uint8)
    page[:, :] = (200, 90, 60)  # art to the very edge: no border-connected paper to learn from
    assert local_content_mask(page).shape == (100, 100)
