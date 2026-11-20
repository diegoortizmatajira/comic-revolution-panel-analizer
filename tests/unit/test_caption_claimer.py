import numpy as np
import pytest
from app.container import build_container
from app.core.config import Settings
from app.evaluation.labels import PageLabels
from app.services.detection.captions import CaptionClaimer, ChainedRefiner
from app.services.detection.refine import EdgeSnapRefiner
from numpy.typing import NDArray
from pydantic import ValidationError

from tests.conftest import panel

WIDTH, HEIGHT = 600, 800


def blank() -> NDArray[np.uint8]:
    return np.full((HEIGHT, WIDTH, 3), 255, dtype=np.uint8)


def ink(page: NDArray[np.uint8], x0: int, y0: int, x1: int, y1: int) -> None:
    page[y0:y1, x0:x1] = 20


def box(page: NDArray[np.uint8], x0: int, y0: int, x1: int, y1: int) -> None:
    """A caption box: a hollow rectangle with a 3 px outline (solid blocks have no contrast)."""
    ink(page, x0, y0, x1, y0 + 3)
    ink(page, x0, y1 - 3, x1, y1)
    ink(page, x0, y0, x0 + 3, y1)
    ink(page, x1 - 3, y0, x1, y1)


def tops(page: NDArray[np.uint8], *boxes: tuple[int, int, int, int]) -> list[int]:
    panels = [panel(x, y, w, h) for x, y, w, h in boxes]
    return [p.bbox.y for p in CaptionClaimer().refine(page, panels)]


def test_claims_the_lettering_that_touches_the_top_edge() -> None:
    page = blank()
    ink(page, 100, 300, 500, 600)  # the frame's art
    box(page, 120, 240, 480, 296)  # a caption just above it
    assert tops(page, (100, 300, 400, 300)) == [240]


def test_leaves_a_plain_gutter_alone() -> None:
    page = blank()
    ink(page, 100, 300, 500, 600)
    box(page, 120, 200, 480, 230)  # lettering far from the edge: not this panel's
    assert tops(page, (100, 300, 400, 300)) == [300]


def test_stops_at_the_end_of_the_lettering() -> None:
    page = blank()
    ink(page, 100, 300, 500, 600)
    box(page, 120, 260, 480, 296)  # attached
    box(page, 120, 100, 480, 130)  # separate, after a clean gap
    assert tops(page, (100, 300, 400, 300)) == [260]


def test_never_claims_into_the_panel_above() -> None:
    page = blank()
    ink(page, 100, 100, 500, 400)
    ink(page, 100, 410, 500, 700)
    box(page, 120, 400, 480, 410)  # lettering touching both: the row below takes it, once
    result = tops(page, (100, 100, 400, 300), (100, 410, 400, 290))
    assert result[0] == 100
    assert result[1] >= 400  # bounded by the bottom of the panel above


def test_ignores_lettering_outside_the_panel_columns() -> None:
    page = blank()
    ink(page, 300, 300, 500, 600)
    box(page, 50, 240, 250, 296)  # beside the panel, not above it
    assert tops(page, (300, 300, 200, 300)) == [300]


def test_chain_applies_every_refiner_in_order() -> None:
    chain = ChainedRefiner(EdgeSnapRefiner(), CaptionClaimer())
    assert chain.version == f"edge-snap@{EdgeSnapRefiner().version}+caption-claim@1"
    assert chain.refine(blank(), []) == ()


def test_settings_wire_the_claimer() -> None:
    def refiner(**kwargs: object) -> object:
        settings = Settings(api_keys="k", log_json=False, _env_file=None, **kwargs)  # type: ignore[arg-type]
        return build_container(settings).analyzer._refiner

    assert refiner() is None
    assert isinstance(refiner(claim_captions=True), CaptionClaimer)
    assert isinstance(refiner(claim_captions=True, refine_edges=True), ChainedRefiner)


def test_caption_panels_must_match_the_panels() -> None:
    box = {"x": 0.1, "y": 0.1, "width": 0.3, "height": 0.3}
    PageLabels.model_validate({"panels": [box], "caption_panels": [box]})
    with pytest.raises(ValidationError):
        PageLabels.model_validate({"panels": [box], "caption_panels": [box, box]})
