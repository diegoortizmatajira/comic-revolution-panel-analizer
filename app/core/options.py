"""Processing parameters that a request may override, with their allowed ranges.

One definition serves both the settings (the server-wide default, ``PANELS_<NAME>``) and the
request options (``AnalyzeOptions``), so the two cannot disagree about a range. The names are
the same in both places.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field

DetectorName = Annotated[
    Literal["projection", "classical", "ml", "hybrid"],
    Field(description="Detector. `ml` and `hybrid` need a model configured on the server."),
]
ReadingDirectionName = Annotated[
    Literal["ltr", "rtl"], Field(description="Reading direction: `ltr` (western) or `rtl` (manga).")
]
RefineEdges = Annotated[
    bool, Field(description="Grow edges that cut through balloons or captions out to a gutter.")
]
RefineMaxExpand = Annotated[
    float,
    Field(gt=0, le=0.5, description="Farthest an edge may grow, as a fraction of the page side."),
]
ClaimCaptions = Annotated[
    bool, Field(description="Extend each panel's top edge over the lettering that hangs above it.")
]
ColorTolerance = Annotated[
    float,
    Field(
        ge=5,
        le=120,
        description="projection: how far a colour may differ from the paper to count as content. "
        "Raise it for stained or noisy paper, lower it for pale art.",
    ),
]
GutterContentRatio = Annotated[
    float,
    Field(
        ge=0,
        le=0.05,
        description="projection: share of a row or column that may be content and still count as "
        "gutter. Raise it when art or balloons bleed into the gutters.",
    ),
]
MinGutterRatio = Annotated[
    float,
    Field(
        ge=0.0003,
        le=0.02,
        description="projection: narrowest gutter, as a fraction of the page side. Lower it to cut "
        "pages with very narrow gutters.",
    ),
]
MinAreaRatio = Annotated[
    float,
    Field(
        ge=0.002,
        le=0.2,
        description="projection, classical: smallest panel, as a fraction of the page area. Lower "
        "it to keep small insets, raise it to drop lettering pieces.",
    ),
]
HybridMinPanels = Annotated[
    int,
    Field(
        ge=1,
        le=10,
        description="hybrid: ask the ML model when projection finds fewer panels than this.",
    ),
]
HybridMaxAspect = Annotated[
    float,
    Field(
        ge=2,
        le=20,
        description="hybrid: ask the ML model when projection returns a panel whose long side is "
        "this many times its short side (a caption strip).",
    ),
]
MlConfidence = Annotated[
    float,
    Field(ge=0.01, le=0.99, description="ml, hybrid: lowest model score that counts as a panel."),
]
MlNmsIou = Annotated[
    float,
    Field(
        ge=0.05,
        le=0.95,
        description="ml, hybrid: overlap above which the model's duplicate boxes are merged.",
    ),
]
OrderGapTolerance = Annotated[
    float,
    Field(
        ge=0,
        le=0.3,
        description="Reading order: overlap tolerated between panels of different rows or "
        "columns, as a fraction of the group's extent.",
    ),
]

# Request options that have a server-wide default: option name == setting name.
OVERRIDABLE = (
    "detector",
    "reading_direction",
    "refine_edges",
    "refine_max_expand",
    "claim_captions",
    "color_tolerance",
    "gutter_content_ratio",
    "min_gutter_ratio",
    "min_area_ratio",
    "hybrid_min_panels",
    "hybrid_max_aspect",
    "ml_confidence",
    "ml_nms_iou",
    "order_gap_tolerance",
)
