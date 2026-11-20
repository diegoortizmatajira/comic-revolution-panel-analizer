"""Builds the analyzer for a request: the server defaults with the request's overrides applied.

Detectors are cheap to build except the ML one, whose weights load once and are shared by every
analyzer made here (a per-request threshold change only clones the wrapper, see
``OnnxPanelDetector.with_thresholds``).
"""

from __future__ import annotations

import hashlib
import json
import threading
from typing import TYPE_CHECKING, Any

from app.core.errors import InvalidOptionError
from app.core.options import OVERRIDABLE
from app.services.analyzer import PanelAnalyzer
from app.services.detection.captions import CaptionClaimer, ChainedRefiner
from app.services.detection.classical import ClassicalPanelDetector
from app.services.detection.hybrid import HybridPanelDetector, ReliabilityPolicy
from app.services.detection.onnx import OnnxPanelDetector
from app.services.detection.projection import ProjectionPanelDetector
from app.services.detection.refine import EdgeSnapRefiner
from app.services.ordering.xy_cut import XYCutOrdering

if TYPE_CHECKING:
    from collections.abc import Mapping

    from app.core.config import Settings
    from app.domain.ports import PanelDetector, PanelRefiner
    from app.services.detection.registry import MlConfig

# The reading direction does not shape the analyzer, it is passed to ``analyze``.
_PROCESSING = tuple(name for name in OVERRIDABLE if name != "reading_direction")


class PipelineFactory:
    def __init__(self, settings: Settings, ml: MlConfig | None) -> None:
        self._settings = settings
        self._ml = ml
        self._onnx: OnnxPanelDetector | None = None
        self._lock = threading.Lock()

    def defaults(self) -> dict[str, Any]:
        """The default of every option that shapes the analyzer (all but the reading direction)."""
        return {name: getattr(self._settings, name) for name in _PROCESSING}

    def analyzer(self, overrides: Mapping[str, Any] | None = None) -> PanelAnalyzer:
        """Analyzer for the server defaults, changed by ``overrides`` (option name to value).

        An override equal to the default changes nothing, so it does not alter the ``version``
        of the result either.
        """
        defaults = self.defaults()
        changed = {k: v for k, v in (overrides or {}).items() if k in defaults and v != defaults[k]}
        options = defaults | changed
        refiners: list[PanelRefiner] = []
        if options["refine_edges"]:
            refiners.append(EdgeSnapRefiner(max_expand_ratio=options["refine_max_expand"]))
        if options["claim_captions"]:
            refiners.append(CaptionClaimer())
        refiner = ChainedRefiner(*refiners) if len(refiners) > 1 else next(iter(refiners), None)
        return PanelAnalyzer(
            self._detector(options),
            XYCutOrdering(options["order_gap_tolerance"]),
            refiner,
            variant=_variant(changed),
        )

    def _detector(self, options: Mapping[str, Any]) -> PanelDetector:
        name = options["detector"]
        if name == "classical":
            return ClassicalPanelDetector(min_area_ratio=options["min_area_ratio"])
        projection = ProjectionPanelDetector(
            color_tolerance=options["color_tolerance"],
            gutter_content_ratio=options["gutter_content_ratio"],
            min_gutter_ratio=options["min_gutter_ratio"],
            min_area_ratio=options["min_area_ratio"],
        )
        if name == "projection":
            return projection
        model = self._model().with_thresholds(options["ml_confidence"], options["ml_nms_iou"])
        if name == "ml":
            return model
        policy = ReliabilityPolicy(
            min_panels=options["hybrid_min_panels"], max_aspect=options["hybrid_max_aspect"]
        )
        return HybridPanelDetector(projection, model, policy)

    def _model(self) -> OnnxPanelDetector:
        if self._ml is None:
            msg = "This detector needs an ML model: the server has none configured."
            raise InvalidOptionError(msg, {"option": "detector"})
        with self._lock:
            if self._onnx is None:
                ml = self._ml
                self._onnx = OnnxPanelDetector(
                    ml.model,
                    input_size=ml.input_size,
                    confidence=ml.confidence,
                    nms_iou=ml.nms_iou,
                    output_format=ml.output_format,
                    panel_class=ml.panel_class,
                )
            return self._onnx


def _variant(changed: Mapping[str, Any]) -> str:
    """Short stable tag of the overrides that differ from the defaults ('' when none do)."""
    if not changed:
        return ""
    text = json.dumps(changed, sort_keys=True, default=str)
    return hashlib.sha256(text.encode()).hexdigest()[:8]
