"""Named detectors, shared by the API container and the evaluation harness."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path  # noqa: TC003  (used at runtime in the dataclass field)
from typing import TYPE_CHECKING

from app.services.detection.classical import ClassicalPanelDetector
from app.services.detection.hybrid import HybridPanelDetector
from app.services.detection.onnx import OnnxPanelDetector, OutputFormat
from app.services.detection.projection import ProjectionPanelDetector

if TYPE_CHECKING:
    from collections.abc import Callable

    from app.domain.ports import PanelDetector

# Detectors that need nothing but the code.
DETECTORS: dict[str, Callable[[], PanelDetector]] = {
    "projection": ProjectionPanelDetector,
    "classical": ClassicalPanelDetector,
}
# Detectors that need an ML model supplied by the deployer (see docs/adr/0002).
ML_DETECTORS = ("ml", "hybrid")


@dataclass(frozen=True, slots=True)
class MlConfig:
    model: Path
    input_size: int = 640
    confidence: float = 0.25
    nms_iou: float = 0.45
    output_format: OutputFormat = "yolov8"
    panel_class: int = 0


def detector_names() -> list[str]:
    return [*DETECTORS, *ML_DETECTORS]


def _onnx(ml: MlConfig) -> OnnxPanelDetector:
    return OnnxPanelDetector(
        ml.model,
        input_size=ml.input_size,
        confidence=ml.confidence,
        nms_iou=ml.nms_iou,
        output_format=ml.output_format,
        panel_class=ml.panel_class,
    )


def build_detector(name: str, ml: MlConfig | None = None) -> PanelDetector:
    if name in ML_DETECTORS:
        if ml is None:
            msg = f"Detector {name!r} needs an ML model: set PANELS_ML_MODEL (or --ml-model)."
            raise ValueError(msg)
        onnx = _onnx(ml)
        return onnx if name == "ml" else HybridPanelDetector(ProjectionPanelDetector(), onnx)
    try:
        return DETECTORS[name]()
    except KeyError:
        known = ", ".join(sorted(detector_names()))
        msg = f"Unknown detector {name!r}; known detectors: {known}"
        raise ValueError(msg) from None
