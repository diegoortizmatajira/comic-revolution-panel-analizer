"""Typed application settings (12-factor: environment variables, prefix PANELS_)."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Self

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="PANELS_", env_file=".env", extra="ignore")

    # Comma-separated list of accepted API keys.
    api_keys: SecretStr = SecretStr("")
    rate_limit_per_minute: int = Field(default=60, gt=0)

    max_image_bytes: int = Field(default=15 * 1024 * 1024, gt=0)
    max_image_pixels: int = Field(default=40_000_000, gt=0)

    url_fetch_timeout_seconds: float = Field(default=10.0, gt=0)
    url_max_redirects: int = Field(default=3, ge=0)
    allow_private_urls: bool = False  # development only

    enable_demo_page: bool = False  # serves /demo, a developer test page
    samples_dir: Path = Path("samples")  # local pages listed by the demo page
    fixtures_dir: Path = Path("tests/fixtures")  # committed test pages, also listed

    detector: Literal["projection", "classical", "ml", "hybrid"] = "projection"

    # Optional ML detector (needs the `ml` extra and a model file you are allowed to use).
    ml_model: Path | None = None
    ml_input_size: int = Field(default=640, gt=0)
    ml_confidence: float = Field(default=0.25, ge=0, le=1)
    ml_nms_iou: float = Field(default=0.45, ge=0, le=1)
    ml_output_format: Literal["yolov8", "end2end"] = "yolov8"
    ml_panel_class: int = Field(default=0, ge=0)

    log_level: str = "INFO"
    log_json: bool = True

    @model_validator(mode="after")
    def _ml_detectors_need_a_model(self) -> Self:
        if self.detector in {"ml", "hybrid"} and self.ml_model is None:
            msg = f"PANELS_DETECTOR={self.detector} requires PANELS_ML_MODEL"
            raise ValueError(msg)
        return self

    def parsed_api_keys(self) -> frozenset[str]:
        raw = self.api_keys.get_secret_value()
        return frozenset(k.strip() for k in raw.split(",") if k.strip())
