"""Typed application settings (12-factor: environment variables, prefix PANELS_)."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
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
    samples_dir: Path = Path("samples")  # images listed by the demo page

    detector: Literal["projection", "classical"] = "projection"

    log_level: str = "INFO"
    log_json: bool = True

    def parsed_api_keys(self) -> frozenset[str]:
        raw = self.api_keys.get_secret_value()
        return frozenset(k.strip() for k in raw.split(",") if k.strip())
