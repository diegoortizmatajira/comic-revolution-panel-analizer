import base64
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from app.container import build_container
from app.core.config import Settings
from app.main import create_app
from fastapi.testclient import TestClient

from tests.conftest import make_page

KEY = "test-key"
HEADERS = {"X-API-Key": KEY}


class FakeFetcher:
    def __init__(self, data: bytes) -> None:
        self.data = data
        self.urls: list[str] = []

    async def fetch(self, url: str) -> bytes:
        self.urls.append(url)
        return self.data


@pytest.fixture
def settings() -> Settings:
    return Settings(api_keys=KEY, rate_limit_per_minute=1000, log_json=False, _env_file=None)


@pytest.fixture
def client(settings: Settings, page_2x2: bytes) -> Iterator[TestClient]:
    container = replace(build_container(settings), fetcher=FakeFetcher(page_2x2))
    with TestClient(create_app(settings, container)) as c:
        yield c


def test_health(client: TestClient) -> None:
    assert client.get("/healthz").json() == {"status": "ok"}


def assert_four_panels(body: dict[str, Any], direction: str = "ltr") -> None:
    assert body["image"] == {"width": 800, "height": 1200}
    assert body["reading_direction"] == direction
    assert [p["order"] for p in body["panels"]] == [1, 2, 3, 4]
    assert body["sequence_confidence"] == 1.0
    assert body["method"] == "projection"
    first = body["panels"][0]
    assert first["bbox"] == {"x": 40, "y": 40, "width": 340, "height": 540}
    assert first["bbox_normalized"]["x"] == pytest.approx(0.05)


def test_analyze_base64(client: TestClient, page_2x2: bytes) -> None:
    payload = {"image": {"base64": base64.b64encode(page_2x2).decode()}}
    response = client.post("/v1/panels/analyze", json=payload, headers=HEADERS)
    assert response.status_code == 200
    assert_four_panels(response.json())
    assert response.headers["x-request-id"]


def test_analyze_url_rtl(client: TestClient) -> None:
    payload = {
        "image": {"url": "https://example.com/p.png"},
        "options": {"reading_direction": "rtl"},
    }
    response = client.post("/v1/panels/analyze", json=payload, headers=HEADERS)
    assert response.status_code == 200
    body = response.json()
    assert body["panels"][0]["bbox"]["x"] == 420  # right panel first in manga order


def test_analyze_upload(client: TestClient, page_2x2: bytes) -> None:
    response = client.post(
        "/v1/panels/analyze/upload",
        files={"file": ("page.png", page_2x2, "image/png")},
        data={"reading_direction": "ltr"},
        headers=HEADERS,
    )
    assert response.status_code == 200
    assert_four_panels(response.json())


def test_upload_too_large(client: TestClient, settings: Settings) -> None:
    big = b"x" * (settings.max_image_bytes + 1)
    response = client.post(
        "/v1/panels/analyze/upload", files={"file": ("a.png", big)}, headers=HEADERS
    )
    assert response.status_code == 413
    assert response.json()["code"] == "image_too_large"


def test_requires_api_key(client: TestClient) -> None:
    for headers in ({}, {"X-API-Key": "nope"}):
        response = client.post(
            "/v1/panels/analyze", json={"image": {"url": "https://a/b"}}, headers=headers
        )
        assert response.status_code == 401
        assert response.json()["code"] == "unauthorized"


def test_validation_error_is_structured(client: TestClient) -> None:
    response = client.post("/v1/panels/analyze", json={"image": {}}, headers=HEADERS)
    body = response.json()
    assert response.status_code == 422
    assert body["code"] == "validation_error"
    assert body["request_id"]


def test_both_sources_rejected(client: TestClient) -> None:
    payload = {"image": {"url": "https://a/b", "base64": "AAAA"}}
    assert client.post("/v1/panels/analyze", json=payload, headers=HEADERS).status_code == 422


def test_invalid_image_error_shape(client: TestClient) -> None:
    payload = {"image": {"base64": base64.b64encode(b"junk").decode()}}
    response = client.post("/v1/panels/analyze", json=payload, headers=HEADERS)
    assert response.status_code == 400
    assert set(response.json()) == {"code", "message", "details", "request_id"}


def test_unsupported_format(client: TestClient) -> None:
    gif = make_page([], size=(10, 10), fmt="GIF")
    response = client.post(
        "/v1/panels/analyze",
        json={"image": {"base64": base64.b64encode(gif).decode()}},
        headers=HEADERS,
    )
    assert response.status_code == 415


def test_rate_limit_sets_retry_after(page_2x2: bytes) -> None:
    settings = Settings(api_keys=KEY, rate_limit_per_minute=1, log_json=False, _env_file=None)
    payload = {"image": {"base64": base64.b64encode(page_2x2).decode()}}
    with TestClient(create_app(settings)) as c:
        assert c.post("/v1/panels/analyze", json=payload, headers=HEADERS).status_code == 200
        limited = c.post("/v1/panels/analyze", json=payload, headers=HEADERS)
    assert limited.status_code == 429
    assert int(limited.headers["retry-after"]) >= 1


def test_demo_page_disabled_by_default(client: TestClient) -> None:
    assert client.get("/demo").status_code == 404


def test_demo_page_served_when_enabled() -> None:
    settings = Settings(api_keys=KEY, enable_demo_page=True, log_json=False, _env_file=None)
    with TestClient(create_app(settings)) as c:
        response = c.get("/demo")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "/v1/panels/analyze/upload" in response.text


def test_demo_samples_list_and_serve(tmp_path: Path, page_2x2: bytes) -> None:
    (tmp_path / "a.png").write_bytes(page_2x2)
    (tmp_path / "notes.txt").write_text("ignore me")
    (tmp_path / ".gitignore").write_text("*")
    settings = Settings(
        api_keys=KEY, enable_demo_page=True, samples_dir=tmp_path, log_json=False, _env_file=None
    )
    with TestClient(create_app(settings)) as c:
        assert c.get("/demo/samples").json() == ["a.png"]
        served = c.get("/demo/samples/a.png")
        assert served.status_code == 200
        assert served.content == page_2x2
        assert c.get("/demo/samples/notes.txt").status_code == 404
        assert c.get("/demo/samples/..%2Fsecret.png").status_code == 404


def test_demo_samples_missing_folder_is_empty(tmp_path: Path) -> None:
    settings = Settings(
        api_keys=KEY,
        enable_demo_page=True,
        samples_dir=tmp_path / "nope",
        log_json=False,
        _env_file=None,
    )
    with TestClient(create_app(settings)) as c:
        assert c.get("/demo/samples").json() == []


def test_detector_is_selectable_by_setting(page_2x2: bytes) -> None:
    settings = Settings(api_keys=KEY, detector="classical", log_json=False, _env_file=None)
    with TestClient(create_app(settings)) as c:
        response = c.post(
            "/v1/panels/analyze/upload", files={"file": ("p.png", page_2x2)}, headers=HEADERS
        )
    assert response.json()["method"] == "classical"
