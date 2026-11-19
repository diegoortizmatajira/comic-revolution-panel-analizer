import base64

import httpx
import pytest
from app.core.errors import (
    ImageFetchError,
    ImageFetchTimeoutError,
    ImageTooLargeError,
    InvalidImageError,
    InvalidUrlError,
)
from app.services.image.sources import UrlImageFetcher, decode_base64_image


def test_base64_plain_and_data_uri() -> None:
    encoded = base64.b64encode(b"abc").decode()
    assert decode_base64_image(encoded, 100) == b"abc"
    assert decode_base64_image(f"data:image/png;base64,{encoded}", 100) == b"abc"


@pytest.mark.parametrize("value", ["!!!notbase64", "data:image/png"])
def test_base64_invalid(value: str) -> None:
    with pytest.raises(InvalidImageError):
        decode_base64_image(value, 100)


def test_base64_too_large() -> None:
    with pytest.raises(ImageTooLargeError):
        decode_base64_image("A" * 1000, 10)


class FakeResolver:
    def __init__(self, mapping: dict[str, list[str]]) -> None:
        self._mapping = mapping

    async def resolve(self, host: str, port: int) -> list[str]:
        return self._mapping[host]


def fetcher(handler: httpx.MockTransport | None = None, **kw: object) -> UrlImageFetcher:
    resolver = FakeResolver({"example.com": ["93.184.216.34"], "evil.test": ["10.0.0.5"]})
    return UrlImageFetcher(
        max_bytes=100,
        timeout_seconds=1,
        max_redirects=2,
        resolver=resolver,
        transport=handler,
        **kw,  # type: ignore[arg-type]
    )


async def test_fetch_ok() -> None:
    f = fetcher(httpx.MockTransport(lambda r: httpx.Response(200, content=b"img")))
    assert await f.fetch("https://example.com/a.png") == b"img"


@pytest.mark.parametrize("url", ["ftp://example.com/a", "file:///etc/passwd", "/relative"])
async def test_rejects_non_http_urls(url: str) -> None:
    with pytest.raises(InvalidUrlError):
        await fetcher().fetch(url)


async def test_blocks_private_addresses() -> None:
    with pytest.raises(InvalidUrlError):
        await fetcher().fetch("http://evil.test/a.png")


async def test_blocks_redirect_to_private_host() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(302, headers={"location": "http://evil.test/x"})

    with pytest.raises(InvalidUrlError):
        await fetcher(httpx.MockTransport(handler)).fetch("https://example.com/a")


async def test_follows_safe_redirect() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/a":
            return httpx.Response(302, headers={"location": "/b"})
        return httpx.Response(200, content=b"ok")

    assert await fetcher(httpx.MockTransport(handler)).fetch("https://example.com/a") == b"ok"


async def test_too_many_redirects() -> None:
    f = fetcher(httpx.MockTransport(lambda r: httpx.Response(302, headers={"location": "/loop"})))
    with pytest.raises(ImageFetchError, match="redirects"):
        await f.fetch("https://example.com/loop")


async def test_upstream_error_status() -> None:
    f = fetcher(httpx.MockTransport(lambda r: httpx.Response(404)))
    with pytest.raises(ImageFetchError):
        await f.fetch("https://example.com/a")


async def test_download_size_limit() -> None:
    f = fetcher(httpx.MockTransport(lambda r: httpx.Response(200, content=b"x" * 500)))
    with pytest.raises(ImageTooLargeError):
        await f.fetch("https://example.com/a")


async def test_timeout_maps_to_domain_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow", request=request)

    with pytest.raises(ImageFetchTimeoutError):
        await fetcher(httpx.MockTransport(handler)).fetch("https://example.com/a")


async def test_allow_private_skips_resolution() -> None:
    f = fetcher(
        httpx.MockTransport(lambda r: httpx.Response(200, content=b"ok")), allow_private=True
    )
    assert await f.fetch("http://unresolvable.local/a") == b"ok"
