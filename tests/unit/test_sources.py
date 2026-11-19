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


class CountingResolver:
    """Answers differently on every call, like a DNS-rebinding attacker would."""

    def __init__(self, answers: list[list[str]]) -> None:
        self._answers = answers
        self.calls = 0

    async def resolve(self, host: str, port: int) -> list[str]:
        answer = self._answers[min(self.calls, len(self._answers) - 1)]
        self.calls += 1
        return answer


async def test_connection_is_pinned_to_the_validated_ip() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, content=b"ok")

    resolver = CountingResolver([["93.184.216.34"], ["10.0.0.5"]])  # second answer is internal
    f = UrlImageFetcher(
        max_bytes=100,
        timeout_seconds=1,
        max_redirects=2,
        resolver=resolver,
        transport=httpx.MockTransport(handler),
    )
    assert await f.fetch("https://example.com:8443/a.png") == b"ok"
    (request,) = seen
    assert request.url.host == "93.184.216.34"  # connects to the IP that was validated
    assert request.headers["host"] == "example.com:8443"
    assert request.extensions["sni_hostname"] == "example.com"
    assert resolver.calls == 1  # resolved once per hop, so a later answer is never used


async def test_http_hop_pins_without_sni() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, content=b"ok")

    await fetcher(httpx.MockTransport(handler)).fetch("http://example.com/a.png")
    assert seen[0].url.host == "93.184.216.34"
    assert "sni_hostname" not in seen[0].extensions


async def test_ipv6_address_is_pinned() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, content=b"ok")

    f = UrlImageFetcher(
        max_bytes=100,
        timeout_seconds=1,
        max_redirects=0,
        resolver=FakeResolver({"example.com": ["2606:2800:220:1:248:1893:25c8:1946"]}),
        transport=httpx.MockTransport(handler),
    )
    await f.fetch("https://example.com/a.png")
    assert seen[0].url.host == "2606:2800:220:1:248:1893:25c8:1946"


async def test_every_redirect_hop_is_resolved_and_pinned_again() -> None:
    hosts: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        hosts.append(request.headers["host"])
        if request.url.path == "/a":
            return httpx.Response(302, headers={"location": "https://example.com/b"})
        return httpx.Response(200, content=b"ok")

    resolver = CountingResolver([["93.184.216.34"]])
    f = UrlImageFetcher(
        max_bytes=100,
        timeout_seconds=1,
        max_redirects=2,
        resolver=resolver,
        transport=httpx.MockTransport(handler),
    )
    assert await f.fetch("https://example.com/a") == b"ok"
    assert hosts == ["example.com", "example.com"]
    assert resolver.calls == 2


@pytest.mark.parametrize(
    "url", ["http://127.0.0.1/a", "http://[::1]/a", "http://169.254.169.254/latest/meta-data"]
)
async def test_blocks_private_ip_literals_without_dns(url: str) -> None:
    with pytest.raises(InvalidUrlError):
        await fetcher().fetch(url)


async def test_public_ip_literal_is_allowed() -> None:
    f = fetcher(httpx.MockTransport(lambda r: httpx.Response(200, content=b"ok")))
    assert await f.fetch("http://93.184.216.34/a.png") == b"ok"


async def test_rejects_embedded_credentials() -> None:
    with pytest.raises(InvalidUrlError, match="credentials"):
        await fetcher().fetch("https://user:pass@example.com/a.png")


async def test_blocks_when_any_resolved_address_is_private() -> None:
    f = UrlImageFetcher(
        max_bytes=100,
        timeout_seconds=1,
        max_redirects=0,
        resolver=FakeResolver({"mixed.test": ["93.184.216.34", "10.0.0.5"]}),
    )
    with pytest.raises(InvalidUrlError):
        await f.fetch("http://mixed.test/a")


async def test_empty_resolution_is_a_fetch_error() -> None:
    f = UrlImageFetcher(
        max_bytes=100,
        timeout_seconds=1,
        max_redirects=0,
        resolver=FakeResolver({"empty.test": []}),
    )
    with pytest.raises(ImageFetchError):
        await f.fetch("http://empty.test/a")


async def test_sends_descriptive_user_agent() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, content=b"ok")

    await fetcher(httpx.MockTransport(handler)).fetch("https://example.com/a.png")
    assert seen[0].headers["user-agent"].startswith("comic-panel-analyzer/")
