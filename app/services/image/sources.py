"""Adapters that turn each input method into raw image bytes."""

from __future__ import annotations

import asyncio
import base64
import binascii
import ipaddress
import socket
from dataclasses import dataclass, field
from typing import TYPE_CHECKING
from urllib.parse import urljoin, urlsplit

import httpx

from app.core.errors import (
    ImageFetchError,
    ImageFetchTimeoutError,
    ImageTooLargeError,
    InvalidImageError,
    InvalidUrlError,
)

if TYPE_CHECKING:
    from app.domain.ports import HostResolver


# Many image hosts reject the default httpx agent; identify ourselves instead.
USER_AGENT = "comic-panel-analyzer/0.1"


def decode_base64_image(value: str, max_bytes: int) -> bytes:
    """Decode a base64 string or `data:` URI."""
    if value.startswith("data:"):
        _, sep, value = value.partition(",")
        if not sep:
            msg = "Malformed data URI."
            raise InvalidImageError(msg)
    # Reject before decoding: base64 inflates size by 4/3.
    if len(value) > max_bytes * 4 // 3 + 4:
        msg = "Image exceeds the maximum allowed size."
        raise ImageTooLargeError(msg, {"max_bytes": max_bytes})
    try:
        return base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError) as exc:
        msg = "Image is not valid base64."
        raise InvalidImageError(msg) from exc


class SystemHostResolver:
    async def resolve(self, host: str, port: int) -> list[str]:
        loop = asyncio.get_running_loop()
        try:
            infos = await loop.getaddrinfo(host, port, type=socket.SOCK_STREAM)
        except socket.gaierror as exc:
            msg = "Could not resolve the image host."
            raise ImageFetchError(msg, {"host": host}) from exc
        return [str(info[4][0]) for info in infos]


class UrlImageFetcher:
    """Downloads images over http(s) with SSRF, redirect, timeout and size protection."""

    def __init__(  # noqa: PLR0913
        self,
        *,
        max_bytes: int,
        timeout_seconds: float,
        max_redirects: int,
        allow_private: bool = False,
        resolver: HostResolver | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._max_bytes = max_bytes
        self._timeout = timeout_seconds
        self._max_redirects = max_redirects
        self._allow_private = allow_private
        self._resolver: HostResolver = resolver or SystemHostResolver()
        self._transport = transport

    async def fetch(self, url: str) -> bytes:
        async with httpx.AsyncClient(
            timeout=self._timeout,
            follow_redirects=False,
            transport=self._transport,
            headers={"User-Agent": USER_AGENT, "Accept": "image/*"},
        ) as client:
            current = url
            for _ in range(self._max_redirects + 1):
                target = await self._prepare(current)
                try:
                    async with client.stream(
                        "GET", target.url, headers=target.headers, extensions=target.extensions
                    ) as response:
                        if response.is_redirect:
                            location = response.headers.get("location")
                            if not location:
                                msg = "Redirect without a location."
                                raise ImageFetchError(msg)
                            current = urljoin(current, location)
                            continue
                        if response.status_code != httpx.codes.OK:
                            msg = "The image host returned an error."
                            raise ImageFetchError(msg, {"status": response.status_code})
                        return await self._read_limited(response)
                except httpx.TimeoutException as exc:
                    msg = "Timed out fetching the image."
                    raise ImageFetchTimeoutError(msg) from exc
                except httpx.HTTPError as exc:
                    msg = "Could not fetch the image."
                    raise ImageFetchError(msg) from exc
        msg = "Too many redirects."
        raise ImageFetchError(msg, {"max_redirects": self._max_redirects})

    async def _read_limited(self, response: httpx.Response) -> bytes:
        chunks: list[bytes] = []
        total = 0
        async for chunk in response.aiter_bytes():
            total += len(chunk)
            if total > self._max_bytes:
                msg = "Image exceeds the maximum allowed size."
                raise ImageTooLargeError(msg, {"max_bytes": self._max_bytes})
            chunks.append(chunk)
        return b"".join(chunks)

    async def _prepare(self, url: str) -> _Target:
        """Validate a hop and pin it to the address that was validated.

        The hostname is resolved exactly once; the connection then goes to that IP, with
        the original Host header and TLS server name, so a second (different) DNS answer
        cannot redirect the request to an internal address (DNS rebinding).
        """
        parts = urlsplit(url)
        if parts.scheme not in {"http", "https"} or not parts.hostname:
            msg = "Only absolute http(s) URLs are allowed."
            raise InvalidUrlError(msg)
        if parts.username is not None or parts.password is not None:
            msg = "URLs with embedded credentials are not allowed."
            raise InvalidUrlError(msg)
        host = parts.hostname
        if self._allow_private:
            return _Target(url=url)

        try:
            literal = ipaddress.ip_address(host)
        except ValueError:
            literal = None
        if literal is not None:
            _require_public(literal)
            return _Target(url=url)

        port = parts.port or (443 if parts.scheme == "https" else 80)
        addresses = [ipaddress.ip_address(a) for a in await self._resolver.resolve(host, port)]
        if not addresses:
            msg = "Could not resolve the image host."
            raise ImageFetchError(msg, {"host": host})
        for address in addresses:
            _require_public(address)

        return _Target(
            url=str(httpx.URL(url).copy_with(host=str(addresses[0]))),
            headers={"Host": f"{host}:{parts.port}" if parts.port else host},
            extensions={"sni_hostname": host} if parts.scheme == "https" else {},
        )


def _require_public(address: ipaddress.IPv4Address | ipaddress.IPv6Address) -> None:
    if not address.is_global:
        msg = "The URL points to a non-public address."
        raise InvalidUrlError(msg)


@dataclass(frozen=True, slots=True)
class _Target:
    url: str
    headers: dict[str, str] = field(default_factory=dict)
    extensions: dict[str, str] = field(default_factory=dict)
