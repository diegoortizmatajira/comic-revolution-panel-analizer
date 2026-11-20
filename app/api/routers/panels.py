from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, File, Form, UploadFile
from fastapi.concurrency import run_in_threadpool

from app.api.deps import AuthDep, ContainerDep  # noqa: TC001
from app.core.errors import ImageTooLargeError
from app.core.options import OVERRIDABLE
from app.domain.models import ReadingDirection
from app.schemas.panels import (
    AnalyzeOptions,
    AnalyzeRequest,
    AnalyzeResponse,
    ErrorBody,
    OptionsResponse,
    UploadOptions,
    describe_options,
)
from app.services.image.decoder import decode_image
from app.services.image.sources import decode_base64_image

router = APIRouter(prefix="/v1/panels", tags=["panels"])

_ERRORS: dict[int | str, dict[str, Any]] = {
    code: {"model": ErrorBody} for code in (400, 401, 413, 415, 422, 429, 502, 504)
}


async def _analyze(
    container: ContainerDep, data: bytes, options: AnalyzeOptions
) -> AnalyzeResponse:
    overrides = options.overrides()
    direction = ReadingDirection(
        overrides.pop("reading_direction", container.settings.reading_direction)
    )

    def work() -> AnalyzeResponse:
        image = decode_image(data, container.limits)
        analyzer = container.pipeline.analyzer(overrides) if overrides else container.analyzer
        return AnalyzeResponse.from_result(analyzer.analyze(image, direction))

    return await run_in_threadpool(work)  # CPU-bound: keep the event loop free


@router.post("/analyze", response_model=AnalyzeResponse, responses=_ERRORS)
async def analyze(body: AnalyzeRequest, container: ContainerDep, _key: AuthDep) -> AnalyzeResponse:
    """Analyze a page supplied as a URL or a base64-encoded image."""
    if body.image.url is not None:
        data = await container.fetcher.fetch(body.image.url)
    else:
        assert body.image.base64 is not None  # noqa: S101  (guaranteed by schema validator)
        data = decode_base64_image(body.image.base64, container.limits.max_bytes)
    return await _analyze(container, data, body.options)


@router.post("/analyze/upload", response_model=AnalyzeResponse, responses=_ERRORS)
async def analyze_upload(
    container: ContainerDep,
    _key: AuthDep,
    file: Annotated[UploadFile, File(description="Page image (JPEG, PNG or WebP).")],
    options: Annotated[UploadOptions, Form()],
) -> AnalyzeResponse:
    """Analyze a page supplied as a multipart file upload.

    The processing options are form fields with the same names as in the JSON body.
    """
    limit = container.limits.max_bytes
    data = await file.read(limit + 1)
    if len(data) > limit:
        msg = "Image exceeds the maximum allowed size."
        raise ImageTooLargeError(msg, {"max_bytes": limit})
    return await _analyze(container, data, options)


@router.get("/options", response_model=OptionsResponse, responses=_ERRORS)
async def options(container: ContainerDep, _key: AuthDep) -> OptionsResponse:
    """The processing options a request may override, with this server's default for each."""
    defaults = {name: getattr(container.settings, name) for name in OVERRIDABLE}
    return OptionsResponse(options=describe_options(defaults))
