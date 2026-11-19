"""Developer demo page: pick, upload or paste a page and see the detected panels overlaid."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from app.api.deps import ContainerDep  # noqa: TC001

router = APIRouter(tags=["demo"], include_in_schema=False)

_PAGE = Path(__file__).resolve().parent.parent.parent / "static" / "demo.html"
_IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".webp"})


def _sample_files(folder: Path) -> list[Path]:
    if not folder.is_dir():
        return []
    return sorted(
        p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in _IMAGE_SUFFIXES
    )


@router.get("/demo")
async def demo() -> FileResponse:
    return FileResponse(_PAGE, media_type="text/html")


@router.get("/demo/samples")
async def list_samples(container: ContainerDep) -> list[str]:
    return [p.name for p in _sample_files(container.settings.samples_dir)]


@router.get("/demo/samples/{name}")
async def get_sample(name: str, container: ContainerDep) -> FileResponse:
    # Match against the listing instead of building a path from user input (no traversal).
    for path in _sample_files(container.settings.samples_dir):
        if path.name == name:
            return FileResponse(path)
    raise HTTPException(status_code=404, detail="Sample not found.")
