# Comic Panel Analyzer

API that analyzes a comic book page and returns its panels, reading order, coordinates and
a reliability score, so third-party apps can offer guided panel reading.

## Run

```sh
uv sync
cp .env.example .env            # set PANELS_API_KEYS
uv run uvicorn app.main:app_factory --factory --reload
```

Demo page (upload/paste a page and see numbered panel overlays): set
`PANELS_ENABLE_DEMO_PAGE=true` and open http://localhost:8000/demo. Images placed in
`samples/` (git-ignored, your own pages) and the committed `tests/fixtures/` pages show up in its
"Sample pages" dropdown, in separate groups.

Interactive docs: http://localhost:8000/docs · OpenAPI: `/openapi.json` · Health: `/healthz`

Or with Docker: `docker build -t panels . && docker run -p 8000:8000 -e PANELS_API_KEYS=change-me panels`

## API

All analyze endpoints require an `X-API-Key` header.

```sh
# URL
curl -X POST localhost:8000/v1/panels/analyze -H 'X-API-Key: change-me' \
  -H 'Content-Type: application/json' \
  -d '{"image": {"url": "https://example.com/page.png"}, "options": {"reading_direction": "rtl"}}'

# Base64 (or a data: URI)
curl -X POST localhost:8000/v1/panels/analyze -H 'X-API-Key: change-me' \
  -H 'Content-Type: application/json' \
  -d "{\"image\": {\"base64\": \"$(base64 -w0 page.png)\"}}"

# Upload
curl -X POST localhost:8000/v1/panels/analyze/upload -H 'X-API-Key: change-me' \
  -F file=@page.png -F reading_direction=ltr
```

Response:

```json
{
  "image": {"width": 800, "height": 1200},
  "reading_direction": "ltr",
  "panels": [
    {"id": 1, "order": 1,
     "bbox": {"x": 40, "y": 40, "width": 340, "height": 540},
     "bbox_normalized": {"x": 0.05, "y": 0.033, "width": 0.425, "height": 0.45},
     "confidence": 1.0}
  ],
  "sequence_confidence": 1.0,
  "method": "projection",
  "version": "projection@1+xycut@1",
  "whole_page_fallback": false,
  "warnings": []
}
```

`confidence` is per-panel detection quality; `sequence_confidence` is how unambiguous the
reading order is (1.0 = every panel separated by clean gutters).

### Client examples

Python (`pip install httpx`):

```python
import httpx

API = "http://localhost:8000"
HEADERS = {"X-API-Key": "change-me"}

# Upload a file
with open("page.png", "rb") as f:
    result = httpx.post(
        f"{API}/v1/panels/analyze/upload",
        headers=HEADERS,
        files={"file": ("page.png", f, "image/png")},
        data={"reading_direction": "ltr"},
        timeout=30,
    ).json()

# Or let the service download it
result = httpx.post(
    f"{API}/v1/panels/analyze",
    headers=HEADERS,
    json={"image": {"url": "https://example.com/page.png"}},
    timeout=30,
).json()

if result["whole_page_fallback"] or result["sequence_confidence"] < 0.6:
    print("not reliable: show the whole page")
for panel in result["panels"]:
    b = panel["bbox_normalized"]  # fractions of the page, independent of image size
    print(panel["order"], b["x"], b["y"], b["width"], b["height"])
```

JavaScript (browser or Node 18+):

```js
const API = "http://localhost:8000";

async function analyze(file, readingDirection = "ltr") {
  const form = new FormData();
  form.append("file", file);
  form.append("reading_direction", readingDirection);
  const response = await fetch(`${API}/v1/panels/analyze/upload`, {
    method: "POST",
    headers: { "X-API-Key": "change-me" },
    body: form,
  });
  const body = await response.json();
  if (!response.ok) throw new Error(`${body.code}: ${body.message}`);
  return body.panels; // ordered; use bbox_normalized to position a zoom/pan viewport
}
```

Do not ship the API key in a public web page; call the service from your own backend.

`version` identifies the detector and ordering algorithms that produced the result. Compare it
for equality: cache results by `(image, version)` and recompute when it changes.

When no panel is found the API still returns one panel covering the whole page, with
`confidence` and `sequence_confidence` of 0.1, `whole_page_fallback: true` and a warning, so
a reader can fall back to showing the page. A single detected panel is returned as is.

`method` names the detector that produced the result. Choose it with `PANELS_DETECTOR`:
`projection` (default; gutter-based, works on colour) or `classical` (contour-based).
Both struggle when panels have no gutter between them, when gutters are slanted, or on
dense pages where balloons cross gutters; those pages return fewer, larger panels, and
a warning when only one is found.

Errors share one shape: `{"code", "message", "details", "request_id"}` with statuses
400, 401, 413, 415, 422, 429 (with `Retry-After`), 502 and 504.

## Development

```sh
uv run pytest          # tests + coverage gate (90%)
uv run ruff check . && uv run ruff format --check .
uv run mypy            # strict
uv run pre-commit install
```

Architecture notes: [docs/adr](docs/adr). Configuration: see `.env.example`.
