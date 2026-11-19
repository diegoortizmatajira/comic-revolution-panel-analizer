# ADR 0001: Architecture and detection approach

- Status: accepted
- Context: issue #1

## Decisions

- **FastAPI** for the HTTP layer: async, generated OpenAPI contract, Pydantic validation.
- **Hexagonal layout.** `app/domain` holds value objects and ports (Protocols) with no
  framework or I/O imports. `app/services` holds adapters and application services.
  `app/api` is a thin HTTP adapter. `app/container.py` is the only composition root.
- **Hybrid detection, behind `PanelDetector`.** v1 ships a classical OpenCV detector.
  An ML detector can be added as another implementation without touching callers
  (open/closed). Known limitation of the classical detector: panels that touch with no
  gutter merge into one.
- **Two classical detectors, selected with `PANELS_DETECTOR`.** `projection` (default)
  works on colour: it estimates the background from the dominant edge colour, marks
  non-background pixels as content, and recursively cuts along gutters found in row and
  column profiles. `classical` takes outer contours of content blobs on a grayscale
  image. Projection separated panels the contour detector merged on a dark, hand-framed
  page, but both fail where there is no gutter: touching panels, art that bleeds into
  the next panel, diagonal or slanted gutters (no clean row/column exists) and dense
  pages with balloons crossing gutters. These cases are what the ML detector is for.
- **Reading order:** recursive XY-cut honouring `ltr`/`rtl`. `sequence_confidence`
  measures layout ambiguity only (share of panels that could not be cut cleanly);
  per-panel `confidence` measures detection quality.
- **Synchronous processing.** CPU-bound work runs in a thread pool. Async jobs are a
  possible follow-up.
- **Security.** API keys (compared by SHA-256 digest in constant time), in-memory
  sliding-window rate limit (single process; replace with a shared store to scale out),
  SSRF validation on every redirect hop. Known gap: DNS is resolved for validation and
  again by the HTTP client, so a DNS-rebinding window remains; pin the resolved IP
  before production use.
