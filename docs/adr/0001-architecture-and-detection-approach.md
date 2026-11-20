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
  the next panel and dense pages with balloons crossing gutters. Projection v2 adds a
  fallback for slanted gutters: when no clean row or column exists it cuts along the
  minimum-content path across the region, accepted only if it is nearly empty (up to 2%
  blocked pixels, and then only if nearly straight). That fixed the diagonal and
  slightly slanted pages. Projection v3 then falls back to frame lines (`frames.py`): Hough
  segments, merged when collinear, so a frame broken by balloons still reads as one line. A
  region is cut at a frame line only if the cut pays off, meaning it sets off a region with
  at most one frame side (title lettering, a caption strip) or lets a part split further. Such a
  strip is not thrown away: balloons and captions between panels are shared at 40% of the
  strip's height between the panel above and the one below. A large strip with no panel above
  it (a title banner) is returned as the first panel: lettering and a panel without a frame look
  the same to the detector, and dropping a real panel (samples/002's bleed art) costs more than
  showing a title. Labels for title pages include the banner for the same reason;
  otherwise lines inside artwork would cut real panels (a plain long-line cut broke
  `western_fighters_8`). This fixed `nature_boy_4`. Still failing: `western_fighters_6` (stained
  paper makes the background estimate useless, so almost the whole page counts as content) and
  `samples/002` (top bleed touching the next row; four of five panels found). Pages whose
  frames are too broken for the lines to be found keep their banners and caption strips.
  These cases are what the ML detector is for.
- **Reading order:** recursive XY-cut honouring `ltr`/`rtl`. `sequence_confidence`
  measures layout ambiguity only (share of panels that could not be cut cleanly);
  per-panel `confidence` measures detection quality.
- **Synchronous processing.** CPU-bound work runs in a thread pool. Async jobs are a
  possible follow-up.
- **Security.** API keys (compared by SHA-256 digest in constant time), in-memory
  sliding-window rate limit (single process; replace with a shared store to scale out),
  SSRF validation on every redirect hop. Each hop resolves the hostname once, rejects the
  request if any answer is non-public, then connects to that validated IP (Host header and
  TLS server name preserved), so a second, different DNS answer cannot redirect the request
  (DNS rebinding). Embedded URL credentials and private IP literals are rejected.
