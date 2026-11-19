"""Image validation and decoding shared by every input method."""

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image, UnidentifiedImageError

from app.core.errors import ImageTooLargeError, InvalidImageError, UnsupportedImageFormatError
from app.domain.models import ImageSize

if TYPE_CHECKING:
    from numpy.typing import NDArray

ALLOWED_FORMATS = frozenset({"JPEG", "PNG", "WEBP"})


@dataclass(frozen=True, slots=True)
class ImageLimits:
    max_bytes: int
    max_pixels: int


@dataclass(frozen=True, slots=True)
class DecodedImage:
    gray: NDArray[np.uint8]
    size: ImageSize


def decode_image(data: bytes, limits: ImageLimits) -> DecodedImage:
    """Validate by content (not headers) and decode into a grayscale array."""
    if not data:
        msg = "Image is empty."
        raise InvalidImageError(msg)
    if len(data) > limits.max_bytes:
        msg = "Image exceeds the maximum allowed size."
        raise ImageTooLargeError(msg, {"max_bytes": limits.max_bytes})

    try:
        with Image.open(io.BytesIO(data)) as img:
            if img.format not in ALLOWED_FORMATS:
                msg = "Unsupported image format."
                raise UnsupportedImageFormatError(
                    msg, {"allowed": sorted(ALLOWED_FORMATS), "detected": img.format}
                )
            width, height = img.size
            if width * height > limits.max_pixels:  # checked before pixel data is decoded
                msg = "Image dimensions exceed the maximum allowed."
                raise ImageTooLargeError(msg, {"max_pixels": limits.max_pixels})
            gray = np.asarray(img.convert("L"), dtype=np.uint8)
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        msg = "Could not decode the image."
        raise InvalidImageError(msg) from exc

    return DecodedImage(gray=gray, size=ImageSize(width=width, height=height))
