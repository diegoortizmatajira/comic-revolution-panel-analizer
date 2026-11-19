import pytest
from app.core.errors import ImageTooLargeError, InvalidImageError, UnsupportedImageFormatError
from app.services.image.decoder import ImageLimits, decode_image

from tests.conftest import make_page

LIMITS = ImageLimits(max_bytes=1_000_000, max_pixels=1_000_000)


def test_decodes_png_to_gray_array() -> None:
    img = decode_image(make_page([], size=(30, 20)), LIMITS)
    assert img.gray.shape == (20, 30)
    assert (img.size.width, img.size.height) == (30, 20)


def test_rejects_empty() -> None:
    with pytest.raises(InvalidImageError):
        decode_image(b"", LIMITS)


def test_rejects_garbage() -> None:
    with pytest.raises(InvalidImageError):
        decode_image(b"not an image", LIMITS)


def test_rejects_unsupported_format() -> None:
    with pytest.raises(UnsupportedImageFormatError):
        decode_image(make_page([], size=(10, 10), fmt="GIF"), LIMITS)


def test_rejects_too_many_bytes() -> None:
    with pytest.raises(ImageTooLargeError):
        decode_image(make_page([], size=(50, 50)), ImageLimits(max_bytes=10, max_pixels=10_000))


def test_rejects_too_many_pixels() -> None:
    with pytest.raises(ImageTooLargeError):
        decode_image(
            make_page([], size=(100, 100)), ImageLimits(max_bytes=1_000_000, max_pixels=99)
        )
