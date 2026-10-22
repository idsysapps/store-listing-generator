"""The texture of a design.

A look that is drawn as a flat vector reads as clip-art. The grain, the
halftone dots and the cracked ink are what make it read as a print.

They cannot be asked of the image model: a texture it paints lands on the
background too, and the background of a DTF render is keyed out - a paper
texture would survive the cutout as a grey rectangle and would wreck the
despill (issue #227).

So the texture is applied here, on the artwork, after the keying. The
background is never touched: on a keyed render the texture is masked with
the distance from the key colour, and on a transparent render it is masked
with the alpha.
"""

from __future__ import annotations

import logging
import math
from io import BytesIO
from typing import Final

import numpy as np
from PIL import Image, ImageDraw

logger = logging.getLogger(__name__)

# How strong the grain is, as a share of the colour.
GRAIN_STRENGTH: Final[float] = 0.10
# The distance between two halftone dots, in pixels.
HALFTONE_SPACING: Final[int] = 6
# The radius of a halftone dot, in pixels.
HALFTONE_RADIUS: Final[float] = 1.7
# How strongly a halftone dot darkens the art.
HALFTONE_STRENGTH: Final[float] = 0.18
# How many scratches a cracked print carries.
CRACKLE_COUNT: Final[int] = 260
# The longest scratch, in pixels.
CRACKLE_LENGTH: Final[int] = 46
# The seed of the noise, so the same artwork always gets the same texture.
TEXTURE_SEED: Final[int] = 20261021
# The distance from the key colour at which a pixel counts as artwork.
KEY_DISTANCE: Final[float] = 40.0

__all__ = ["apply_texture"]


def _offset(texture: str, width: int, height: int, rng: np.random.Generator) -> np.ndarray | None:
    """The colour every pixel of the artwork is shifted by.

    The shift is an offset and not a factor, because a factor can only darken
    a colour and the black of a line drawing has to be able to break up too.
    """
    if texture == "grain":
        noise = rng.normal(0.0, GRAIN_STRENGTH, size=(height, width))
        return noise * 255.0

    if texture == "halftone":
        rows, columns = np.mgrid[0:height, 0:width]
        offset_x = columns % HALFTONE_SPACING - HALFTONE_SPACING / 2
        offset_y = rows % HALFTONE_SPACING - HALFTONE_SPACING / 2
        distance = np.sqrt(offset_x**2 + offset_y**2)
        dots = (distance < HALFTONE_RADIUS).astype(float)
        return -HALFTONE_STRENGTH * 255.0 * dots

    if texture == "crackle":
        scratches = Image.new("L", (width, height), 0)
        draw = ImageDraw.Draw(scratches)
        for _ in range(CRACKLE_COUNT):
            start_x = int(rng.integers(0, width))
            start_y = int(rng.integers(0, height))
            angle = float(rng.random()) * math.pi
            length = int(rng.integers(4, CRACKLE_LENGTH))
            end_x = int(start_x + math.cos(angle) * length)
            end_y = int(start_y + math.sin(angle) * length)
            draw.line([start_x, start_y, end_x, end_y], fill=255, width=1)
        lines = np.asarray(scratches).astype(np.float32) / 255.0
        return -0.25 * 255.0 * lines

    logger.warning("Unknown texture %r, drawing the artwork without it", texture)
    return None


def apply_texture(
    image_bytes: bytes,
    texture: str | None,
    *,
    key_colour: tuple[int, int, int] | None = None,
) -> bytes:
    """The artwork with the texture of its look on it.

    ``key_colour`` is the colour of the background of a render that has not
    been keyed out. The texture is not painted on that background.
    """
    if not texture:
        return image_bytes

    try:
        image = Image.open(BytesIO(image_bytes)).convert("RGBA")
    except OSError as error:
        logger.warning("Cannot apply the texture: %s", error)
        return image_bytes

    pixels = np.asarray(image).astype(np.float32)
    rgb = pixels[:, :, :3]
    alpha = pixels[:, :, 3]
    height, width = rgb.shape[:2]

    rng = np.random.default_rng(TEXTURE_SEED)
    offset = _offset(texture, width, height, rng)
    if offset is None:
        return image_bytes

    if key_colour is not None:
        distance = np.abs(rgb - np.array(key_colour, dtype=np.float32)).sum(axis=2)
        mask = (distance > KEY_DISTANCE).astype(np.float32)
    else:
        mask = (alpha > 0).astype(np.float32)

    textured = np.clip(rgb + offset[:, :, None], 0.0, 255.0)
    blended = rgb * (1.0 - mask[:, :, None]) + textured * mask[:, :, None]

    result = np.dstack([blended, alpha[:, :, None]]).astype(np.uint8)
    buffer = BytesIO()
    Image.fromarray(result, mode="RGBA").save(buffer, format="PNG")
    return buffer.getvalue()
