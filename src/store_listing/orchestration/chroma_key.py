"""Key out a flat background colour to produce a transparent PNG.

Design briefs are rendered by Flux on a flat, saturated background so it can be
keyed out exactly. A chroma key is used rather than a segmentation model
because it does three things a salient-object model does not:

* it removes the enclosed counters of letters like O, R, A and P exactly -
  they are the background colour, so they key out with it;
* it keeps white artwork (a cream skeleton on white) perfectly intact;
* it needs no model on the DGX, so it is faster and cannot invert.

The key colour is estimated from the image border, which is background by
construction, and the distance is measured in YCbCr so that a *darker shade* of
the key colour - artwork the model happens to tint towards the background - is
not mistaken for the background itself.
"""

from __future__ import annotations

import io
import logging

from PIL import Image, ImageChops

logger = logging.getLogger(__name__)

# The distance ramp: at or below TOLERANCE_IN the pixel is background, at or
# above TOLERANCE_OUT it is fully opaque artwork.
DEFAULT_TOLERANCE_IN = 25
DEFAULT_TOLERANCE_OUT = 55

# How many border pixels to sample when estimating the key colour.
BORDER_SAMPLE_STEP = 4


def estimate_key_colour(image: Image.Image) -> tuple[int, int, int]:
    """Estimate the background colour as the per-channel median of the border.

    The border of a rendered design is background by construction, and the
    median ignores the odd pixel where the artwork runs off the edge.
    """
    rgb = image.convert("RGB")
    width, height = rgb.size
    values = rgb.tobytes()
    step = BORDER_SAMPLE_STEP
    channels: list[list[int]] = [[], [], []]
    for x in range(0, width, step):
        for y in (0, height - 1):
            offset = (y * width + x) * 3
            for index in range(3):
                channels[index].append(values[offset + index])
    for y in range(0, height, step):
        for x in (0, width - 1):
            offset = (y * width + x) * 3
            for index in range(3):
                channels[index].append(values[offset + index])
    return tuple(sorted(channel)[len(channel) // 2] for channel in channels)  # type: ignore[return-value]


def _key_in_ycbcr(key: tuple[int, int, int]) -> tuple[int, int, int]:
    r, g, b = key
    y = int(0.299 * r + 0.587 * g + 0.114 * b)
    cb = int(128 - 0.168736 * r - 0.331264 * g + 0.5 * b)
    cr = int(128 + 0.5 * r - 0.418688 * g - 0.081312 * b)
    return y, cb, cr


def _distance_image(rgb: Image.Image, key: tuple[int, int, int]) -> Image.Image:
    """How far every pixel is from the key colour, in YCbCr.

    The per-channel difference is combined with a *max* rather than a sum, which
    makes this a box around the key colour in chroma space - the standard
    chroma-key shape. It also keeps the whole keying pass in C instead of a
    per-pixel Python loop.
    """
    y, cb, cr = rgb.convert("YCbCr").split()
    key_y, key_cb, key_cr = _key_in_ycbcr(key)
    size = rgb.size
    return ImageChops.lighter(
        ImageChops.lighter(
            ImageChops.difference(y, Image.new("L", size, key_y)),
            ImageChops.difference(cb, Image.new("L", size, key_cb)),
        ),
        ImageChops.difference(cr, Image.new("L", size, key_cr)),
    )


def _alpha_table(tolerance_in: int, tolerance_out: int) -> list[int]:
    span = max(1, tolerance_out - tolerance_in)
    table: list[int] = []
    for distance in range(256):
        if distance <= tolerance_in:
            table.append(0)
        elif distance >= tolerance_out:
            table.append(255)
        else:
            table.append(int(255 * (distance - tolerance_in) / span))
    return table


def _despill(rgb: Image.Image, alpha: Image.Image, key: tuple[int, int, int]) -> Image.Image:
    """Remove the key colour's contribution from the partially transparent pixels.

    A pixel on the edge is a blend of the artwork and the key colour. Once the
    alpha says how much of it is artwork, the artwork colour can be recovered:
    ``F = (O - (1 - a) * K) / a``. Without this the edges keep a fringe of the
    key colour.
    """
    width, height = rgb.size
    rgb_values = rgb.tobytes()
    alpha_values = alpha.tobytes()
    out = bytearray(rgb_values)
    for index, value in enumerate(alpha_values):
        if value == 0 or value == 255:
            continue
        fraction = value / 255
        base = index * 3
        for channel in range(3):
            corrected = (rgb_values[base + channel] - (1 - fraction) * key[channel]) / fraction
            out[base + channel] = max(0, min(255, int(corrected)))
    return Image.frombytes("RGB", (width, height), bytes(out))


def key_out_background(
    image_bytes: bytes,
    *,
    key_colour: tuple[int, int, int] | None = None,
    tolerance_in: int = DEFAULT_TOLERANCE_IN,
    tolerance_out: int = DEFAULT_TOLERANCE_OUT,
    despill: bool = True,
) -> bytes:
    """Return an RGBA PNG with the flat background keyed out."""
    with Image.open(io.BytesIO(image_bytes)) as opened:
        rgb = opened.convert("RGB")

    key = key_colour or estimate_key_colour(rgb)
    distance = _distance_image(rgb, key)
    alpha = distance.point(_alpha_table(tolerance_in, tolerance_out))

    if despill:
        rgb = _despill(rgb, alpha, key)

    result = rgb.convert("RGBA")
    result.putalpha(alpha)
    buffer = io.BytesIO()
    result.save(buffer, format="PNG")
    return buffer.getvalue()


def background_fraction(
    image_bytes: bytes, *, key_colour: tuple[int, int, int] | None = None
) -> float:
    """How much of the image the key colour covers (0..1).

    Used as a sanity check: a render that was supposed to have a flat
    background but did not will key out an implausible fraction.
    """
    with Image.open(io.BytesIO(image_bytes)) as opened:
        rgb = opened.convert("RGB")
    key = key_colour or estimate_key_colour(rgb)
    distance = _distance_image(rgb, key)
    pixels = distance.tobytes()
    close = sum(1 for value in pixels if value <= DEFAULT_TOLERANCE_IN)
    return close / len(pixels)


def punch_out_key_colour(
    rgba_bytes: bytes,
    source_bytes: bytes,
    *,
    key_colour: tuple[int, int, int] | None = None,
    tolerance: int = DEFAULT_TOLERANCE_IN,
) -> bytes:
    """Clear the alpha wherever the source pixel *is* the flat background colour.

    A segmentation model treats the enclosed counter of a letter as part of the
    subject, so those pixels keep the background colour and come out as dark
    blobs. They are exactly the key colour, so they can be punched out - but
    only exactly: the anti-aliased edge pixels are blends of the artwork and
    the key, and punching them out would harden the edge.
    """
    with Image.open(io.BytesIO(source_bytes)) as opened:
        source = opened.convert("RGB")
    with Image.open(io.BytesIO(rgba_bytes)) as opened:
        result = opened.convert("RGBA")

    key = key_colour or estimate_key_colour(source)
    distance = _distance_image(source, key)

    def is_key(value: int) -> int:
        return 255 if value <= tolerance else 0

    is_key_mask = distance.point(is_key)

    alpha = result.getchannel("A")
    combined = ImageChops.multiply(alpha, ImageChops.invert(is_key_mask))
    result.putalpha(combined)

    buffer = io.BytesIO()
    result.save(buffer, format="PNG")
    return buffer.getvalue()
