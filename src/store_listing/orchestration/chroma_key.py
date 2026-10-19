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

from PIL import Image, ImageChops, ImageDraw, ImageFilter

logger = logging.getLogger(__name__)

# The distance ramp: at or below TOLERANCE_IN the pixel is background, at or
# above TOLERANCE_OUT it is fully opaque artwork.
DEFAULT_TOLERANCE_IN = 25
DEFAULT_TOLERANCE_OUT = 55

# How close to the key colour a pixel has to be to count as the background.
# Tight, so that a darker shade of the key in the artwork is not mistaken for
# it.
KEY_TOLERANCE = 20

# How far the background may reach into the artwork's anti-aliased edge. Only
# used to take the rim off the inside of a hole.
GROW_TOLERANCE = 95

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


def despill_key_spill(
    rgba_bytes: bytes,
    source_bytes: bytes,
    *,
    key_colour: tuple[int, int, int] | None = None,
    strength: float = 1.0,
) -> bytes:
    """Remove the key colour's contribution from the partially transparent pixels.

    A pixel on the edge of the artwork is a blend of the artwork and the key.
    Subtracting the key's share of it stops the edges carrying a tint of the
    background colour, which on a shirt shows up as a coloured fringe.

    The correction is deliberately *not* divided back by the alpha. The alpha
    comes from a segmentation matte, not from an exact mix fraction, so
    dividing by it amplifies the error into nonsense colours. Subtracting only
    the key's share is conservative: it can never over-correct.
    """
    with Image.open(io.BytesIO(source_bytes)) as opened:
        source = opened.convert("RGB")
    with Image.open(io.BytesIO(rgba_bytes)) as opened:
        result = opened.convert("RGBA")

    key = key_colour or estimate_key_colour(source)
    width, height = result.size
    rgb_values = result.convert("RGB").tobytes()
    alpha_values = result.getchannel("A").tobytes()
    out = bytearray(width * height * 4)
    for index, value in enumerate(alpha_values):
        source_base = index * 3
        target_base = index * 4
        out[target_base + 3] = value
        if value == 0 or value == 255:
            out[target_base] = rgb_values[source_base]
            out[target_base + 1] = rgb_values[source_base + 1]
            out[target_base + 2] = rgb_values[source_base + 2]
            continue
        share = (1 - value / 255) * strength
        for channel in range(3):
            corrected = rgb_values[source_base + channel] - share * key[channel]
            out[target_base + channel] = max(0, min(255, int(corrected)))
    despilled = Image.frombytes("RGBA", (width, height), bytes(out))
    buffer = io.BytesIO()
    despilled.save(buffer, format="PNG")
    return buffer.getvalue()


def punch_out_key_colour(
    rgba_bytes: bytes,
    source_bytes: bytes,
    *,
    key_colour: tuple[int, int, int] | None = None,
    tolerance: int = KEY_TOLERANCE,
    grow_tolerance: int = GROW_TOLERANCE,
    grow_rounds: int = 2,
) -> bytes:
    """Clear the background out of the mask, holes included.

    A segmentation model treats the enclosed counter of a letter as part of the
    subject, so those pixels keep the background colour. They are exactly the
    key colour, so they can be punched out.

    The counters are then found as *enclosed* regions of the key colour: the
    background reachable from the edge of the image is the outer background,
    and anything key-coloured that the edge cannot reach is a hole in the
    artwork - a letter's counter. That distinction is what lets the
    anti-aliased rim of a counter be removed without touching the outer edge
    of the artwork, which looks identical pixel for pixel.
    """
    with Image.open(io.BytesIO(source_bytes)) as opened:
        source = opened.convert("RGB")
    with Image.open(io.BytesIO(rgba_bytes)) as opened:
        result = opened.convert("RGBA")

    key = key_colour or estimate_key_colour(source)
    distance = _distance_image(source, key)

    def is_key(value: int) -> int:
        return 0 if value <= tolerance else 255

    # 0 = key coloured, 255 = artwork.
    region = distance.point(is_key)
    seed = _find_key_pixel_on_border(region)
    if seed is not None:
        # 128 = the key colour reachable from the edge, i.e. the outer background.
        ImageDraw.floodfill(region, seed, 128, thresh=0)

    def is_hole(value: int) -> int:
        return 255 if value == 0 else 0

    def is_outer(value: int) -> int:
        return 255 if value == 128 else 0

    hole = region.point(is_hole)
    outer = region.point(is_outer)

    def is_growable(value: int) -> int:
        return 255 if value <= grow_tolerance else 0

    growable = distance.point(is_growable)
    for _ in range(max(0, grow_rounds)):
        adjacent = hole.filter(ImageFilter.MaxFilter(3))
        grown = ImageChops.multiply(adjacent, growable)
        if grown.getbbox() is None:
            break
        hole = ImageChops.lighter(hole, grown)
        growable = ImageChops.subtract(growable, grown)

    background = ImageChops.lighter(hole, outer)
    alpha = result.getchannel("A")
    result.putalpha(ImageChops.multiply(alpha, ImageChops.invert(background)))

    buffer = io.BytesIO()
    result.save(buffer, format="PNG")
    return buffer.getvalue()


def _find_key_pixel_on_border(region: Image.Image) -> tuple[int, int] | None:
    """A key-coloured pixel on the edge of the image, to flood the background from."""
    width, height = region.size
    values = region.tobytes()
    for x in range(width):
        for y in (0, height - 1):
            if values[y * width + x] == 0:
                return (x, y)
    for y in range(height):
        for x in (0, width - 1):
            if values[y * width + x] == 0:
                return (x, y)
    return None
