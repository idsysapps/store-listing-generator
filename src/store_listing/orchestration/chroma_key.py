"""Key out a flat background colour to produce a transparent PNG.

Design briefs are rendered by Flux on a flat, saturated background so it can be
keyed out exactly. A chroma key is used rather than a segmentation model
because it does three things a salient-object model does not:

* it removes the enclosed counters of letters like O, R, A and P exactly -
  they are the background colour, so they key out with it;
* it keeps white artwork (a cream skeleton on white) perfectly intact;
* it needs no model on the DGX, so it is faster and cannot invert.

The key colour is estimated from the image border, which is background by
construction.

The distance is measured in the U/V (chroma) plane only, the way ffmpeg's
``chromakey`` filter does it, so the luma is ignored. That is what keeps a
near-black detail in the artwork - the skeleton's eye sockets, the pumpkin's
dark mouth - safe: their chroma is neutral, so they sit as far from a saturated
key as the white artwork does. Measuring the luma as well would put a near-black
detail *closer* to a dark key than the white artwork is, and it would be keyed
out.
"""

from __future__ import annotations

import io
import logging

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter

logger = logging.getLogger(__name__)

# Keying parameters, in the same units ffmpeg's chromakey filter uses: the
# distance is normalised so that 1.0 is the full chroma range.
#
# A pixel is background when its distance is at or below SIMILARITY, and fully
# opaque artwork when it is at or above SIMILARITY + BLEND. The ramp between
# the two is the soft edge.
#
# Measured on a rendered design: the key and the background sit at 0.00, the
# magenta-tinted edges of the artwork at 0.12, and the artwork itself - white
# text, the orange pumpkin, the cream skeleton, the near-black eye sockets and
# the pumpkin's dark mouth - at 0.30 and above. So 0.13 is a wide margin below
# the artwork and a wide one above the background.
SIMILARITY = 0.13
BLEND = 0.05

# The tolerance for counting a pixel as the background itself, which is what
# finds the holes: a letter's counter is enclosed by the artwork, so the flood
# from the edge of the image cannot reach it.
HOLE_SIMILARITY = 0.03

# How far the background may reach into the artwork's anti-aliased edge. Only
# used to take the rim off the inside of a hole.
GROW_SIMILARITY = 0.16

# The pixels the alpha correction looks at: anything that still carries some of
# the key colour. Measured on a rendered design the artwork's closest pixel sits
# at 0.25, so this is a safe margin below it.
KEY_MIX_SIMILARITY = 0.20

# How many border pixels to sample when estimating the key colour.
BORDER_SAMPLE_STEP = 4

# The chroma distance is kept in an 8-bit image so the masks can be built with
# ImageChops; 255 is the full range.
_SCALE = 255.0


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


def _chroma_distance(rgb: Image.Image, key: tuple[int, int, int]) -> np.ndarray:
    """The chroma distance from every pixel to the key colour, normalised to 0..1.

    The distance is measured in the U/V (chroma) plane only, so the luma is
    ignored. That is what keeps a near-black detail in the artwork - the
    skeleton's eye sockets, the pumpkin's dark mouth - safe: their chroma is
    neutral, so they sit as far from a saturated key as the white artwork does.

    Measuring the luma as well would put a near-black detail *closer* to a dark
    key than the white artwork is, and it would get keyed out.
    """
    ycbcr = np.asarray(rgb.convert("YCbCr")).astype(np.float64)
    key_ycbcr = np.asarray(
        Image.new("RGB", (1, 1), key).convert("YCbCr").getpixel((0, 0)), dtype=np.float64
    )
    du = ycbcr[:, :, 1] - key_ycbcr[1]
    dv = ycbcr[:, :, 2] - key_ycbcr[2]
    return np.sqrt(du * du + dv * dv) / (255.0 * np.sqrt(2.0))


def _distance_image(rgb: Image.Image, key: tuple[int, int, int]) -> Image.Image:
    """The chroma distance as an 8-bit image, scaled so 255 is the full range.

    Kept as an image so the masks can be built with ``ImageChops``.
    """
    distance = _chroma_distance(rgb, key)
    scaled = np.clip(distance * _SCALE, 0, 255).astype(np.uint8)
    return Image.fromarray(scaled, mode="L")


def _alpha_table(similarity: float, blend: float) -> list[int]:
    """The alpha ramp, as ffmpeg's chromakey defines it."""
    inner = similarity * _SCALE
    span = max(1.0, blend * _SCALE)
    table: list[int] = []
    for value in range(256):
        table.append(int(max(0.0, min(1.0, (value - inner) / span)) * 255))
    return table


def key_out_background(
    image_bytes: bytes,
    *,
    key_colour: tuple[int, int, int] | None = None,
    similarity: float = SIMILARITY,
    blend: float = BLEND,
    despill: bool = True,
) -> bytes:
    """Return an RGBA PNG with the flat background keyed out."""
    with Image.open(io.BytesIO(image_bytes)) as opened:
        rgb = opened.convert("RGB")
    width, height = rgb.size

    key = key_colour or estimate_key_colour(rgb)
    distance = _distance_image(rgb, key)
    alpha = distance.point(_alpha_table(similarity, blend))

    if despill:
        alpha_values = alpha.tobytes()
        rgb_values = rgb.tobytes()
        out = bytearray(width * height * 4)
        for index, value in enumerate(alpha_values):
            base = index * 3
            target = index * 4
            out[target + 3] = value
            if value == 0 or value == 255:
                out[target] = rgb_values[base]
                out[target + 1] = rgb_values[base + 1]
                out[target + 2] = rgb_values[base + 2]
                continue
            share = 1 - value / 255
            for channel in range(3):
                corrected = rgb_values[base + channel] - share * key[channel]
                out[target + channel] = max(0, min(255, int(corrected)))
        keyed = Image.frombytes("RGBA", (width, height), bytes(out))
        buffer = io.BytesIO()
        keyed.save(buffer, format="PNG")
        return buffer.getvalue()

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
    distance = _chroma_distance(rgb, key)
    return float((distance <= HOLE_SIMILARITY).mean())


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


def correct_key_mix(
    rgba_bytes: bytes,
    source_bytes: bytes,
    *,
    key_colour: tuple[int, int, int] | None = None,
    similarity: float = KEY_MIX_SIMILARITY,
    search_radius: int = 6,
) -> bytes:
    """Recover the artwork fraction of the pixels that still carry the key colour.

    A pixel on the edge of the artwork is a blend of the artwork and the key:
    ``O = a * F + (1 - a) * K``. The segmentation model over-estimates ``a`` on
    these pixels - it is confident that a letter is solid - so they keep a tint
    of the background colour.

    Given the foreground colour ``F`` of the neighbourhood, ``a`` can be solved
    for directly, and the pixel's colour replaced with the foreground colour so
    no background tint is left in the alpha. ``F`` is taken from the nearest
    opaque pixel that is not itself the key colour, which is the artwork the
    pixel is blending into.
    """
    with Image.open(io.BytesIO(source_bytes)) as opened:
        source = opened.convert("RGB")
    with Image.open(io.BytesIO(rgba_bytes)) as opened:
        result = opened.convert("RGBA")

    key = key_colour or estimate_key_colour(source)
    width, height = result.size
    source_values = np.asarray(source).astype(np.float64)
    distance = _chroma_distance(source, key)
    alpha = np.asarray(result.getchannel("A")).astype(np.float64)
    rgb = np.asarray(result.convert("RGB")).astype(np.float64)

    # Every pixel that still carries some of the key colour is a candidate,
    # whether the model called it fully opaque or not.
    candidates = (alpha > 0) & (distance <= similarity)
    if not candidates.any():
        return rgba_bytes

    key_vector = np.asarray(key, dtype=np.float64)
    # The foreground colour of each candidate: the nearest opaque pixel that is
    # not the key colour. Candidates are tracked by their flat index so that the
    # candidate and the neighbour that supplied the colour cannot be confused.
    flat_candidates = np.flatnonzero(candidates.ravel())
    foreground = np.zeros((flat_candidates.size, 3), dtype=np.float64)
    found = np.zeros(flat_candidates.size, dtype=bool)
    opaque = ((alpha >= 250) & (distance > similarity)).ravel()

    for radius in range(1, search_radius + 1):
        pending = np.flatnonzero(~found)
        if pending.size == 0:
            break
        base = flat_candidates[pending]
        cy = base // width
        cx = base % width
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if abs(dy) != radius and abs(dx) != radius:
                    continue
                sy = cy + dy
                sx = cx + dx
                inside = (sy >= 0) & (sy < height) & (sx >= 0) & (sx < width)
                if not inside.any():
                    continue
                # Clamp before indexing; `inside` still masks the edges out.
                neighbour = np.clip(sy, 0, height - 1) * width + np.clip(sx, 0, width - 1)
                hit = inside & opaque[neighbour]
                if not hit.any():
                    continue
                target = np.flatnonzero(hit)
                foreground[pending[target]] = source_values.reshape(-1, 3)[neighbour[target]]
                found[pending[target]] = True

    if not found.any():
        return rgba_bytes

    observed = source_values.reshape(-1, 3)[flat_candidates[found]]
    fore = foreground[found]
    direction = fore - key_vector
    denominator = (direction**2).sum(axis=1)
    numerator = ((observed - key_vector) * direction).sum(axis=1)
    fraction = np.where(denominator > 0, numerator / np.maximum(denominator, 1e-6), 1.0)
    fraction = np.clip(fraction, 0.0, 1.0)

    out_alpha = alpha.copy()
    out_rgb = rgb.copy()
    corrected = np.flatnonzero(found)
    target = flat_candidates[corrected]
    out_alpha.ravel()[target] = fraction * 255
    out_rgb.ravel()[target * 3] = fore[:, 0]
    out_rgb.ravel()[target * 3 + 1] = fore[:, 1]
    out_rgb.ravel()[target * 3 + 2] = fore[:, 2]

    out = np.dstack([out_rgb, out_alpha]).astype(np.uint8)
    result = Image.frombytes("RGBA", (width, height), out.tobytes())
    buffer = io.BytesIO()
    result.save(buffer, format="PNG")
    return buffer.getvalue()


def punch_out_key_colour(
    rgba_bytes: bytes,
    source_bytes: bytes,
    *,
    key_colour: tuple[int, int, int] | None = None,
    similarity: float = HOLE_SIMILARITY,
    grow_similarity: float = GROW_SIMILARITY,
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

    inner = int(similarity * _SCALE)

    def is_key(value: int) -> int:
        return 0 if value <= inner else 255

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

    grow_inner = int(grow_similarity * _SCALE)

    def is_growable(value: int) -> int:
        return 255 if value <= grow_inner else 0

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
