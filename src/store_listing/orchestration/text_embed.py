"""DTF-safe text embedding.

The compositor still draws every letter from a font, so spelling is exact.
This module only treats the pixels of that text layer: it warps the glyphs by
the surface texture, shades them with the luminance of the carrier, and knocks
out binary distress holes.

The important DTF rule is here: a pixel is either ink (100% opacity) or no ink
(0% opacity). Texture and distress never lower the opacity of an inked pixel,
because a semi-transparent letter prints as a thin white underbase and a
washed-out colour on a dark garment.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Final

import numpy as np
from PIL import Image, ImageChops, ImageFilter

logger = logging.getLogger(__name__)

__all__ = [
    "PROFILES",
    "TextEmbedProfile",
    "composite_embedded",
    "embed_layer",
    "profile_for",
]


@dataclass(frozen=True)
class TextEmbedProfile:
    """How strongly the text is integrated into the carrier."""

    name: str
    # Displacement amplitude at the profile's working scale.
    warp: float
    # Low-frequency curve, in working-scale pixels.
    arc: float
    # Soft-light strength from the carrier luminance.
    texture: float
    # Multiply strength from the blurred carrier luminance.
    lighting: float
    # Share of the coarse wear field that becomes a knockout.
    wear: float
    # Strength of the high-pass edge knockouts.
    alpha_hpass: float
    # Fine colour grain.
    grain: float
    # Supersampling factor used by the compositor.
    scale: int = 4


PROFILES: Final[dict[str, TextEmbedProfile]] = {
    "v11": TextEmbedProfile(
        name="v11",
        warp=10.0,
        arc=10.0,
        texture=0.10,
        lighting=0.18,
        wear=0.55,
        alpha_hpass=0.25,
        grain=0.06,
    ),
    "v12": TextEmbedProfile(
        name="v12",
        warp=8.0,
        arc=8.0,
        texture=0.08,
        lighting=0.14,
        wear=0.45,
        alpha_hpass=0.18,
        grain=0.08,
    ),
    "v13": TextEmbedProfile(
        name="v13",
        warp=12.0,
        arc=12.0,
        texture=0.12,
        lighting=0.20,
        wear=0.65,
        alpha_hpass=0.30,
        grain=0.12,
    ),
}


def profile_for(name: str | None) -> TextEmbedProfile | None:
    """Resolve a feature flag to a profile. Unknown values fail safe to off."""
    if not name or name.strip().lower() in {"off", "none", "false", "0"}:
        return None
    profile = PROFILES.get(name.strip().lower())
    if profile is None:
        logger.warning("Unknown TEXT_EMBED_PROFILE=%r, leaving text embedding off", name)
    return profile


def embed_layer(
    foreground: Image.Image,
    background: Image.Image,
    profile: TextEmbedProfile,
) -> Image.Image:
    """Turn a font-rendered RGBA text layer into a DTF-safe inked layer.

    ``background`` is the carrier crop at the same size as ``foreground``.
    The returned alpha channel contains only 0 or 255: no semi-transparent
    ink is allowed. The RGB channel may be shaded and grained.
    """
    if foreground.size != background.size:
        raise ValueError("foreground and background must have the same size")

    foreground = foreground.convert("RGBA")
    background = background.convert("RGB")
    width, height = foreground.size
    if width < 2 or height < 2:
        return foreground

    luminance = background.convert("L")
    gray = np.asarray(luminance, dtype=np.float32) / 255.0
    blurred = (
        np.asarray(
            luminance.filter(ImageFilter.GaussianBlur(max(2, min(width, height) // 12))),
            dtype=np.float32,
        )
        / 255.0
    )
    high = gray - blurred
    high = high / max(1e-6, float(np.abs(high).max()))

    edges = np.asarray(luminance.filter(ImageFilter.FIND_EDGES), dtype=np.float32) / 255.0
    edges = (
        np.asarray(
            Image.fromarray((edges * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(2)),
            dtype=np.float32,
        )
        / 255.0
    )
    edges = edges / max(1e-6, float(edges.max()))

    rng = np.random.default_rng(11)
    fine = rng.normal(0, 1, (height, width)).astype(np.float32)
    fine = _blurred_noise(fine, 1.2)
    fine = fine / max(1e-6, float(np.abs(fine).max()))

    coarse_height = max(1, height // 14 + 1)
    coarse_width = max(1, width // 14 + 1)
    coarse = rng.normal(0, 1, (coarse_height, coarse_width)).astype(np.float32)
    coarse = _normalise(
        np.asarray(
            Image.fromarray(
                ((coarse - coarse.min()) / (coarse.max() - coarse.min()) * 255).astype(np.uint8)
            ).resize((width, height), Image.Resampling.BILINEAR),
            dtype=np.float32,
        )
        / 255.0
    )

    bump = 0.65 * high + 0.25 * edges + 0.20 * fine
    bump = bump / max(1e-6, float(np.abs(bump).max()))

    xx = np.arange(width, dtype=np.float32)[None, :]
    centre_x = width / 2
    dx = profile.warp * bump
    dy = (
        profile.warp * (0.55 * bump + 0.25 * fine)
        + profile.arc * ((xx - centre_x) / max(1.0, centre_x)) ** 2
    )
    warped = _warp(foreground, dx, dy)

    rgb = warped.convert("RGB")
    if profile.texture:
        gray_rgb = Image.merge("RGB", (luminance, luminance, luminance))
        rgb = Image.blend(rgb, ImageChops.soft_light(rgb, gray_rgb), profile.texture)
    if profile.lighting:
        light = Image.merge(
            "RGB",
            (luminance.point([min(255, 185 + value // 5) for value in range(256)]),) * 3,
        )
        rgb = Image.blend(rgb, ImageChops.multiply(rgb, light), profile.lighting)

    colour = np.asarray(rgb, dtype=np.float32)
    if profile.grain:
        colour = colour * (1.0 + profile.grain * fine[..., None])
    rgb = Image.fromarray(np.clip(colour, 0, 255).astype(np.uint8)).convert("RGBA")

    alpha = np.asarray(warped.getchannel("A"), dtype=np.float32)
    # Every inked pixel stays at full opacity. Anti-aliased font edges are
    # thresholded here so the print has a hard ink boundary.
    alpha = np.where(alpha > 127, 255.0, 0.0)

    # The wear is binary too: coarse surface patches and high-pass edges
    # become holes, never partial ink.
    if profile.wear or profile.alpha_hpass:
        coarse_threshold = 0.82 - 0.55 * profile.wear
        hole = coarse > coarse_threshold
        if profile.alpha_hpass:
            hole |= np.abs(high) > (0.90 - 0.45 * profile.alpha_hpass)
        hole |= (fine > 0.93) & (np.abs(bump) > 0.35)
        alpha[hole] = 0.0

    rgb.putalpha(Image.fromarray(alpha.astype(np.uint8)))
    return rgb


def composite_embedded(
    background: Image.Image,
    foreground: Image.Image,
    profile: TextEmbedProfile,
) -> Image.Image:
    """Embed a text layer over its carrier crop."""
    layer = embed_layer(foreground, background, profile)
    return Image.alpha_composite(background.convert("RGBA"), layer)


def _normalise(values: np.ndarray) -> np.ndarray:
    values = values - float(values.mean())
    return values / max(1e-6, float(np.abs(values).max()))


def _blurred_noise(noise: np.ndarray, radius: float) -> np.ndarray:
    low = float(noise.min())
    high = float(noise.max())
    scaled = (noise - low) / max(1e-6, high - low)
    image = Image.fromarray((scaled * 255).astype(np.uint8)).filter(
        ImageFilter.GaussianBlur(radius)
    )
    return _normalise(np.asarray(image, dtype=np.float32) / 255.0)


def _warp(layer: Image.Image, dx: np.ndarray, dy: np.ndarray) -> Image.Image:
    """Bilinear remap of an RGBA layer."""
    source = np.asarray(layer, dtype=np.float32)
    height, width = source.shape[:2]
    yy, xx = np.mgrid[0:height, 0:width]
    sample_x = np.clip(xx + dx, 0, width - 1)
    sample_y = np.clip(yy + dy, 0, height - 1)
    x0 = np.floor(sample_x).astype(np.int32)
    y0 = np.floor(sample_y).astype(np.int32)
    x1 = np.minimum(x0 + 1, width - 1)
    y1 = np.minimum(y0 + 1, height - 1)
    wx = (sample_x - x0)[..., None]
    wy = (sample_y - y0)[..., None]
    output = (
        source[y0, x0] * (1 - wx) * (1 - wy)
        + source[y0, x1] * wx * (1 - wy)
        + source[y1, x0] * (1 - wx) * wy
        + source[y1, x1] * wx * wy
    )
    return Image.fromarray(np.clip(output, 0, 255).astype(np.uint8), "RGBA")
