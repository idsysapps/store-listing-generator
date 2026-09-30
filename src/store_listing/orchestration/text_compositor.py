"""Composite text onto generated design images using Pillow.

Renders headline/tagline text into reserved zones on the image canvas,
so the diffusion model only generates artwork and text is pixel-perfect.
"""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO

from PIL import Image, ImageDraw, ImageFont


@dataclass
class LayoutSpec:
    layout_type: str  # "full_bleed", "text_top", "text_top_bottom"
    headline_text: str | None = None
    tagline_text: str | None = None
    font_color: str = "#000000"

    @property
    def needs_text(self) -> bool:
        return bool(self.headline_text or self.tagline_text)


def _fit_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    max_width: int,
    max_height: int,
    font_path: str | None = None,
) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    size = max_height
    while size > 12:
        font: ImageFont.FreeTypeFont | ImageFont.ImageFont
        if font_path:
            font = ImageFont.truetype(font_path, size)
        else:
            try:
                font = ImageFont.truetype("DejaVuSans-Bold.ttf", size)
            except OSError:
                try:
                    font = ImageFont.truetype(
                        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size
                    )
                except OSError:
                    font = ImageFont.load_default(size=size)
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        if text_width <= max_width:
            return font
        size -= 2
    return ImageFont.load_default()


def composite_text(image_bytes: bytes, layout: LayoutSpec) -> bytes:
    if layout.layout_type == "full_bleed" or not layout.needs_text:
        return image_bytes

    img = Image.open(BytesIO(image_bytes)).convert("RGBA")
    w, h = img.size

    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    margin_x = int(w * 0.05)
    max_text_width = w - 2 * margin_x

    if layout.headline_text:
        zone_height = int(h * 0.15)
        font = _fit_text(draw, layout.headline_text, max_text_width, zone_height)
        bbox = draw.textbbox((0, 0), layout.headline_text, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        x = (w - text_w) // 2
        y = (zone_height - text_h) // 2
        draw.text((x, y), layout.headline_text, fill=layout.font_color, font=font)

    if layout.tagline_text:
        zone_height = int(h * 0.12)
        font = _fit_text(draw, layout.tagline_text, max_text_width, zone_height)
        bbox = draw.textbbox((0, 0), layout.tagline_text, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        x = (w - text_w) // 2
        y = h - zone_height + (zone_height - text_h) // 2
        draw.text((x, y), layout.tagline_text, fill=layout.font_color, font=font)

    result = Image.alpha_composite(img, overlay)
    buf = BytesIO()
    result.save(buf, format="PNG")
    return buf.getvalue()
