"""Image generation from design briefs via ComfyUI + Flux Schnell on the DGX.

Leonardo.ai, the Ideogram text-routing workaround and the AWS Bedrock image
calls were removed in #111: the self-hosted DGX is the only generation path.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from store_listing.orchestration.art_direction import negative_for, preset_for, prompt_for
from store_listing.orchestration.comfyui_client import ComfyUIClient, ComfyUIError
from store_listing.orchestration.design_briefs import DesignBrief
from store_listing.orchestration.scene_text import strip_lettering
from store_listing.orchestration.text_compositor import resolve_layout

logger = logging.getLogger(__name__)

PRODUCT_TYPE_CONTEXT: dict[str, str] = {
    # dtf_apparel is the one product whose background is removed, so it is
    # rendered on a flat saturated magenta that can be keyed out exactly.
    # A dark background does not work: the background removal then fills the
    # enclosed counters of letters like O, R and A instead of cutting them out.
    # Magenta also keeps white artwork and white text visible. See issue #111.
    "dtf_apparel": (
        "sticker illustration, bold outlines, flat colors, compact centered composition, "
        "DTF transfer print"
    ),
    # These are printed full-bleed, so their background is kept and has to be
    # white - it becomes the paper.
    "sublimation": "full-color sublimation print, seamless edges, vibrant colors",
    "sticker_vinyl": "sticker decal, clean cut lines, bold outlines",
}

# The background of the artwork, per product type. It is the last instruction
# of the prompt: a visual style that asks for a black background must not win
# over the white paper a print-through product is printed on (issue #188).
BACKGROUND_INSTRUCTIONS: dict[str, str] = {
    "dtf_apparel": (
        "isolated on a solid pure magenta background, plain magenta backdrop, "
        "chroma key background, uniform flat magenta color"
    ),
    "sublimation": "isolated on a solid pure white background, plain white backdrop",
    "sticker_vinyl": "isolated on a solid pure white background, plain white backdrop",
}

# Every product is printed as artwork, never as a photo of a printed product.
NO_MOCKUP = (
    "no product mockup, no garment, no bag, no tote bag, no pouch, no shirt, no hoodie, "
    "no mug, no tumbler, no phone case, no pillow, no blanket, no sticker sheet, "
    "no photo of a printed item"
)

# The space the text is drawn in, per layout. The text is composited onto the
# artwork afterwards, so the artwork has to leave the room for it - otherwise
# the text is drawn over the picture (issue #227).
LAYOUT_RESERVES: dict[str, str] = {
    "text_top": (
        "Leave the top of the canvas clear and empty: plain background only, nothing in that area."
    ),
    "text_top_bottom": (
        "Leave the top and the bottom of the canvas clear and empty: plain background "
        "only, nothing in those areas."
    ),
    "badge": (
        "Draw the artwork as a compact centred emblem with an empty margin all around "
        "it, and leave a clear band at the top and a clear band at the bottom of the "
        "canvas: plain background only in those bands."
    ),
    "banner": (
        "Draw the artwork to fill the whole canvas, and leave a clear horizontal band "
        "across the middle of the canvas: plain background only in that band."
    ),
    "arch": (
        "Draw the artwork as a tall arch shape in the middle of the canvas, and leave "
        "the top of the arch and the bottom of the canvas clear and empty: plain "
        "background only in those areas."
    ),
    "stack_side": (
        "Draw the artwork on the left half of the canvas and leave the whole right "
        "side of the canvas clear and empty: plain background only on the right."
    ),
}

# The image is the graphic that gets printed, not the thing it is printed on.
# A scene that names the product - a trick-or-treat bag, a hoodie - makes the
# model draw the product (issue #195).
DESIGN_ONLY = "print artwork only, the graphic that gets printed, not the object it is printed on"

# What the retry asks for when the artwork came back as a product.
ARTWORK_RETRY_HINT = (
    "Only the flat graphic: no bag, no tote, no pouch, no garment, no product, no object."
)

# The artwork must not carry words: the text of the design is drawn on it
# afterwards, with a font (issue #208).
NO_LETTERING_RETRY_HINT = "No lettering, no text, no words and no numbers in the artwork."

# The product types whose background is printed as-is and is therefore white.
WHITE_BACKGROUND_PRODUCTS = frozenset({"sublimation", "sticker_vinyl"})

DEFAULT_FONT_COLOR = "#000000"


@dataclass
class ImageResult:
    image_bytes: bytes | None
    prompt: str
    error: str | None = None
    raw_bytes: bytes | None = None
    background_removed: bool = False


class PromptBuildError(ValueError):
    """The brief cannot be turned into an image prompt."""


def _scene_for_prompt(brief: DesignBrief) -> str | None:
    """The visual scene description of the brief, or None when it has none.

    The concept is deliberately not used as a fallback. A concept names the
    design idea - "Trick or Treat Bag with Candy Scatter" - and sending it
    to the image model draws the bag instead of the design (issue #195).

    The scene is stripped of any words it asks the model to spell: the image
    model cannot spell, so a quoted punchline comes back misspelled (issue
    #208). The text of the design is drawn on the artwork afterwards.
    """
    scene = strip_lettering((brief.scene_description or "").strip())
    return scene or None


def _feedback_section(brief: DesignBrief) -> str:
    if not brief.regeneration_feedback:
        return ""
    return f" {brief.regeneration_feedback}."


def _is_near_white(colour: str) -> bool:
    """True for a white or near-white hex colour."""
    digits = colour.lstrip("#")
    if len(digits) != 6:
        return False
    try:
        channels = [int(digits[index : index + 2], 16) for index in (0, 2, 4)]
    except ValueError:
        return False
    return min(channels) >= 200


def font_colour(brief: DesignBrief) -> str:
    """The colour the headline and the tagline are drawn in.

    White text is fine on the magenta background of a dtf_apparel brief - that
    background is removed before printing. On the white background of the
    print-through product types it would be invisible, so it is corrected here.
    Briefs written before the backgrounds changed can still carry one.
    """
    colour = brief.font_color or DEFAULT_FONT_COLOR
    if brief.product_type in WHITE_BACKGROUND_PRODUCTS and _is_near_white(colour):
        logger.warning("Brief %r has a near-white font colour on a white background", brief.concept)
        return DEFAULT_FONT_COLOR
    return colour


def build_flux_prompt(brief: DesignBrief) -> str:
    """Build the Flux prompt for a brief, folding the layout fields in.

    The prompt carries the artwork only. The headline and the tagline are
    composited onto the artwork afterwards, because the model cannot spell
    (issue #187) and because a colour in a prompt is not a colour.

    full_bleed       -> the scene only, explicitly no lettering
    text_top         -> the scene, with room for a headline at the top
    text_top_bottom  -> the scene, with room for a headline at the top and a
                        tagline at the bottom
    text_only        -> not an image at all: the design is the text, and it is
                        drawn on a blank canvas without the image model
    """
    product_context = PRODUCT_TYPE_CONTEXT.get(brief.product_type, "print-ready design")
    background = BACKGROUND_INSTRUCTIONS.get(brief.product_type, "plain background")
    feedback = _feedback_section(brief)
    layout_type = resolve_layout(
        brief.layout_type or "full_bleed", brief.headline_text, brief.tagline_text
    )

    if layout_type == "text_only":
        raise PromptBuildError("a text_only brief is drawn without the image model")

    scene = _scene_for_prompt(brief)
    if not scene:
        raise PromptBuildError("the brief has no scene_description")

    if layout_type == "full_bleed" and brief.headline_text:
        logger.warning(
            "Brief has layout_type 'full_bleed' but a headline (%r) - rendering without text",
            brief.headline_text,
        )

    parts: list[str] = []

    parts.append(f"{DESIGN_ONLY}: {scene}.")
    # The look of the design is a preset, not the free text of the brief: free
    # text made every design come out as the same flat vector clip-art
    # (issue #227). A brief that was written before the presets existed keeps
    # its own style.
    look = prompt_for(preset_for(brief.style_preset)) or brief.visual_style
    parts.append(f"{look}, {product_context}, {NO_MOCKUP}.")

    if layout_type == "full_bleed":
        parts.append("Do not include any text or lettering.")
    else:
        parts.append("Print artwork without any lettering: no text, no words, no watermark.")
        reserve = LAYOUT_RESERVES.get(layout_type)
        if reserve:
            parts.append(reserve)

    parts.append("Do not draw anything else.")

    # The background comes last so that it survives a visual style that asks
    # for something else (issue #188).
    parts.append(f"{background}.")

    if feedback:
        parts.append(feedback.strip())

    return " ".join(parts)


def build_negative_prompt(brief: DesignBrief) -> str:
    """What the image model is told not to draw.

    The negative prompt keeps the look of the design and keeps the paper
    clean: a texture that is painted on the background would survive the DTF
    cutout as a grey rectangle and would wreck the despill (issue #227).
    """
    return negative_for(preset_for(brief.style_preset))


def generate_image(
    brief: DesignBrief,
    *,
    comfyui_client: ComfyUIClient,
    seed: int | None = None,
    extra_prompt: str = "",
) -> ImageResult:
    """Generate an image for a brief on the DGX.

    ``dtf_apparel`` briefs get their background removed with BiRefNet on the DGX;
    the original render is kept in ``raw_bytes``. Failures never raise - they are
    reported through ``ImageResult.error``.
    """
    try:
        prompt = build_flux_prompt(brief)
    except PromptBuildError as error:
        logger.warning("Cannot build an image prompt for the brief: %s", error)
        return ImageResult(image_bytes=None, prompt="", error=str(error))

    if extra_prompt:
        prompt = f"{prompt} {extra_prompt}"
    negative_prompt = build_negative_prompt(brief)
    try:
        result = comfyui_client.generate_image(
            prompt=prompt, seed=seed, negative_prompt=negative_prompt
        )
    except ComfyUIError as e:
        logger.warning("ComfyUI image generation failed: %s", e)
        return ImageResult(image_bytes=None, prompt=prompt, error=str(e))

    if result.image_bytes is None:
        error = result.error or "ComfyUI returned no image"
        logger.warning("ComfyUI image generation failed: %s", error)
        return ImageResult(image_bytes=None, prompt=prompt, error=error)

    raw_bytes = result.image_bytes
    image_bytes = raw_bytes
    background_removed = False

    if brief.product_type == "dtf_apparel":
        try:
            transparent = comfyui_client.remove_background(raw_bytes)
        except ComfyUIError as e:
            logger.warning("Background removal failed, returning raw image: %s", e)
        else:
            if transparent.image_bytes is not None:
                image_bytes = transparent.image_bytes
                background_removed = True
            else:
                logger.warning(
                    "Background removal returned no image, returning raw image: %s",
                    transparent.error,
                )

    return ImageResult(
        image_bytes=image_bytes,
        prompt=prompt,
        raw_bytes=raw_bytes,
        background_removed=background_removed,
    )
