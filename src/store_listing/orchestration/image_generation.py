"""Image generation from design briefs via ComfyUI + Flux Schnell on the DGX.

Leonardo.ai, the Ideogram text-routing workaround and the AWS Bedrock image
calls were removed in #111: the self-hosted DGX is the only generation path.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from store_listing.orchestration.comfyui_client import ComfyUIClient, ComfyUIError
from store_listing.orchestration.design_briefs import DesignBrief

logger = logging.getLogger(__name__)

PRODUCT_TYPE_CONTEXT: dict[str, str] = {
    "dtf_apparel": (
        "sticker illustration on plain white background, bold outlines, "
        "flat colors, compact centered composition, DTF transfer print"
    ),
    "sublimation": "full-color sublimation print, seamless edges, vibrant colors",
    "sticker_vinyl": "sticker decal, clean cut lines, bold outlines",
}

DEFAULT_FONT_COLOR = "#000000"


@dataclass
class ImageResult:
    image_bytes: bytes | None
    prompt: str
    error: str | None = None
    raw_bytes: bytes | None = None
    background_removed: bool = False


def _split_concept(concept: str) -> tuple[str, str | None]:
    """Split concept into scene description and short text to render.

    Handles patterns like 'Concept Name - Subtitle' or 'Title: Description'.
    Returns (scene_description, short_text_or_none).
    """
    for sep in (" - ", ": ", " — "):
        if sep in concept:
            parts = concept.split(sep, 1)
            shorter = min(parts, key=len)
            longer = max(parts, key=len)
            if len(shorter.split()) <= 6:
                return longer, shorter
            return concept, None
    if len(concept.split()) <= 5:
        return concept, concept
    return concept, None


def _scene_for_prompt(brief: DesignBrief) -> str:
    """Return the visual scene description, preferring the dedicated field."""
    if brief.scene_description:
        return brief.scene_description
    if brief.headline_text:
        return brief.concept
    scene, _ = _split_concept(brief.concept)
    return scene


def _feedback_section(brief: DesignBrief) -> str:
    if not brief.regeneration_feedback:
        return ""
    return f" {brief.regeneration_feedback}."


def build_flux_prompt(brief: DesignBrief) -> str:
    """Build the Flux prompt for a brief, folding the layout fields in.

    full_bleed       -> the scene only, explicitly no lettering
    text_top         -> the headline at the top, then the scene
    text_top_bottom  -> the headline at the top, the scene, the tagline at the bottom
    """
    product_context = PRODUCT_TYPE_CONTEXT.get(brief.product_type, "print-ready design")
    scene = _scene_for_prompt(brief)
    feedback = _feedback_section(brief)
    font_color = brief.font_color or DEFAULT_FONT_COLOR
    layout_type = brief.layout_type or "full_bleed"

    if layout_type == "full_bleed" and brief.headline_text:
        logger.warning(
            "Brief has layout_type 'full_bleed' but a headline (%r) - rendering without text",
            brief.headline_text,
        )

    parts: list[str] = []

    if layout_type != "full_bleed" and brief.headline_text:
        parts.append(f'Bold text reading "{brief.headline_text}" in {font_color} at the top.')

    parts.append(f"{scene}, {brief.visual_style}, {product_context}.")

    if layout_type == "full_bleed":
        parts.append("Do not include any text or lettering.")
    elif layout_type == "text_top_bottom" and brief.tagline_text:
        parts.append(f'Text reading "{brief.tagline_text}" in {font_color} at the bottom.')

    parts.append("Do not draw anything else.")

    if feedback:
        parts.append(feedback.strip())

    return " ".join(parts)


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
    prompt = build_flux_prompt(brief)
    if extra_prompt:
        prompt = f"{prompt} {extra_prompt}"

    try:
        result = comfyui_client.generate_image(prompt=prompt, seed=seed)
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
