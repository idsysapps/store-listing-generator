"""Image generation from design briefs.

Supports Leonardo.ai (Flux) and AWS Bedrock (Stability) backends.
Leonardo is the default for generation; Bedrock is kept for background
removal. Both are temporary until DGX hardware arrives for self-hosted
ComfyUI + Flux Schnell. See issue #111.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
from dataclasses import dataclass
from typing import Any

from store_listing.orchestration.design_briefs import DesignBrief
from store_listing.orchestration.leonardo_client import LeonardoClient, LeonardoGenerationResult

logger = logging.getLogger(__name__)

BEDROCK_MODEL_ID: str = os.environ.get("BEDROCK_IMAGE_MODEL_ID", "stability.sd3-5-large-v1:0")
BEDROCK_REGION: str = os.environ.get("AWS_BEDROCK_REGION", "us-west-2")

PRODUCT_TYPE_CONTEXT: dict[str, str] = {
    "dtf_apparel": (
        "sticker illustration on plain white background, bold outlines, "
        "flat colors, compact centered composition, DTF transfer print"
    ),
    "sublimation": "full-color sublimation print, seamless edges, vibrant colors",
    "sticker_vinyl": "sticker decal, clean cut lines, bold outlines",
}


@dataclass
class ImageResult:
    image_bytes: bytes | None
    prompt: str
    error: str | None = None
    raw_bytes: bytes | None = None


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


def build_prompt_from_brief(brief: DesignBrief) -> str:
    product_context = PRODUCT_TYPE_CONTEXT.get(brief.product_type, "print-ready design")
    scene = _scene_for_prompt(brief)
    feedback = _feedback_section(brief)

    parts: list[str] = []
    parts.append(f"{scene}, {brief.visual_style}, {product_context}.")
    parts.append("Do not include any text or lettering.")
    parts.append("Do not draw anything else.")
    if feedback:
        parts.append(feedback.strip())
    return " ".join(parts)


def generate_image(
    brief: DesignBrief,
    *,
    bedrock_client: Any,
    model_id: str | None = None,
) -> ImageResult:
    prompt = build_prompt_from_brief(brief)
    model = model_id or BEDROCK_MODEL_ID

    try:
        response = bedrock_client.invoke_model(
            modelId=model,
            body=json.dumps(
                {
                    "prompt": prompt,
                    "mode": "text-to-image",
                    "aspect_ratio": "1:1",
                    "output_format": "png",
                }
            ),
        )
        result = json.loads(response["body"].read())
        image_b64 = result["images"][0]
        image_bytes = base64.b64decode(image_b64)
    except Exception as e:  # noqa: BLE001
        logger.warning("Bedrock image generation failed: %s", e)
        return ImageResult(image_bytes=None, prompt=prompt, error=str(e))

    raw_bytes = image_bytes
    if brief.product_type == "dtf_apparel":
        try:
            image_bytes = remove_background(image_bytes, bedrock_client=bedrock_client)
        except Exception as e:  # noqa: BLE001
            logger.warning("Background removal failed, returning raw image: %s", e)

    return ImageResult(image_bytes=image_bytes, prompt=prompt, raw_bytes=raw_bytes)


def generate_image_leonardo(
    brief: DesignBrief,
    *,
    leonardo_client: LeonardoClient,
    bedrock_client: Any | None = None,
) -> ImageResult:
    prompt = build_prompt_from_brief(brief)

    try:
        leo_result: LeonardoGenerationResult = asyncio.get_event_loop().run_until_complete(
            leonardo_client.generate(prompt=prompt)
        )
    except RuntimeError:
        loop = asyncio.new_event_loop()
        try:
            leo_result = loop.run_until_complete(leonardo_client.generate(prompt=prompt))
        finally:
            loop.close()

    if leo_result.image_bytes is None:
        logger.warning("Leonardo image generation failed: %s", leo_result.error)
        return ImageResult(image_bytes=None, prompt=prompt, error=leo_result.error)

    raw_bytes = leo_result.image_bytes
    image_bytes = raw_bytes

    if brief.product_type == "dtf_apparel" and bedrock_client is not None:
        try:
            image_bytes = remove_background(raw_bytes, bedrock_client=bedrock_client)
        except Exception as e:  # noqa: BLE001
            logger.warning("Background removal failed, returning raw image: %s", e)

    return ImageResult(image_bytes=image_bytes, prompt=prompt, raw_bytes=raw_bytes)


REMOVE_BG_MODEL_ID: str = "us.stability.stable-image-remove-background-v1:0"


def remove_background(image_bytes: bytes, *, bedrock_client: Any) -> bytes:
    image_b64 = base64.b64encode(image_bytes).decode()
    response = bedrock_client.invoke_model(
        modelId=REMOVE_BG_MODEL_ID,
        body=json.dumps({"image": image_b64}),
    )
    result = json.loads(response["body"].read())
    return base64.b64decode(result["images"][0])
