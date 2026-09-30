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
        "centered raster illustration on a plain white background, "
        "compact composition with generous negative space on all sides, "
        "the design should look like a decal or patch not edge-to-edge, "
        "suitable for DTF transfer print on apparel"
    ),
    "sublimation": "full-color sublimation print design, seamless edges, vibrant colors, print-ready",
    "sticker_vinyl": "sticker or vinyl decal design, clean cut lines, bold outlines, print-ready",
}


@dataclass
class ImageResult:
    image_bytes: bytes | None
    prompt: str
    error: str | None = None
    raw_bytes: bytes | None = None


def _layout_space_instruction(layout_type: str) -> str:
    if layout_type == "text_top":
        return "leave the top 15% of the canvas empty as plain background for text overlay, "
    if layout_type == "text_top_bottom":
        return (
            "leave the top 15% and bottom 12% of the canvas empty as plain background "
            "for text overlay, "
        )
    return ""


def build_prompt_from_brief(brief: DesignBrief) -> str:
    product_context = PRODUCT_TYPE_CONTEXT.get(brief.product_type, "print-ready design")
    space_instruction = _layout_space_instruction(brief.layout_type)
    if brief.product_type == "dtf_apparel":
        return (
            f"{brief.concept}. Style: {brief.visual_style}. "
            f"{space_instruction}"
            f"{product_context}, high quality, professional illustration"
        )
    return (
        f"{brief.concept}, {brief.visual_style}, "
        f"{space_instruction}"
        f"{product_context}, high quality, professional illustration"
    )


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
