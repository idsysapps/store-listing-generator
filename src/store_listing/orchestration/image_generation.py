"""Bedrock-based image generation from design briefs.

Temporary solution using AWS Bedrock Stability SD 3.5 until DGX hardware
is available for self-hosted ComfyUI + Flux Schnell. See issue #111.
"""

from __future__ import annotations

import base64
import json
import logging
import os
from dataclasses import dataclass
from typing import Any

from store_listing.orchestration.design_briefs import DesignBrief

logger = logging.getLogger(__name__)

BEDROCK_MODEL_ID: str = os.environ.get("BEDROCK_IMAGE_MODEL_ID", "stability.sd3-large-v1:0")

PRODUCT_TYPE_CONTEXT: dict[str, str] = {
    "dtf_apparel": "design for DTF print on apparel, t-shirt graphic, clean edges, print-ready",
    "sublimation": "full-color sublimation print design, seamless edges, vibrant colors, print-ready",
    "sticker_vinyl": "sticker or vinyl decal design, clean cut lines, bold outlines, print-ready",
}


@dataclass
class ImageResult:
    image_bytes: bytes | None
    prompt: str
    error: str | None = None


def build_prompt_from_brief(brief: DesignBrief) -> str:
    product_context = PRODUCT_TYPE_CONTEXT.get(brief.product_type, "print-ready design")
    return (
        f"{brief.concept}, {brief.visual_style}, "
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
                    "width": 1024,
                    "height": 1024,
                }
            ),
        )
        result = json.loads(response["body"].read())
        image_b64 = result["images"][0]
        image_bytes = base64.b64decode(image_b64)
    except Exception as e:  # noqa: BLE001
        logger.warning("Bedrock image generation failed: %s", e)
        return ImageResult(image_bytes=None, prompt=prompt, error=str(e))

    return ImageResult(image_bytes=image_bytes, prompt=prompt)
