"""Pipeline: generate images for design briefs, store in S3, update DB.

Images are rendered on the DGX with ComfyUI + Flux Schnell. When a brief carries
text, the rendered text is read back with a vision model and compared with the
requested headline and tagline. The gate fails closed for the headline: an image
whose message cannot be verified is never uploaded. The tagline is secondary and
diffusion drops it often, so a missing tagline is logged and the image is kept.
"""

from __future__ import annotations

import logging
import os
import random
from typing import Any

from store_listing.orchestration.comfyui_client import ComfyUIClient
from store_listing.orchestration.design_briefs import DesignBrief
from store_listing.orchestration.image_generation import font_colour, generate_image
from store_listing.orchestration.image_storage import ImageStorageClient
from store_listing.orchestration.text_compositor import LayoutSpec, composite_text
from store_listing.orchestration.text_validation import TextValidator

logger = logging.getLogger(__name__)

DEFAULT_MAX_TEXT_RETRIES = 2


def _max_text_retries() -> int:
    """Retries allowed after a text mismatch (so 2 means up to 3 generations)."""
    raw = os.environ.get("MAX_TEXT_REGENERATION_ATTEMPTS", str(DEFAULT_MAX_TEXT_RETRIES))
    try:
        return max(0, int(raw))
    except ValueError:
        logger.warning("Invalid MAX_TEXT_REGENERATION_ATTEMPTS=%r, using the default", raw)
        return DEFAULT_MAX_TEXT_RETRIES


def _brief_from_row(brief_row: dict[str, Any]) -> DesignBrief:
    return DesignBrief(
        concept=brief_row["concept"],
        product_type=brief_row["product_type"],
        specific_products=[],
        audience=brief_row.get("audience", ""),
        visual_style=brief_row.get("visual_style", ""),
        confidence=0,
        reasoning="",
        source_seed_ids=[],
        layout_type=brief_row.get("layout_type", "full_bleed"),
        headline_text=brief_row.get("headline_text"),
        tagline_text=brief_row.get("tagline_text"),
        font_color=brief_row.get("font_color"),
        regeneration_feedback=brief_row.get("regeneration_feedback"),
        scene_description=brief_row.get("scene_description"),
    )


def _retry_hint(brief: DesignBrief) -> str:
    """Nudge for the next attempt: keep the area the text goes in clear.

    The text is composited onto the artwork, so a retry cannot ask the model
    to spell it - it can only ask for a clearer area to put it in.
    """
    layout_type = brief.layout_type or "full_bleed"
    if layout_type == "text_top_bottom":
        return "Leave the top and the bottom of the canvas clear and empty."
    if layout_type == "text_top":
        return "Leave the top of the canvas clear and empty."
    return "Render the requested scene more clearly."


def generate_images_for_briefs(
    *,
    db_client: Any,
    comfyui_client: ComfyUIClient,
    s3_client: Any,
    bucket: str = "store-listing-designs",
    endpoint_url: str | None = None,
    limit: int = 20,
    text_validator: TextValidator | None = None,
) -> dict[str, Any]:
    briefs = db_client.list_briefs_without_images(limit)
    if not briefs:
        return {"status": "success", "images_generated": 0, "errors": 0}

    storage = ImageStorageClient(s3_client=s3_client, bucket=bucket, endpoint_url=endpoint_url)

    generated = 0
    errors = 0

    for brief_row in briefs:
        success, _ = _generate_single_brief(
            brief_row,
            storage=storage,
            comfyui_client=comfyui_client,
            text_validator=text_validator,
            db_client=db_client,
        )
        if success:
            generated += 1
        else:
            errors += 1

    logger.info("Image pipeline: generated=%d, errors=%d", generated, errors)
    return {"status": "success", "images_generated": generated, "errors": errors}


def _generate_single_brief(
    brief_row: dict[str, Any],
    *,
    storage: ImageStorageClient,
    comfyui_client: ComfyUIClient,
    text_validator: TextValidator | None,
    db_client: Any,
) -> tuple[bool, str | None]:
    brief = _brief_from_row(brief_row)
    brief_id = brief_row["id"]
    is_dtf = brief_row["product_type"] == "dtf_apparel"
    layout = LayoutSpec(
        layout_type=brief.layout_type or "full_bleed",
        headline_text=brief.headline_text,
        tagline_text=brief.tagline_text,
        font_color=font_colour(brief),
    )
    needs_text_check = layout.needs_text
    max_retries = _max_text_retries()

    base_seed = random.randint(0, 2**31 - 1)
    retry_hint = ""
    attempts = 0

    while True:
        attempts += 1
        result = generate_image(
            brief,
            comfyui_client=comfyui_client,
            seed=base_seed + attempts - 1,
            extra_prompt=retry_hint,
        )

        if result.image_bytes is None:
            logger.warning(
                "Image generation failed for brief %d: %s",
                brief_id,
                result.error,
            )
            return False, result.error or "image generation failed"

        # The text is drawn onto the artwork, not asked of the model: it cannot
        # spell, and a colour in a prompt is not a colour (issue #187).
        raw_bytes = composite_text(result.raw_bytes or result.image_bytes, layout)

        if text_validator is None or not needs_text_check:
            break

        verdict = text_validator.validate(
            raw_bytes,
            headline=layout.headline_text,
            tagline=layout.tagline_text,
        )
        if verdict.matched:
            if verdict.missing or verdict.unverifiable or verdict.duplicated:
                logger.warning(
                    "Text validation passed with gaps for brief %d: missing %s, unverifiable %s, "
                    "duplicated %s, observed %r",
                    brief_id,
                    verdict.missing,
                    verdict.unverifiable,
                    verdict.duplicated,
                    verdict.observed,
                )
            break

        if attempts > max_retries:
            logger.error(
                "Text validation failed after %d attempt(s) for brief %d: expected %s, observed %r",
                attempts,
                brief_id,
                verdict.expected,
                verdict.observed,
            )
            return False, (
                f"text validation failed after {attempts} attempt(s): "
                f"expected {verdict.expected}, observed {verdict.observed!r}"
            )

        retry_hint = _retry_hint(brief)

    transparent_bytes: bytes | None = None
    if is_dtf and result.background_removed and result.image_bytes:
        transparent_bytes = composite_text(result.image_bytes, layout)

    keys = storage.upload_design(
        raw_bytes=raw_bytes,
        transparent_bytes=transparent_bytes,
        brief_id=brief_id,
        product_type=brief_row["product_type"],
    )

    db_client.update_brief_image_keys(
        brief_id=brief_id,
        image_key_raw=keys["raw"],
        image_key_transparent=keys.get("transparent"),
    )
    return True, None


def generate_image_for_brief(
    *,
    brief_id: int,
    db_client: Any,
    comfyui_client: ComfyUIClient,
    s3_client: Any,
    bucket: str = "store-listing-designs",
    endpoint_url: str | None = None,
    text_validator: TextValidator | None = None,
) -> dict[str, Any]:
    brief_row = db_client.get_brief_by_id(brief_id)
    if brief_row is None:
        return {"status": "error", "error": f"Brief {brief_id} not found"}

    storage = ImageStorageClient(s3_client=s3_client, bucket=bucket, endpoint_url=endpoint_url)
    success, error = _generate_single_brief(
        brief_row,
        storage=storage,
        comfyui_client=comfyui_client,
        text_validator=text_validator,
        db_client=db_client,
    )
    if success:
        return {"status": "success", "brief_id": brief_id}
    return {"status": "error", "brief_id": brief_id, "error": error or "Image generation failed"}
