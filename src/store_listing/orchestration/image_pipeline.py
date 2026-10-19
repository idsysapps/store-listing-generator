"""Pipeline: generate images for design briefs, store in S3, update DB.

Images are rendered on the DGX with ComfyUI + Flux Schnell. When a brief carries
text, the rendered text is read back out with a vision model and compared with
the requested headline and tagline. The gate fails closed: an image whose text
cannot be verified is never uploaded.
"""

from __future__ import annotations

import logging
import os
import random
from typing import Any

from store_listing.orchestration.comfyui_client import ComfyUIClient
from store_listing.orchestration.design_briefs import DesignBrief
from store_listing.orchestration.image_generation import generate_image
from store_listing.orchestration.image_storage import ImageStorageClient
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
    """Nudge for the next attempt, carrying the text the model got wrong."""
    if brief.headline_text:
        return f'The text must read exactly "{brief.headline_text}". Render no other words.'
    if brief.tagline_text:
        return f'The text must read exactly "{brief.tagline_text}". Render no other words.'
    return "Render the requested text exactly, with no other words."


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
    needs_text_check = bool(brief.headline_text or brief.tagline_text)
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

        if text_validator is None or not needs_text_check:
            break

        verdict = text_validator.validate(
            result.raw_bytes or result.image_bytes,
            headline=brief.headline_text,
            tagline=brief.tagline_text,
        )
        if verdict.matched:
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

    raw_bytes = result.raw_bytes if result.raw_bytes else result.image_bytes
    transparent_bytes: bytes | None = None
    if is_dtf and result.background_removed:
        transparent_bytes = result.image_bytes

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
