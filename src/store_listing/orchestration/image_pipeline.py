"""Pipeline: generate images for design briefs, store in S3, update DB."""

from __future__ import annotations

import logging
from typing import Any

from store_listing.orchestration.design_briefs import DesignBrief
from store_listing.orchestration.image_generation import generate_image, generate_image_leonardo
from store_listing.orchestration.image_storage import ImageStorageClient
from store_listing.orchestration.leonardo_client import LeonardoClient

logger = logging.getLogger(__name__)


def generate_images_for_briefs(
    *,
    db_client: Any,
    bedrock_client: Any,
    s3_client: Any,
    bucket: str = "store-listing-designs",
    endpoint_url: str | None = None,
    limit: int = 20,
    leonardo_client: LeonardoClient | None = None,
) -> dict[str, Any]:
    briefs = db_client.list_briefs_without_images(limit)
    if not briefs:
        return {"status": "success", "images_generated": 0, "errors": 0}

    storage = ImageStorageClient(s3_client=s3_client, bucket=bucket, endpoint_url=endpoint_url)

    generated = 0
    errors = 0

    for brief_row in briefs:
        success = _generate_single_brief(
            brief_row,
            storage=storage,
            bedrock_client=bedrock_client,
            leonardo_client=leonardo_client,
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
    bedrock_client: Any,
    leonardo_client: LeonardoClient | None,
    db_client: Any,
) -> bool:
    layout_type = brief_row.get("layout_type", "full_bleed")
    brief = DesignBrief(
        concept=brief_row["concept"],
        product_type=brief_row["product_type"],
        specific_products=[],
        audience=brief_row.get("audience", ""),
        visual_style=brief_row.get("visual_style", ""),
        confidence=0,
        reasoning="",
        source_seed_ids=[],
        layout_type=layout_type,
        headline_text=brief_row.get("headline_text"),
        tagline_text=brief_row.get("tagline_text"),
        font_color=brief_row.get("font_color"),
        regeneration_feedback=brief_row.get("regeneration_feedback"),
        scene_description=brief_row.get("scene_description"),
    )

    if leonardo_client is not None:
        result = generate_image_leonardo(
            brief, leonardo_client=leonardo_client, bedrock_client=bedrock_client
        )
    else:
        result = generate_image(brief, bedrock_client=bedrock_client)

    if result.image_bytes is None:
        logger.warning(
            "Image generation failed for brief %d: %s",
            brief_row["id"],
            result.error,
        )
        return False

    is_dtf = brief_row["product_type"] == "dtf_apparel"
    raw_bytes = result.raw_bytes if result.raw_bytes else result.image_bytes
    transparent_bytes: bytes | None = None
    if is_dtf and result.image_bytes:
        transparent_bytes = result.image_bytes

    keys = storage.upload_design(
        raw_bytes=raw_bytes,
        transparent_bytes=transparent_bytes,
        brief_id=brief_row["id"],
        product_type=brief_row["product_type"],
    )

    db_client.update_brief_image_keys(
        brief_id=brief_row["id"],
        image_key_raw=keys["raw"],
        image_key_transparent=keys.get("transparent"),
    )
    return True


def generate_image_for_brief(
    *,
    brief_id: int,
    db_client: Any,
    bedrock_client: Any,
    s3_client: Any,
    bucket: str = "store-listing-designs",
    endpoint_url: str | None = None,
    leonardo_client: LeonardoClient | None = None,
) -> dict[str, Any]:
    brief_row = db_client.get_brief_by_id(brief_id)
    if brief_row is None:
        return {"status": "error", "error": f"Brief {brief_id} not found"}

    storage = ImageStorageClient(s3_client=s3_client, bucket=bucket, endpoint_url=endpoint_url)
    success = _generate_single_brief(
        brief_row,
        storage=storage,
        bedrock_client=bedrock_client,
        leonardo_client=leonardo_client,
        db_client=db_client,
    )
    if success:
        return {"status": "success", "brief_id": brief_id}
    return {"status": "error", "brief_id": brief_id, "error": "Image generation failed"}
