"""Pipeline: generate images for design briefs, store in S3, update DB.

Images are rendered on the DGX with ComfyUI + Flux Schnell. When a brief carries
text, the rendered text is read back with a vision model and compared with the
requested headline and tagline. The gate fails closed for the headline: an image
whose message cannot be verified is never uploaded. The tagline is secondary and
diffusion drops it often, so a missing tagline is logged and the image is kept.

The artwork itself is checked too: an image that shows the product instead of
the design - a bag, a hoodie, a mug - is never uploaded (issue #195). A brief
that has no usable artwork is rendered as a typography-only design when it has
text, and left pending when it has none (issue #196).
"""

from __future__ import annotations

import logging
import os
import random
from typing import Any

from store_listing.orchestration.art_direction import family_for, preset_for, tagline_family_for
from store_listing.orchestration.artwork_validation import (
    ArtworkTextValidator,
    ArtworkValidator,
)
from store_listing.orchestration.comfyui_client import ComfyUIClient
from store_listing.orchestration.design_briefs import DesignBrief
from store_listing.orchestration.design_templates import composition_for, resolve_template
from store_listing.orchestration.headline_voice import DEFAULT_VOICE
from store_listing.orchestration.image_generation import (
    ARTWORK_RETRY_HINT,
    NO_LETTERING_RETRY_HINT,
    ImageResult,
    font_colour,
    generate_image,
)
from store_listing.orchestration.image_storage import ImageStorageClient
from store_listing.orchestration.scene_text import strip_lettering
from store_listing.orchestration.text_compositor import (
    DTF_KEY_COLOUR,
    LayoutSpec,
    blank_canvas,
    composite_text,
)
from store_listing.orchestration.text_integration import text_plan_from_value
from store_listing.orchestration.text_validation import StrictTextValidator, TextValidator
from store_listing.orchestration.textures import apply_texture, despeckle_image
from store_listing.orchestration.typographic_hero import (
    TypographicHeroGenerator,
    hero_candidate_count,
    hero_copy_from_brief,
    is_typographic_hero_brief,
    typographic_hero_enabled,
)

logger = logging.getLogger(__name__)

DEFAULT_MAX_TEXT_RETRIES = 2
DEFAULT_MAX_ARTWORK_RETRIES = 1
DEFAULT_MAX_LETTERING_RETRIES = 1


def _max_text_retries() -> int:
    """Retries allowed after a text mismatch (so 2 means up to 3 generations)."""
    raw = os.environ.get("MAX_TEXT_REGENERATION_ATTEMPTS", str(DEFAULT_MAX_TEXT_RETRIES))
    try:
        return max(0, int(raw))
    except ValueError:
        logger.warning("Invalid MAX_TEXT_REGENERATION_ATTEMPTS=%r, using the default", raw)
        return DEFAULT_MAX_TEXT_RETRIES


def _max_artwork_retries() -> int:
    """Retries allowed after the artwork came back as a product."""
    raw = os.environ.get("MAX_ARTWORK_REGENERATION_ATTEMPTS", str(DEFAULT_MAX_ARTWORK_RETRIES))
    try:
        return max(0, int(raw))
    except ValueError:
        logger.warning("Invalid MAX_ARTWORK_REGENERATION_ATTEMPTS=%r, using the default", raw)
        return DEFAULT_MAX_ARTWORK_RETRIES


def _max_lettering_retries() -> int:
    """Retries allowed after the artwork came back with lettering in it."""
    raw = os.environ.get("MAX_LETTERING_REGENERATION_ATTEMPTS", str(DEFAULT_MAX_LETTERING_RETRIES))
    try:
        return max(0, int(raw))
    except ValueError:
        logger.warning("Invalid MAX_LETTERING_REGENERATION_ATTEMPTS=%r, using the default", raw)
        return DEFAULT_MAX_LETTERING_RETRIES


def _brief_from_row(brief_row: dict[str, Any]) -> DesignBrief:
    # The scene is stripped of the words it asks the image model to spell: a
    # scene that is only words leaves nothing to draw, and the design falls
    # back to typography (issue #208).
    scene = strip_lettering((brief_row.get("scene_description") or "").strip())
    voice = brief_row.get("voice") or DEFAULT_VOICE
    # The type follows the look, so a design that is rendered in another look
    # is drawn in the type of that look (issue #228).
    preset = preset_for(brief_row.get("style_preset"))
    font_family = family_for(preset, voice) if preset else brief_row.get("font_family")
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
        scene_description=scene or None,
        voice=voice,
        font_family=font_family,
        style_preset=brief_row.get("style_preset"),
        text_plan=text_plan_from_value(brief_row.get("text_plan")),
    )


def _textured(
    result: ImageResult,
    brief: DesignBrief,
    is_dtf: bool,
    *,
    keyed: bool = False,
) -> bytes:
    """The artwork of a render, with the texture of its look on it.

    The texture is applied after the keying, and never on the background: a
    texture on the background of a DTF render would survive the cutout as a
    grey rectangle (issue #227).
    """
    source = result.image_bytes if keyed else (result.raw_bytes or result.image_bytes)
    if not source:
        return b""

    preset = preset_for(brief.style_preset)
    if preset and preset.despeckle:
        source = despeckle_image(source)
    texture = preset.texture if preset else None
    if not texture:
        return source

    if keyed:
        return apply_texture(source, texture)
    alpha_mask = result.image_bytes if is_dtf and result.background_removed else None
    return apply_texture(
        source,
        texture,
        key_colour=DTF_KEY_COLOUR if is_dtf else None,
        alpha_mask_bytes=alpha_mask,
    )


def _retry_hint(brief: DesignBrief) -> str:
    """Nudge for the next attempt: keep the area the text goes in clear.

    The text is composited onto the artwork, so a retry cannot ask the model
    to spell it - it can only ask for a clearer area to put it in.
    """
    template_id = resolve_template(brief.layout_type, brief.headline_text, brief.tagline_text)
    if template_id in ("full_bleed", "text_only"):
        return "Render the requested scene more clearly."
    return composition_for(template_id, brief.concept)


def generate_images_for_briefs(
    *,
    db_client: Any,
    comfyui_client: ComfyUIClient,
    s3_client: Any,
    bucket: str = "store-listing-designs",
    endpoint_url: str | None = None,
    limit: int = 20,
    text_validator: TextValidator | None = None,
    artwork_validator: ArtworkValidator | None = None,
    artwork_text_validator: ArtworkTextValidator | None = None,
    hero_validator: StrictTextValidator | None = None,
    product_types: list[str] | None = None,
) -> dict[str, Any]:
    # A store that focuses on one production method does not spend GPU time
    # on the briefs of the others (issue #215).
    briefs = db_client.list_briefs_without_images(limit, product_types)
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
            artwork_validator=artwork_validator,
            artwork_text_validator=artwork_text_validator,
            hero_validator=hero_validator,
            db_client=db_client,
        )
        if success:
            generated += 1
        else:
            errors += 1

    logger.info("Image pipeline: generated=%d, errors=%d", generated, errors)
    return {"status": "success", "images_generated": generated, "errors": errors}


def _upload_text_only(
    brief_row: dict[str, Any],
    layout: LayoutSpec,
    *,
    storage: ImageStorageClient,
    db_client: Any,
    text_validator: TextValidator | None,
) -> tuple[bool, str | None]:
    """Render a typography-only design and upload it.

    There is no artwork: the design is the text, drawn on a blank canvas. The
    DTF print file is the same drawing on a transparent canvas.
    """
    brief_id = brief_row["id"]
    product_type = brief_row["product_type"]

    if not layout.needs_text:
        return False, "a typography-only design needs a headline or a tagline"

    raw_bytes = composite_text(blank_canvas(product_type), layout)
    transparent_bytes: bytes | None = None
    if product_type == "dtf_apparel":
        transparent_bytes = composite_text(
            blank_canvas(product_type, transparent=True), layout, transparent=True
        )

    if text_validator is not None:
        verdict = text_validator.validate(
            raw_bytes,
            headline=layout.headline_text,
            tagline=layout.tagline_text,
        )
        if not verdict.matched:
            return False, (
                f"text validation failed: expected {verdict.expected}, "
                f"observed {verdict.observed!r}"
            )

    keys = storage.upload_design(
        raw_bytes=raw_bytes,
        transparent_bytes=transparent_bytes,
        brief_id=brief_id,
        product_type=product_type,
    )
    db_client.update_brief_image_keys(
        brief_id=brief_id,
        image_key_raw=keys["raw"],
        image_key_transparent=keys.get("transparent"),
    )
    return True, None


def _generate_typographic_hero_brief(
    brief_row: dict[str, Any],
    brief: DesignBrief,
    *,
    storage: ImageStorageClient,
    comfyui_client: ComfyUIClient,
    validator: StrictTextValidator,
    db_client: Any,
) -> tuple[bool, str | None]:
    """Render a stacked emblem with Qwen-Image and upload it.

    Qwen draws the kicker and the hero word; there is no compositor pass. The
    descriptor is not part of this image: it is added later, if at all. The DTF
    print file is keyed from the white sticker background.
    """
    brief_id = brief_row["id"]
    is_dtf = brief_row["product_type"] == "dtf_apparel"

    try:
        copy = hero_copy_from_brief(brief)
    except ValueError as error:
        return False, str(error)

    generator = TypographicHeroGenerator(
        comfyui_client=comfyui_client,
        validator=validator,
        candidates=hero_candidate_count(),
    )
    result = generator.generate(copy, seed=random.randint(0, 2**31 - 1))
    if result.image_bytes is None:
        logger.warning(
            "Typographic hero generation failed for brief %d: %s", brief_id, result.error
        )
        return False, result.error or "typographic hero generation failed"

    raw_bytes = result.image_bytes
    transparent_bytes: bytes | None = None
    if is_dtf:
        keyed = comfyui_client.remove_background(raw_bytes)
        if keyed.image_bytes is None:
            logger.warning("Background removal failed for hero brief %d: %s", brief_id, keyed.error)
        else:
            transparent_bytes = keyed.image_bytes

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


def _generate_single_brief(
    brief_row: dict[str, Any],
    *,
    storage: ImageStorageClient,
    comfyui_client: ComfyUIClient,
    text_validator: TextValidator | None,
    db_client: Any,
    artwork_validator: ArtworkValidator | None = None,
    artwork_text_validator: ArtworkTextValidator | None = None,
    hero_validator: StrictTextValidator | None = None,
) -> tuple[bool, str | None]:
    brief = _brief_from_row(brief_row)
    brief_id = brief_row["id"]
    is_dtf = brief_row["product_type"] == "dtf_apparel"

    # The typographic hero path is the only path where the image model draws
    # the words. It is opt-in: a brief has to be marked as a hero layout, or
    # named in a trial batch, and the feature flag has to be on (issue #254).
    if (
        typographic_hero_enabled()
        and hero_validator is not None
        and is_typographic_hero_brief(brief_row, brief)
    ):
        return _generate_typographic_hero_brief(
            brief_row,
            brief,
            storage=storage,
            comfyui_client=comfyui_client,
            validator=hero_validator,
            db_client=db_client,
        )

    preset = preset_for(brief.style_preset)
    layout = LayoutSpec(
        layout_type=resolve_template(
            brief.layout_type or "full_bleed", brief.headline_text, brief.tagline_text
        ),
        headline_text=brief.headline_text,
        tagline_text=brief.tagline_text,
        font_color=font_colour(brief),
        font_family=brief.font_family,
        tagline_font_family=tagline_family_for(preset, brief.voice) if preset else None,
        voice=brief.voice,
        text_plan=brief.text_plan,
    )
    needs_text_check = layout.needs_text
    max_retries = _max_text_retries()
    max_artwork_retries = _max_artwork_retries()
    max_lettering_retries = _max_lettering_retries()

    # A typography-only design is the text and nothing else: the image model
    # is never asked for an illustration (issue #196).
    if layout.layout_type == "text_only":
        return _upload_text_only(
            brief_row,
            layout,
            storage=storage,
            db_client=db_client,
            text_validator=text_validator,
        )

    # The concept is never used as the visual prompt: it names the design
    # idea, and a concept that names the product draws the product
    # (issue #195). A brief without a scene has no artwork.
    if not (brief.scene_description or "").strip():
        logger.warning("Brief %d has no scene_description", brief_id)
        if needs_text_check:
            return _upload_text_only(
                brief_row,
                layout,
                storage=storage,
                db_client=db_client,
                text_validator=text_validator,
            )
        return False, "the brief has no scene_description"

    base_seed = random.randint(0, 2**31 - 1)
    retry_hint = ""
    attempts = 0
    artwork_attempts = 0
    lettering_attempts = 0
    text_attempts = 0

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

        # The artwork gate: the picture has to be the design, not the product
        # it is printed on (issue #195).
        if artwork_validator is not None:
            artwork = artwork_validator.validate(result.raw_bytes or result.image_bytes)
            if artwork.is_product:
                artwork_attempts += 1
                if artwork_attempts <= max_artwork_retries:
                    logger.warning(
                        "Brief %d: the artwork came back as a product, retrying", brief_id
                    )
                    retry_hint = ARTWORK_RETRY_HINT
                    continue
                if needs_text_check:
                    logger.warning(
                        "Brief %d: the artwork is a product, falling back to the "
                        "typography-only design",
                        brief_id,
                    )
                    return _upload_text_only(
                        brief_row,
                        layout,
                        storage=storage,
                        db_client=db_client,
                        text_validator=text_validator,
                    )
                return False, "the artwork came back as a product and cannot be printed"
            if artwork.verdict == "unknown":
                logger.warning(
                    "Brief %d: the artwork could not be judged (%s), keeping the image",
                    brief_id,
                    artwork.error or "unknown verdict",
                )

        # The lettering gate: the artwork must not carry words of its own. The
        # image model cannot spell, and the text of the design is drawn on the
        # artwork afterwards, so a letter it drew is a letter that is wrong
        # (issue #208).
        if artwork_text_validator is not None:
            lettering = artwork_text_validator.validate(result.raw_bytes or result.image_bytes)
            if lettering.has_text:
                lettering_attempts += 1
                if lettering_attempts <= max_lettering_retries:
                    logger.warning(
                        "Brief %d: the artwork came back with lettering in it, retrying",
                        brief_id,
                    )
                    retry_hint = NO_LETTERING_RETRY_HINT
                    if brief.text_plan is not None and brief.text_plan.needs_carrier:
                        retry_hint += (
                            " The planned carrier is blank; no other sign, menu, "
                            "billboard or label may have writing."
                        )
                    continue
                if needs_text_check:
                    logger.warning(
                        "Brief %d: the artwork carries lettering, falling back to the "
                        "typography-only design",
                        brief_id,
                    )
                    return _upload_text_only(
                        brief_row,
                        layout,
                        storage=storage,
                        db_client=db_client,
                        text_validator=text_validator,
                    )
                return False, "the artwork came back with lettering in it and cannot be printed"
            if lettering.verdict == "unknown":
                logger.warning(
                    "Brief %d: the lettering could not be judged (%s), keeping the image",
                    brief_id,
                    lettering.error or "unknown verdict",
                )

        # The text is drawn onto the artwork, not asked of the model: it cannot
        # spell, and a colour in a prompt is not a colour (issue #187).
        raw_bytes = composite_text(_textured(result, brief, is_dtf), layout)

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

        text_attempts += 1
        if text_attempts > max_retries:
            logger.error(
                "Text validation failed after %d attempt(s) for brief %d: expected %s, observed %r",
                text_attempts,
                brief_id,
                verdict.expected,
                verdict.observed,
            )
            return False, (
                f"text validation failed after {text_attempts} attempt(s): "
                f"expected {verdict.expected}, observed {verdict.observed!r}"
            )

        retry_hint = _retry_hint(brief)

    transparent_bytes: bytes | None = None
    if is_dtf and result.background_removed and result.image_bytes:
        transparent_bytes = composite_text(
            _textured(result, brief, is_dtf, keyed=True), layout, transparent=True
        )

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
    artwork_validator: ArtworkValidator | None = None,
    artwork_text_validator: ArtworkTextValidator | None = None,
    hero_validator: StrictTextValidator | None = None,
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
        artwork_validator=artwork_validator,
        artwork_text_validator=artwork_text_validator,
        hero_validator=hero_validator,
        db_client=db_client,
    )
    if success:
        return {"status": "success", "brief_id": brief_id}
    return {"status": "error", "brief_id": brief_id, "error": error or "Image generation failed"}
