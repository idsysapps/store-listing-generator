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

import json
import logging
import os
import random
from collections.abc import Callable
from dataclasses import replace
from io import BytesIO
from typing import Any

from PIL import Image

from store_listing.orchestration.art_direction import family_for, preset_for, tagline_family_for
from store_listing.orchestration.artwork_validation import (
    ArtworkTextValidator,
    ArtworkValidator,
)
from store_listing.orchestration.chroma_key import (
    opaque_fraction,
    remove_dark_ink,
    trim_sticker_border,
)
from store_listing.orchestration.comfyui_client import ComfyUIClient
from store_listing.orchestration.copy_modes import (
    CAPTION,
    HERO_LAYOUT_TYPES,
    NO_TEXT,
    TYPE_HERO,
    copy_mode_from_row,
    scene_fills_print_area,
)
from store_listing.orchestration.design_briefs import DesignBrief
from store_listing.orchestration.design_quality import (
    Candidate,
    QualityMetrics,
    choose_best,
    contrast_ratio,
    garment_hex,
    image_metrics,
    text_overlap,
    with_contrast,
)
from store_listing.orchestration.design_templates import composition_for, resolve_template
from store_listing.orchestration.headline_voice import DEFAULT_VOICE
from store_listing.orchestration.hero_descriptor import (
    compose_hero_descriptor,
    descriptor_enabled,
)
from store_listing.orchestration.image_generation import (
    ARTWORK_RETRY_HINT,
    NO_LETTERING_RETRY_HINT,
    ImageResult,
    font_colour,
    generate_image,
    is_dark_garment,
)
from store_listing.orchestration.image_storage import ImageStorageClient
from store_listing.orchestration.model_text import (
    ModelTextGenerator,
    is_model_text_brief,
    model_text_copy_from_brief,
    model_text_enabled,
)
from store_listing.orchestration.scene_text import strip_lettering
from store_listing.orchestration.text_compositor import (
    DTF_KEY_COLOUR,
    LayoutSpec,
    blank_canvas,
    composite_text,
)
from store_listing.orchestration.text_embed import profile_for
from store_listing.orchestration.text_integration import text_plan_from_value
from store_listing.orchestration.text_validation import StrictTextValidator, TextValidator
from store_listing.orchestration.textures import apply_texture, despeckle_image
from store_listing.orchestration.typographic_hero import (
    HeroCopy,
    TypographicHeroGenerator,
    hero_candidate_count,
    hero_copy_from_brief,
    is_typographic_hero_brief,
    typographic_hero_enabled,
)
from store_listing.orchestration.typography import accent_for

logger = logging.getLogger(__name__)

DEFAULT_MAX_TEXT_RETRIES = 2
DEFAULT_MAX_ARTWORK_RETRIES = 1
DEFAULT_MAX_LETTERING_RETRIES = 1
# A DTF design that covers most of its placement prints as a solid sheet and
# feels like plastic. This is a warning, not a rejection (issue #288).
DTF_COVERAGE_WARN = 0.65

_TRUTHY = frozenset({"1", "true", "yes", "on"})


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


def _text_embed_profile() -> str | None:
    """The DTF text embedding profile, or None when the feature is off."""
    profile = profile_for(os.environ.get("TEXT_EMBED_PROFILE"))
    return profile.name if profile is not None else None


def _text_plan_rendering_enabled() -> bool:
    """Whether the renderer executes the stored text-integration plan.

    The bake-off showed the carrier/region/method layer adds defects faster
    than it adds design. It is off by default; a brief that needs it for an
    experiment can turn it back on (issue #307).
    """
    value = os.environ.get("TEXT_PLAN_RENDERING_ENABLED", "").strip().lower()
    return value in _TRUTHY


def _model_text_caption_enabled() -> bool:
    """Whether a caption may send its words to Qwen instead of the compositor.

    Qwen is the type_hero model. The bake-off found it lost the headline on
    five of nine caption briefs, so caption routing is opt-in (issue #307).
    """
    value = os.environ.get("MODEL_TEXT_CAPTION_ENABLED", "").strip().lower()
    return value in _TRUTHY


def _recommended_materials(brief_row: dict[str, Any]) -> list[dict[str, Any]] | None:
    value = brief_row.get("recommended_materials")
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return None
    return value if isinstance(value, list) else None


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
    # The stored mode, or the mode a row written before it existed implies.
    # A no-text row may carry stale copy from an earlier plan; it is cleared
    # here so neither the compositor nor the text gate can see it.
    copy_mode = copy_mode_from_row(brief_row)
    layout_type = brief_row.get("layout_type", "full_bleed")
    headline_text = brief_row.get("headline_text")
    tagline_text = brief_row.get("tagline_text")
    kicker_text = brief_row.get("kicker_text") if copy_mode == TYPE_HERO else None
    if copy_mode == NO_TEXT:
        layout_type = "full_bleed"
        headline_text = None
        tagline_text = None
        kicker_text = None
    elif (copy_mode == TYPE_HERO and layout_type in ("full_bleed", "text_only")) or (
        copy_mode == CAPTION
        and layout_type == "full_bleed"
        and (headline_text or tagline_text or brief_row.get("text_plan"))
    ):
        layout_type = "auto"
    return DesignBrief(
        concept=brief_row["concept"],
        product_type=brief_row["product_type"],
        specific_products=[],
        audience=brief_row.get("audience", ""),
        visual_style=brief_row.get("visual_style", ""),
        confidence=0,
        reasoning="",
        source_seed_ids=[],
        layout_type=layout_type,
        headline_text=headline_text,
        tagline_text=tagline_text,
        copy_mode=copy_mode,
        kicker_text=kicker_text,
        font_color=brief_row.get("font_color"),
        regeneration_feedback=brief_row.get("regeneration_feedback"),
        scene_description=scene or None,
        voice=voice,
        font_family=font_family,
        style_preset=brief_row.get("style_preset"),
        recommended_materials=_recommended_materials(brief_row),
        text_plan=text_plan_from_value(brief_row.get("text_plan"))
        if copy_mode != NO_TEXT
        else None,
    )


def _brief_for_render(brief_row: dict[str, Any]) -> DesignBrief:
    """The brief as the renderer should see it.

    The stored text plan is off by default: the bake-off showed the
    carrier/region/method layer adds defects faster than it adds design. An
    explicit experiment can turn it back on (issue #307).

    A scene that fills the print area is not a hero subject. An already stored
    type_hero row with that scene renders as a caption (issue #312).
    """
    brief = _brief_from_row(brief_row)
    if not _text_plan_rendering_enabled():
        brief = replace(brief, text_plan=None)
    return _demote_fill_scene(brief_row, brief)


def _demote_fill_scene(brief_row: dict[str, Any], brief: DesignBrief) -> DesignBrief:
    """Render a fill-the-canvas hero as a caption of isolated motifs."""
    layout = (brief.layout_type or "").strip()
    heroish = brief.copy_mode == TYPE_HERO or layout in HERO_LAYOUT_TYPES
    if not heroish or not scene_fills_print_area(brief.scene_description):
        return brief
    logger.info(
        "Brief %s: scene fills the print area, rendering as a caption",
        brief_row.get("id"),
    )
    if layout in HERO_LAYOUT_TYPES or layout in ("full_bleed", "text_only"):
        layout = "auto"
    return replace(brief, copy_mode=CAPTION, kicker_text=None, layout_type=layout or "auto")


def _image_candidate_count() -> int:
    """How many renders to compare before uploading one.

    Two is the bake-off setting: it lets the pipeline drop a bad seed (text on
    the subject, a keyed fringe, a 65%-coverage plate). The default is one so
    a local run or a test does not silently double its GPU time; production
    sets IMAGE_CANDIDATE_COUNT=2 (issue #307).
    """
    raw = os.environ.get("IMAGE_CANDIDATE_COUNT", "1")
    try:
        return max(1, int(raw))
    except ValueError:
        logger.warning("Invalid IMAGE_CANDIDATE_COUNT=%r, using 1", raw)
        return 1


def _planned_ink(layout: LayoutSpec) -> str:
    """The ink the plan asks for, for the contrast check."""
    plan = layout.text_plan
    if plan is None:
        return layout.font_color
    if plan.ink == "light":
        return "#F7F3E8"
    if plan.ink == "dark":
        return "#111111"
    if plan.ink == "accent":
        return accent_for(layout.font_color)
    return plan.ink


def _quality_candidate(
    *,
    name: str,
    raw_bytes: bytes,
    transparent_bytes: bytes | None,
    layout: LayoutSpec,
    subject_bytes: bytes | None,
    garment_color: str | None,
) -> Candidate:
    """Score one render with the deterministic DTF gate."""
    metrics = image_metrics(transparent_bytes or raw_bytes)
    if layout.needs_text and subject_bytes is not None:
        with Image.open(BytesIO(subject_bytes)) as opened:
            size = opened.size
        overlay = composite_text(
            blank_canvas("dtf_apparel", transparent=True, size=size),
            layout,
            transparent=True,
            ink_mask_bytes=subject_bytes,
        )
        metrics = replace(metrics, text_overlap=text_overlap(overlay, subject_bytes))
    metrics = with_contrast(
        metrics, contrast_ratio(_planned_ink(layout), garment_hex(garment_color))
    )
    return Candidate(
        name=name,
        raw_bytes=raw_bytes,
        transparent_bytes=transparent_bytes or raw_bytes,
        metrics=metrics,
        layout=layout,
        subject_bytes=subject_bytes,
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
        success, _ = _generate_best_candidate(
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
    _log_dtf_coverage(brief_id, transparent_bytes)

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


def _hero_with_descriptor(
    image_bytes: bytes,
    descriptor: str | None,
    *,
    transparent: bool = False,
) -> bytes:
    """The hero artwork with its descriptor, unless it has none or is off.

    The descriptor is drawn after the strict gate because Qwen garbles a third
    block. For the DTF file it is drawn on the keyed artwork, so the outline
    can do its job on any garment colour.
    """
    if not descriptor or not descriptor.strip() or not descriptor_enabled():
        return image_bytes
    return compose_hero_descriptor(image_bytes, descriptor, transparent=transparent)


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

    if is_dtf:
        return _generate_gated_hero(
            brief_row,
            brief,
            copy,
            storage=storage,
            comfyui_client=comfyui_client,
            validator=validator,
            db_client=db_client,
        )

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

    raw_bytes = _hero_with_descriptor(result.image_bytes, copy.descriptor)
    keys = storage.upload_design(
        raw_bytes=raw_bytes,
        transparent_bytes=None,
        brief_id=brief_id,
        product_type=brief_row["product_type"],
    )
    db_client.update_brief_image_keys(
        brief_id=brief_id,
        image_key_raw=keys["raw"],
        image_key_transparent=keys.get("transparent"),
    )
    return True, None


# A keyed file that is not a decodable image cannot be measured. Test doubles
# use that shape; a real print is a PNG and is scored. The stand-in passes the
# gate so an unreadable double is uploaded the way it was before the gate.
_UNMEASURED_PRINT = QualityMetrics(
    contrast=None,
    text_overlap=0.0,
    magenta_fraction=0.0,
    coverage=0.22,
    light_fill=0.0,
    near_white_border=0.0,
)


def _measured_print_metrics(image_bytes: bytes) -> QualityMetrics | None:
    """Print metrics for a keyed PNG, or None when the bytes are not an image."""
    try:
        return image_metrics(image_bytes)
    except OSError:
        return None


def _keyed_hero_artwork(
    image_bytes: bytes,
    copy: HeroCopy,
    *,
    comfyui_client: ComfyUIClient,
    garment_color: str | None,
) -> tuple[bytes, bytes] | None:
    """The raw hero and the keyed print file, or None when the key fails."""
    raw_bytes = _hero_with_descriptor(image_bytes, copy.descriptor)
    keyed = comfyui_client.remove_background(image_bytes)
    if keyed.image_bytes is None:
        return None
    keyed_bytes = keyed.image_bytes
    if not copy.sticker:
        keyed_bytes = trim_sticker_border(keyed_bytes)
    transparent_bytes = _hero_with_descriptor(keyed_bytes, copy.descriptor, transparent=True)
    if is_dark_garment(garment_color):
        transparent_bytes = remove_dark_ink(transparent_bytes)
    return raw_bytes, transparent_bytes


def _generate_gated_hero(
    brief_row: dict[str, Any],
    brief: DesignBrief,
    copy: HeroCopy,
    *,
    storage: ImageStorageClient,
    comfyui_client: ComfyUIClient,
    validator: StrictTextValidator,
    db_client: Any,
) -> tuple[bool, str | None]:
    """Draw several spelling-ok heroes and upload one that passes the print gate.

    The spelling retries stay inside each draw. The number of draws is the
    image candidate count, so the Qwen budget is the product of the two.
    A candidate that is not a decodable image cannot be judged and is kept,
    which is how a test double still uploads. A real print that fails coverage,
    magenta, light fill or a sticker border is not uploaded.
    """
    brief_id = brief_row["id"]
    count = _image_candidate_count()
    generator = TypographicHeroGenerator(
        comfyui_client=comfyui_client,
        validator=validator,
        candidates=hero_candidate_count(),
    )
    passed: list[Candidate] = []
    failed: list[Candidate] = []
    last_error = "typographic hero generation failed"

    for index in range(count):
        result = generator.generate(copy, seed=random.randint(0, 2**31 - 1))
        if result.image_bytes is None:
            last_error = result.error or last_error
            logger.warning(
                "Typographic hero generation failed for brief %d: %s", brief_id, last_error
            )
            continue
        prepared = _keyed_hero_artwork(
            result.image_bytes,
            copy,
            comfyui_client=comfyui_client,
            garment_color=brief.garment_color,
        )
        if prepared is None:
            last_error = "background removal failed"
            logger.warning("Background removal failed for hero brief %d", brief_id)
            continue
        raw_bytes, transparent_bytes = prepared
        metrics = _measured_print_metrics(transparent_bytes)
        if metrics is None:
            passed.append(
                Candidate(
                    name=f"hero-{index}",
                    raw_bytes=raw_bytes,
                    transparent_bytes=transparent_bytes,
                    metrics=_UNMEASURED_PRINT,
                )
            )
            continue
        candidate = Candidate(
            name=f"hero-{index}",
            raw_bytes=raw_bytes,
            transparent_bytes=transparent_bytes,
            metrics=metrics,
        )
        if metrics.passed:
            passed.append(candidate)
        else:
            failed.append(candidate)

    if not passed and not failed:
        return False, last_error
    if not passed:
        best_failure = choose_best(failed)
        return _needs_attention(
            db_client,
            brief_id,
            "Every hero candidate failed the print gate: " + "; ".join(best_failure.metrics.issues),
        )

    best = choose_best(passed)
    logger.info(
        "Brief %d: chose %s of %d hero candidates (magenta %.1f%%, coverage %.0f%%)",
        brief_id,
        best.name,
        len(passed) + len(failed),
        best.metrics.magenta_fraction * 100,
        best.metrics.coverage * 100,
    )
    _log_dtf_coverage(brief_id, best.transparent_bytes)
    keys = storage.upload_design(
        raw_bytes=best.raw_bytes,
        transparent_bytes=best.transparent_bytes,
        brief_id=brief_id,
        product_type=brief_row["product_type"],
    )
    db_client.update_brief_image_keys(
        brief_id=brief_id,
        image_key_raw=keys["raw"],
        image_key_transparent=keys.get("transparent"),
    )
    return True, None


def _log_dtf_coverage(brief_id: int, image_bytes: bytes | None) -> None:
    """Log how much of the DTF print area is ink, and warn when it is large."""
    if image_bytes is None:
        return
    coverage = opaque_fraction(image_bytes)
    logger.info("Brief %d: DTF opaque coverage %.1f%%", brief_id, coverage * 100)
    if coverage > DTF_COVERAGE_WARN:
        logger.warning(
            "Brief %d: DTF coverage %.1f%% is above %.0f%%; the design may "
            "print as a large solid area and feel like plastic",
            brief_id,
            coverage * 100,
            DTF_COVERAGE_WARN * 100,
        )


def _flatten_for_ocr(image_bytes: bytes) -> bytes:
    """Composite a transparent render onto white so the OCR gate can see it."""
    try:
        image = Image.open(BytesIO(image_bytes)).convert("RGBA")
    except OSError:
        return image_bytes
    background = Image.new("RGB", image.size, (255, 255, 255))
    background.paste(image, mask=image.split()[-1])
    buffer = BytesIO()
    background.save(buffer, format="PNG")
    return buffer.getvalue()


def _generate_model_text_brief(
    brief_row: dict[str, Any],
    brief: DesignBrief,
    *,
    storage: ImageStorageClient,
    comfyui_client: ComfyUIClient,
    validator: StrictTextValidator,
    db_client: Any,
) -> tuple[bool, str | None]:
    """Render a painted line into the artwork with Qwen-Image and upload it.

    The headline is part of the image: there is no compositor pass for it.
    The tagline is still composited afterwards, the same way the hero path
    treats its descriptor, because Qwen garbles a third text block.

    The keyed DTF file is checked again: BiRefNet can decide the line is
    background and cut it off, which leaves a clean sticker with no words.
    A render that loses the text is retried with a new seed.
    """
    brief_id = brief_row["id"]
    is_dtf = brief_row["product_type"] == "dtf_apparel"

    try:
        copy = model_text_copy_from_brief(brief)
    except ValueError as error:
        return False, str(error)

    generator = ModelTextGenerator(
        comfyui_client=comfyui_client,
        validator=validator,
        candidates=hero_candidate_count(),
    )
    last_error = "model-drawn text generation failed"

    for _attempt in range(2):
        result = generator.generate(copy, seed=random.randint(0, 2**31 - 1))
        if result.image_bytes is None:
            logger.warning(
                "Model-drawn text generation failed for brief %d: %s", brief_id, result.error
            )
            return False, result.error or last_error

        raw_bytes = _hero_with_descriptor(result.image_bytes, copy.descriptor)
        transparent_bytes: bytes | None = None
        if is_dtf:
            keyed = comfyui_client.remove_background(result.image_bytes)
            if keyed.image_bytes is None:
                logger.warning(
                    "Background removal failed for model-text brief %d: %s",
                    brief_id,
                    keyed.error,
                )
                last_error = keyed.error or "background removal failed"
                continue
            keyed_bytes = keyed.image_bytes
            if not copy.sticker:
                keyed_bytes = trim_sticker_border(keyed_bytes)
            transparent_bytes = _hero_with_descriptor(
                keyed_bytes, copy.descriptor, transparent=True
            )
            keyed_check = validator.validate(
                _flatten_for_ocr(transparent_bytes),
                expected=list(copy.text_blocks),
            )
            if not keyed_check.matched:
                logger.warning(
                    "Model-text brief %d: the keyed artwork lost the text (missing=%s), retrying",
                    brief_id,
                    keyed_check.missing,
                )
                last_error = "the keyed artwork lost the text"
                continue
        if transparent_bytes is not None and is_dark_garment(brief.garment_color):
            transparent_bytes = remove_dark_ink(transparent_bytes)
        _log_dtf_coverage(brief_id, transparent_bytes)

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

    return False, last_error


def _needs_attention(
    db_client: Any,
    brief_id: int,
    message: str,
) -> tuple[bool, str]:
    """Stop silently substituting text-only when an illustration is rejected.

    The operator has to see that the render was rejected and why. Marking the
    brief keeps it out of the automatic sweep until it is explicitly
    regenerated (issue #299).
    """
    logger.warning("Brief %d needs attention: %s", brief_id, message)
    update = getattr(db_client, "update_brief_image_status", None)
    if callable(update):
        update(brief_id=brief_id, status="needs_attention", error=message)
    return False, message


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
    candidate_sink: Callable[[Candidate], None] | None = None,
) -> tuple[bool, str | None]:
    brief = _brief_for_render(brief_row)
    brief_id = brief_row["id"]
    is_dtf = brief_row["product_type"] == "dtf_apparel"
    copy_mode = brief.copy_mode

    # The typographic hero path is the only path where the image model draws
    # the words. It is opt-in: a brief has to be marked as a hero layout or
    # carry the type_hero copy mode, and the feature flag has to be on
    # (issues #254 and #258).
    if (
        copy_mode != NO_TEXT
        and typographic_hero_enabled()
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

    # A painted line is part of the art, not a layer on top of it. The
    # compositor can fill a sign or a label, but it cannot paint into smoke,
    # so these briefs go to the image model too (issue #133).
    if (
        copy_mode != NO_TEXT
        and model_text_enabled()
        and _model_text_caption_enabled()
        and hero_validator is not None
        and is_model_text_brief(brief)
    ):
        return _generate_model_text_brief(
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
        garment_color=brief.garment_color,
        text_plan=brief.text_plan,
        text_embed_profile=_text_embed_profile(),
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
                return _needs_attention(
                    db_client,
                    brief_id,
                    "The illustration came back as a product or mockup. Update the "
                    "scene to a flat graphic with no product, mockup or product photo, "
                    "then regenerate.",
                )
            if artwork.verdict == "unknown":
                logger.warning(
                    "Brief %d: the artwork could not be judged (%s), keeping the image",
                    brief_id,
                    artwork.error or "unknown verdict",
                )

        # The lettering gate: the artwork must not carry words of its own. The
        # image model cannot spell, and the text of the design is drawn on the
        # artwork afterwards, so a letter it drew is a letter that is wrong
        # (issue #208). A no-text design has no compositor pass, so the same
        # check is inverted: an explicit "no lettering" verdict is the only
        # pass, and lettering is the design failing (issue #258).
        if artwork_text_validator is not None:
            if copy_mode == NO_TEXT:
                no_text = artwork_text_validator.validate_no_text(
                    result.raw_bytes or result.image_bytes
                )
                if not no_text.matched:
                    lettering_attempts += 1
                    if lettering_attempts <= max_lettering_retries:
                        logger.warning(
                            "Brief %d: the no-text design came back with lettering, retrying",
                            brief_id,
                        )
                        retry_hint = (
                            f"{NO_LETTERING_RETRY_HINT} This design must contain no words "
                            "at all: no letters, no numbers, no signage and no labels."
                        )
                        continue
                    return _needs_attention(
                        db_client,
                        brief_id,
                        "The no-text illustration came back with lettering. Rewrite the "
                        "scene without signs, labels, badges or other readable objects, "
                        "then regenerate.",
                    )
            else:
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
                    return _needs_attention(
                        db_client,
                        brief_id,
                        "The illustration came back with lettering. Rewrite the scene "
                        "without signs, labels, badges or other readable objects, then "
                        "regenerate.",
                    )
                if lettering.verdict == "unknown":
                    logger.warning(
                        "Brief %d: the lettering could not be judged (%s), keeping the image",
                        brief_id,
                        lettering.error or "unknown verdict",
                    )

        # The text is drawn onto the artwork, not asked of the model: it cannot
        # spell, and a colour in a prompt is not a colour (issue #187). A
        # no-text design has no compositor pass at all.
        ink_mask = result.image_bytes if is_dtf and result.background_removed else None
        artwork = _textured(result, brief, is_dtf)
        raw_bytes = (
            composite_text(artwork, layout, ink_mask_bytes=ink_mask)
            if needs_text_check
            else artwork
        )
        # The distressed embed is intentionally hard for OCR to read. Validate
        # the same lockup without the wear, while keeping the embedded render
        # for the uploaded asset. The font, not the model, still owns spelling.
        validation_bytes = raw_bytes
        if needs_text_check and layout.text_embed_profile:
            validation_bytes = composite_text(
                artwork,
                replace(layout, text_embed_profile=None),
                ink_mask_bytes=ink_mask,
            )

        if text_validator is None or not needs_text_check:
            break

        verdict = text_validator.validate(
            validation_bytes,
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
    subject_bytes: bytes | None = None
    if is_dtf and result.background_removed and result.image_bytes:
        keyed_artwork = _textured(result, brief, is_dtf, keyed=True)
        subject_bytes = keyed_artwork
        transparent_bytes = (
            composite_text(keyed_artwork, layout, transparent=True, ink_mask_bytes=ink_mask)
            if needs_text_check
            else keyed_artwork
        )
    if transparent_bytes is not None and is_dark_garment(brief.garment_color):
        transparent_bytes = remove_dark_ink(transparent_bytes)
    if candidate_sink is not None:
        candidate_sink(
            _quality_candidate(
                name=f"seed-{base_seed + attempts - 1}",
                raw_bytes=raw_bytes,
                transparent_bytes=transparent_bytes,
                layout=layout,
                subject_bytes=subject_bytes,
                garment_color=brief.garment_color,
            )
        )
        return True, None
    _log_dtf_coverage(brief_id, transparent_bytes)

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


def _generate_best_candidate(
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
    """Render N candidates, upload the one the deterministic gate likes best.

    The hero and text-only paths have their own candidate logic, so they keep
    the single-call behaviour.
    """
    brief = _brief_for_render(brief_row)
    template = resolve_template(
        brief.layout_type or "full_bleed", brief.headline_text, brief.tagline_text
    )
    count = _image_candidate_count() if brief_row.get("product_type") == "dtf_apparel" else 1
    model_text_path = (
        brief.copy_mode != NO_TEXT
        and model_text_enabled()
        and _model_text_caption_enabled()
        and hero_validator is not None
        and is_model_text_brief(brief)
    )
    if (
        count <= 1
        or brief.copy_mode == TYPE_HERO
        or template == "text_only"
        or not (brief.scene_description or "").strip()
        or model_text_path
    ):
        return _generate_single_brief(
            brief_row,
            storage=storage,
            comfyui_client=comfyui_client,
            text_validator=text_validator,
            artwork_validator=artwork_validator,
            artwork_text_validator=artwork_text_validator,
            hero_validator=hero_validator,
            db_client=db_client,
        )

    candidates: list[Candidate] = []
    for _index in range(count):
        success, error = _generate_single_brief(
            brief_row,
            storage=storage,
            comfyui_client=comfyui_client,
            text_validator=text_validator,
            artwork_validator=artwork_validator,
            artwork_text_validator=artwork_text_validator,
            hero_validator=hero_validator,
            db_client=db_client,
            candidate_sink=candidates.append,
        )
        if not success:
            return False, error

    if not candidates:
        return False, "no candidate was rendered"

    best = choose_best(candidates)
    brief_id = brief_row["id"]
    logger.info(
        "Brief %d: chose %s of %d candidates (overlap %.0f%%, magenta %.1f%%, coverage %.0f%%)",
        brief_id,
        best.name,
        len(candidates),
        best.metrics.text_overlap * 100,
        best.metrics.magenta_fraction * 100,
        best.metrics.coverage * 100,
    )
    if best.metrics.issues:
        logger.warning(
            "Brief %d: best candidate still has QA issues: %s",
            brief_id,
            "; ".join(best.metrics.issues),
        )
    _log_dtf_coverage(brief_id, best.transparent_bytes)
    keys = storage.upload_design(
        raw_bytes=best.raw_bytes,
        transparent_bytes=best.transparent_bytes,
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
    success, error = _generate_best_candidate(
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
