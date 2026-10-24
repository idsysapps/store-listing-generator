"""Generate a design brief from a free-text user description via LLM."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

from store_listing.orchestration.art_direction import family_for, preset_for
from store_listing.orchestration.brief_prompts import (
    DESIGN_ONLY_RULES,
    art_direction_rules,
    layout_rules,
    product_belongs_to,
    product_catalog,
    voice_rules,
)
from store_listing.orchestration.design_briefs import (
    VALID_LAYOUT_TYPES,
    DesignBrief,
    DesignBriefDBClient,
    assign_batch_devices,
    assign_batch_templates,
    headline_for_text_only,
    plan_batch_text_integration,
)
from store_listing.orchestration.headline_voice import normalise_voice, screen_text
from store_listing.orchestration.idea_devices import device_for, idea_device_rules
from store_listing.orchestration.json_response import extract_json_objects, repair_truncated_object
from store_listing.orchestration.llm_client import (
    create_answer,
    is_model_missing_error,
    model_error_message,
)
from store_listing.orchestration.scene_text import strip_lettering
from store_listing.orchestration.seed_curation import PRODUCT_TAGS, _log_llm_response

logger = logging.getLogger(__name__)


def _system_prompt(enabled: Sequence[str] | None, personal_use: bool) -> str:
    """The prompt for a custom brief, offering only what is in production.

    The catalog names the product types that are in production right now, so
    a store that focuses on DTF apparel is never offered mugs (issue #215).
    """
    types = list(enabled) if enabled else sorted(PRODUCT_TAGS)
    product_types = ", ".join(f"'{t}'" for t in types)

    personal_use_rules = ""
    introduction = (
        "You are a print-on-demand design strategist. A customer has described a design "
        "idea. Turn it into one actionable design brief that a graphic designer can "
        "immediately execute.\n\n"
    )
    ip_rules = (
        "IMPORTANT: Do not use licensed IP (movie characters, team logos, brand names). "
        "Create original concepts. Respond ONLY with valid JSON."
    )
    if personal_use:
        introduction = (
            "You are a print-on-demand design strategist. A customer has described a design "
            "idea for PERSONAL USE ONLY (not for resale). Turn it into one actionable design "
            "brief that a graphic designer can immediately execute.\n\n"
        )
        personal_use_rules = (
            "Since this is for personal use only, you MAY reference specific franchises, "
            "characters, brands, team logos, or other IP in the design concept. The customer "
            "wants a design inspired by or featuring these elements for their own private use.\n\n"
        )
        ip_rules = "Respond ONLY with valid JSON."

    return (
        f"{introduction}"
        f"{personal_use_rules}"
        f"{product_catalog(types)}\n"
        "Return a single JSON object with this structure:\n"
        '{"brief": {\n'
        '  "concept": "design-ready phrase or visual idea",\n'
        f'  "product_type": "{product_types}",\n'
        '  "specific_products": ["t-shirts"],\n'
        '  "audience": "who would buy this",\n'
        '  "visual_style": "art direction (colors, style, typography). No text/slogans here.",\n'
        '  "confidence": 0-100,\n'
        '  "reasoning": "why this design works",\n'
        '  "layout_type": "auto | full_bleed | text_only",\n'
        '  "headline_text": "short punchy text (1-6 words, words only, no emoji) '
        'or null if full_bleed",\n'
        '  "tagline_text": "secondary text or null",\n'
        '  "font_color": "#FFFFFF",\n'
        '  "voice": "deadpan | loud | cozy",\n'
        '  "idea_device": "juxtaposition | transformation | scale | narrative | '
        'collection | pattern",\n'
        '  "style_preset": "vintage_screenprint | seventies_retro | halftone_comic | '
        "riso_two_color | hand_inked_woodcut | kawaii_flat | art_nouveau | "
        'distressed_band_tee",\n'
        '  "scene_description": "concise visual description of the illustration only, '
        'no text. Under 35 words."\n'
        "}}\n\n"
        f"{layout_rules(types)}\n"
        f"{voice_rules()}\n"
        f"{art_direction_rules()}\n"
        f"{idea_device_rules()}\n"
        f"{DESIGN_ONLY_RULES}\n"
        f"{ip_rules}"
    )


def build_custom_brief_prompt(
    description: str,
    product_type: str,
    *,
    personal_use: bool = False,
    enabled: Sequence[str] | None = None,
) -> list[dict[str, str]]:
    prompt = _system_prompt(enabled, personal_use)
    today = datetime.now(tz=UTC).strftime("%Y-%m-%d")
    user_content = (
        f"Today's date: {today}\n"
        f"Design idea: {description}\n"
        f"Requested product type: {product_type}\n\n"
        "Create one design brief for this idea."
    )
    return [
        {"role": "system", "content": prompt},
        {"role": "user", "content": user_content},
    ]


def _read_brief(
    b: Any,
    enabled: Sequence[str] | None = None,
    personal_use: bool = False,
) -> tuple[DesignBrief | None, str]:
    """The brief in one answer object, and why it is not usable when it is not.

    The reason is what the caller tells the operator with, so a brief that
    was dropped for a licensed brand does not look like an answer that could
    not be read (issue #232).
    """
    if not isinstance(b, dict):
        return None, "the answer has no 'brief' object"

    concept = b.get("concept", "")
    if not concept:
        return None, "the answer has no concept"

    product_type = b.get("product_type", "")
    if product_type not in PRODUCT_TAGS:
        return None, f"the product type {product_type!r} is not one we produce"

    if enabled and product_type not in enabled:
        if len(enabled) > 1:
            # The model answered with a type that is not in production and
            # there is no single type to fall back to.
            logger.warning(
                "Skipping the brief %r: the product type %s is not in production",
                concept,
                product_type,
            )
            return None, f"the product type {product_type} is not in production"
        product_type = enabled[0]

    raw_products = b.get("specific_products", [])
    specific_products = (
        [p for p in raw_products if isinstance(p, str)] if isinstance(raw_products, list) else []
    )
    specific_products = [
        product for product in specific_products if product_belongs_to(product, product_type)
    ]

    confidence = b.get("confidence", 0)
    if not isinstance(confidence, int):
        confidence = 0
    confidence = max(0, min(100, confidence))

    layout_type = b.get("layout_type", "full_bleed")
    if layout_type not in VALID_LAYOUT_TYPES:
        layout_type = "full_bleed"

    headline_text = b.get("headline_text")
    if headline_text is not None and not isinstance(headline_text, str):
        headline_text = None

    tagline_text = b.get("tagline_text")
    if tagline_text is not None and not isinstance(tagline_text, str):
        tagline_text = None

    font_color = b.get("font_color")
    if font_color is not None and not isinstance(font_color, str):
        font_color = None

    scene_description = b.get("scene_description")
    if scene_description is not None and not isinstance(scene_description, str):
        scene_description = None
    if scene_description is not None:
        scene_description = strip_lettering(scene_description) or None

    if layout_type == "text_only":
        headline_text = headline_for_text_only(concept, headline_text)
        if headline_text is None:
            logger.warning("A text_only design needs a headline: %r", concept)
            return None, "a text_only design needs a headline"

    voice = normalise_voice(b.get("voice") if isinstance(b.get("voice"), str) else None)
    # How the scene carries its idea. The custom path has no batch slots, so
    # the device the model named is the one the scene was written for.
    idea_device = b.get("idea_device")
    if not isinstance(idea_device, str) or device_for(idea_device) is None:
        idea_device = None
    # The look of the design. A brief that names a look we do not print has no
    # look, and the voice decides the type then (issue #227).
    style_preset = b.get("style_preset")
    if not isinstance(style_preset, str):
        style_preset = None
    preset = preset_for(style_preset)
    if preset is None:
        style_preset = None
    # The family of the type is decided here, not by the model: the model picks
    # a tone, and the type follows the art direction (issue #226).
    font_family = family_for(preset, voice)

    # The text of a design is the part the buyer reads, so it is the part that
    # has to be good. A brief for personal use may name a franchise on
    # purpose, so the brand is not screened there (issue #225).
    reason = screen_text(
        headline_text,
        tagline=tagline_text,
        scene=scene_description,
        voice=voice,
        allow_brands=personal_use,
    )
    if reason is not None:
        logger.warning("Skipping the custom brief %r: %s", concept, reason)
        return None, reason

    return (
        DesignBrief(
            concept=concept,
            product_type=product_type,
            specific_products=specific_products,
            audience=str(b.get("audience", "")),
            visual_style=str(b.get("visual_style", "")),
            confidence=confidence,
            reasoning=str(b.get("reasoning", "")),
            source_seed_ids=[],
            layout_type=layout_type,
            headline_text=headline_text,
            tagline_text=tagline_text,
            font_color=font_color,
            scene_description=scene_description,
            voice=voice,
            font_family=font_family,
            style_preset=style_preset,
            idea_device=idea_device,
        ),
        "",
    )


def read_custom_brief(
    text: str,
    enabled: Sequence[str] | None = None,
    personal_use: bool = False,
) -> tuple[DesignBrief | None, str | None]:
    """The brief in the answer, and why there is none when there is none.

    The reason is None when the answer held no JSON at all, and a sentence
    when the brief was read but was not usable.

    The model does not always answer with only JSON: it appends a note, or
    answers with a second object. The answer is read value by value, because
    a parser that read two values as one lost the brief (issue #209).
    """
    objects = extract_json_objects(text)
    if objects:
        reason = "the answer has no usable brief"
        for obj in objects:
            brief, why = _read_brief(obj.get("brief"), enabled, personal_use)
            if brief is not None:
                return brief, None
            reason = why
        return None, reason

    # The answer held no JSON at all. A model that ran out of room stops in
    # the middle of its answer, and the parts that arrived are still good, so
    # the object is closed and read (issue #232).
    repaired = repair_truncated_object(text)
    if repaired is not None:
        brief, why = _read_brief(repaired.get("brief"), enabled, personal_use)
        if brief is not None:
            logger.warning(
                "The answer was cut short; the brief was read from the part that arrived"
            )
            return brief, None
        return None, why
    return None, None


def parse_custom_brief_response(
    text: str,
    enabled: Sequence[str] | None = None,
    personal_use: bool = False,
) -> DesignBrief | None:
    """The brief in the answer, or None when it cannot be read."""
    brief, _ = read_custom_brief(text, enabled, personal_use)
    return brief


def create_custom_brief(
    db_client: DesignBriefDBClient,
    llm_completions: Any,
    model: str,
    description: str,
    product_type: str,
    *,
    personal_use: bool = False,
    enabled: Sequence[str] | None = None,
) -> dict[str, Any]:
    if enabled and product_type not in enabled:
        if len(enabled) > 1:
            # There is no single type to fall back to, so the request is
            # refused instead of answered with the wrong thing.
            return {
                "status": "error",
                "error": f"the product type {product_type} is not in production",
            }
        product_type = enabled[0]

    messages = build_custom_brief_prompt(
        description, product_type, personal_use=personal_use, enabled=enabled
    )

    raw, error = create_answer(llm_completions, model, messages, "custom_brief", temperature=0)
    if error is not None:
        return {"status": "error", "error": error}

    try:
        headers = raw.headers
        response = raw.parse()
        _log_llm_response(response, headers, "custom_brief")
        if not response.choices:
            return {"status": "error", "error": "LLM returned empty choices"}
        content = response.choices[0].message.content or ""
    except Exception as e:  # noqa: BLE001
        message = model_error_message(e, model)
        if is_model_missing_error(e):
            logger.error("Custom brief LLM call failed: %s", message)
        else:
            logger.warning("Custom brief LLM call failed: %s", e)
        return {"status": "error", "error": message}

    brief, reason = read_custom_brief(content, enabled, personal_use)
    if brief is None:
        if reason is None:
            # The raw answer is the only way to see what the model actually sent.
            logger.error("The LLM custom brief response could not be read")
            logger.error("Raw LLM custom brief response: %s", content)
            return {"status": "error", "error": "Failed to parse LLM response"}
        # The answer was read, and the brief was not usable. Saying so is
        # better than blaming the parser (issue #232).
        logger.error("The LLM custom brief was dropped: %s", reason)
        return {"status": "error", "error": f"The brief was not created: {reason}"}

    # A single brief still gets a template, so the custom path does not always
    # come back as the same centred emblem (issue #238).
    brief = assign_batch_templates([brief], f"custom:{brief.concept}")[0]
    if brief.idea_device is None:
        # The model named no device we use; give the brief one anyway.
        brief = assign_batch_devices([brief], f"custom:{brief.concept}")[0]
    # The template is known now, so the model can choose how the words live
    # in it. A failed pass leaves the brief on the flat treatment.
    brief = plan_batch_text_integration(llm_completions, model, [brief])[0]

    brief_id = db_client.insert_design_brief(
        concept=brief.concept,
        product_type=brief.product_type,
        specific_products=brief.specific_products,
        audience=brief.audience,
        visual_style=brief.visual_style,
        confidence=brief.confidence,
        reasoning=brief.reasoning,
        llm_model=model,
        batch_id="custom",
        layout_type=brief.layout_type,
        headline_text=brief.headline_text,
        tagline_text=brief.tagline_text,
        font_color=brief.font_color,
        scene_description=brief.scene_description,
        personal_use=personal_use,
        voice=brief.voice,
        font_family=brief.font_family,
        style_preset=brief.style_preset,
        text_plan=brief.text_plan.to_dict() if brief.text_plan else None,
    )

    return {"status": "success", "brief_id": brief_id, "personal_use": personal_use}
