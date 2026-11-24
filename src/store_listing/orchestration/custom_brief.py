"""Generate a design brief from a free-text user description via LLM."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import replace
from datetime import UTC, datetime
from typing import Any

from store_listing.orchestration.art_direction import family_for, preset_for
from store_listing.orchestration.brand_guard import licensed_brand
from store_listing.orchestration.brief_prompts import (
    DESIGN_ONLY_RULES,
    art_direction_rules,
    content_scope_rules,
    copy_mode_rules,
    layout_rules,
    personal_content_scope_rules,
    product_belongs_to,
    product_catalog,
)
from store_listing.orchestration.content_scope import content_rejection
from store_listing.orchestration.copy_modes import (
    CAPTION,
    NO_TEXT,
    TYPE_HERO,
    normalise_copy_mode,
    normalise_kicker,
)
from store_listing.orchestration.design_briefs import (
    VALID_LAYOUT_TYPES,
    DesignBrief,
    DesignBriefDBClient,
    assign_batch_devices,
    assign_batch_templates,
    headline_for_text_only,
    merchandise_rejection,
)
from store_listing.orchestration.headline_voice import normalise_voice, screen_text
from store_listing.orchestration.idea_devices import device_for, idea_device_rules
from store_listing.orchestration.image_progress import GenerationWatch, publishing
from store_listing.orchestration.json_response import extract_json_objects, repair_truncated_object
from store_listing.orchestration.label_plates import plates_to_value
from store_listing.orchestration.llm_client import (
    create_answer,
    get_vision_client,
    get_vision_model,
    is_model_missing_error,
    model_error_message,
)
from store_listing.orchestration.personal_design import (
    MAX_QUERIES,
    look_at_reference,
    research_has_sources,
    research_personal_design,
    search_personal,
)
from store_listing.orchestration.scene_text import strip_lettering
from store_listing.orchestration.seed_curation import PRODUCT_TAGS, _log_llm_response

logger = logging.getLogger(__name__)


def _licensed_name_rejection(*parts: tuple[str, str | None]) -> str | None:
    """Why one of these texts names a licence, without quoting the name."""
    for label, text in parts:
        if licensed_brand(text):
            return f"the {label} names a licensed name"
    return None


def _system_prompt(enabled: Sequence[str] | None, personal_use: bool) -> str:
    """The prompt for a custom brief, offering only what is in production.

    The catalog names the product types that are in production right now, so
    a store that focuses on DTF apparel is never offered mugs (issue #215).
    """
    types = list(enabled) if enabled else sorted(PRODUCT_TAGS)
    product_types = ", ".join(f"'{t}'" for t in types)

    literal_rules = (
        "LITERAL WORDS: concept, headline_text, and scene_description are the customer's "
        "words, copied exactly. Do not invent a slogan, a joke, a new identity, or a "
        "substitute subject. Do not drop a word the customer wrote.\n\n"
    )
    personal_use_rules = ""
    introduction = (
        "You fill in the production fields of one design brief. The customer already "
        "wrote the design.\n\n"
    )
    ip_rules = (
        "IMPORTANT: Do not use licensed IP (movie characters, team logos, brand names). "
        "Respond ONLY with valid JSON."
    )
    if personal_use:
        introduction = (
            "You fill in the production fields of one design brief for a PERSONAL ITEM "
            "(not for resale). The customer already wrote the design.\n\n"
        )
        personal_use_rules = (
            "This is a personal item, and it is a print-ready shirt graphic.\n"
            "- concept and headline_text are the customer's words, copied exactly.\n"
            "- scene_description is an isolated print of that same concept. Keep every "
            "named character, place, and object. Do not swap one place for another and "
            "do not replace a named character with a generic one.\n"
            "- The garment is the background. No park, no sky, no ground, no backdrop, "
            "and no full scene behind the subject.\n"
            "- When the customer asks for a lifelike picture, the subject is lifelike "
            "and still isolated.\n\n"
        )
        ip_rules = "Respond ONLY with valid JSON."

    return (
        f"{introduction}"
        f"{literal_rules}"
        f"{personal_use_rules}"
        f"{product_catalog(types)}\n"
        f"{personal_content_scope_rules() if personal_use else content_scope_rules()}\n"
        "Return a single JSON object with this structure:\n"
        '{"brief": {\n'
        '  "concept": "design-ready phrase or visual idea",\n'
        f'  "product_type": "{product_types}",\n'
        '  "specific_products": ["t-shirts"],\n'
        '  "audience": "who would buy this",\n'
        '  "visual_style": "art direction (colors, style, typography). No text/slogans here.",\n'
        '  "confidence": 0-100,\n'
        '  "reasoning": "why this design works",\n'
        '  "copy_mode": "type_hero | caption | no_text",\n'
        '  "layout_type": "auto | full_bleed | text_only",\n'
        '  "headline_text": "short punchy text (1-6 words, words only, no emoji) '
        'or null for no_text",\n'
        '  "tagline_text": "secondary text or null",\n'
        '  "kicker_text": "short top line for type_hero (1-6 words) or null",\n'
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
        f"{copy_mode_rules()}\n"
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
        "Copy the design idea into the concept, the headline, and the scene. Do not rewrite it."
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

    scope_reason = merchandise_rejection(b)
    if scope_reason is not None:
        logger.warning("Skipping the custom brief %r: %s", concept, scope_reason)
        return None, scope_reason

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

    raw_layout = b.get("layout_type")
    layout_type = raw_layout if raw_layout in VALID_LAYOUT_TYPES else "full_bleed"
    layout_hint = raw_layout if raw_layout in VALID_LAYOUT_TYPES else None

    headline_text = b.get("headline_text")
    if headline_text is not None and not isinstance(headline_text, str):
        headline_text = None

    tagline_text = b.get("tagline_text")
    if tagline_text is not None and not isinstance(tagline_text, str):
        tagline_text = None

    raw_copy_mode = b.get("copy_mode")
    if not isinstance(raw_copy_mode, str):
        raw_copy_mode = None
    raw_scene = b.get("scene_description")
    copy_mode = normalise_copy_mode(
        raw_copy_mode,
        layout_type=layout_hint,
        headline_text=headline_text,
        tagline_text=tagline_text,
        scene_description=raw_scene if isinstance(raw_scene, str) else None,
    )
    kicker_text = normalise_kicker(b.get("kicker_text"), copy_mode=copy_mode)
    if copy_mode == NO_TEXT:
        layout_type = "full_bleed"
        headline_text = None
        tagline_text = None
        kicker_text = None
    elif copy_mode == TYPE_HERO:
        if layout_type in ("full_bleed", "text_only"):
            layout_type = "auto"
    elif (
        layout_type == "full_bleed"
        and (headline_text or tagline_text)
        and (raw_layout is None or (raw_copy_mode == CAPTION and raw_layout == "full_bleed"))
    ):
        layout_type = "auto"

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
    # purpose, so the brand is not screened there (issue #225). A no-text
    # design has no text to screen.
    if copy_mode != NO_TEXT and not personal_use:
        reason = screen_text(
            headline_text,
            tagline=tagline_text,
            scene=scene_description,
            voice=voice,
        )
        if reason is not None:
            logger.warning("Skipping the custom brief %r: %s", concept, reason)
            return None, reason

    if not personal_use:
        licensed = _licensed_name_rejection(
            ("concept", concept if isinstance(concept, str) else None),
            ("scene", raw_scene if isinstance(raw_scene, str) else None),
            ("audience", b.get("audience") if isinstance(b.get("audience"), str) else None),
            (
                "visual style",
                b.get("visual_style") if isinstance(b.get("visual_style"), str) else None,
            ),
            ("reasoning", b.get("reasoning") if isinstance(b.get("reasoning"), str) else None),
            ("kicker", kicker_text),
        )
        if licensed is not None:
            logger.warning("Skipping the custom brief %r: %s", concept, licensed)
            return None, licensed

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
            copy_mode=copy_mode,
            kicker_text=kicker_text,
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


def _arrangement_seed(words: str) -> int:
    """A stable seed for the first arrangement of this sentence."""
    return sum(ord(character) for character in words) or 1


def _personal_answer(llm_completions: Any, model: str, messages: list[dict[str, str]]) -> str:
    """One Qwen answer for the personal research step. A failure is empty."""
    raw, error = create_answer(llm_completions, model, messages, "personal_design", temperature=0.7)
    if error is not None or raw is None:
        return ""
    try:
        response = raw.parse()
    except Exception:  # noqa: BLE001
        logger.warning("The personal design answer could not be read")
        return ""
    if not response.choices:
        return ""
    return response.choices[0].message.content or ""


def _stored_text_plan(
    brief: DesignBrief, design: Any, *, image_model: str | None = None
) -> dict[str, Any] | None:
    """The text plan column. A personal graphic stores its research there.

    There is no reference-image column. The urls, facts, the one line, and the
    chosen image model ride in the existing JSON so a later render can use them.
    """
    if design is not None:
        plan: dict[str, Any] = {
            "reference_urls": list(design.reference_urls),
            "facts": list(design.facts),
            "subjects": design.subjects,
            "lettering": design.lettering,
            "customer_lettering": design.customer_lettering,
            "search_query_count": int(getattr(design, "search_query_count", 0) or 0),
            "note_count": int(getattr(design, "note_count", 0) or 0),
            "fact_count": len(design.facts),
            "reference_count": len(design.reference_urls),
            "search_errors": list(getattr(design, "search_errors", ()) or ()),
        }
        if image_model is not None:
            plan["image_model"] = image_model
        return plan
    if brief.text_plan is not None:
        return brief.text_plan.to_dict()
    return None


def _watch_phase(
    watch: GenerationWatch | None, phase: str, step: int = 0, total_steps: int = 0
) -> None:
    if watch is not None:
        watch.phase(phase, step=step, total_steps=total_steps)


def _open_watch(brief_id: int, personal_use: bool, client: Any | None) -> GenerationWatch | None:
    """A clock for this brief, continued later by the image task."""
    if client is None:
        return None
    return GenerationWatch(
        brief_id,
        personal=personal_use,
        publish=publishing(client),
        client=client,
    )


def _abandon(
    db_client: DesignBriefDBClient,
    brief_id: int,
    watch: GenerationWatch | None,
    error: str,
) -> dict[str, Any]:
    """The row is already visible. Say that this attempt stopped."""
    db_client.update_brief_image_status(brief_id, "needs_attention", error)
    _watch_phase(watch, "failed")
    return {"status": "error", "error": error, "brief_id": brief_id}


_DEFAULT_PERSONAL_STYLE = "2d"
_EMPTY_RESEARCH = "Personal research returned no sources"


def create_custom_brief(
    db_client: DesignBriefDBClient,
    llm_completions: Any,
    model: str,
    description: str,
    product_type: str,
    *,
    personal_use: bool = False,
    image_model: str | None = None,
    style_preset: str | None = None,
    enabled: Sequence[str] | None = None,
    progress_client: Any | None = None,
) -> dict[str, Any]:
    from store_listing.orchestration.personal_image_models import (
        normalize_personal_image_model,
    )

    chosen_image_model = normalize_personal_image_model(image_model) if personal_use else None
    requested_style = (style_preset or "").strip() or None
    if requested_style is not None and preset_for(requested_style) is None:
        return {
            "status": "error",
            "error": f"Unknown style preset: {requested_style}",
        }
    if requested_style is None and personal_use:
        requested_style = _DEFAULT_PERSONAL_STYLE

    scope_reason = content_rejection(("description", description))
    if scope_reason is not None:
        # A personal item waives the trademark check. It does not waive the library.
        return {"status": "error", "error": f"The brief was not created: {scope_reason}"}

    if not personal_use:
        licensed = _licensed_name_rejection(("description", description))
        if licensed is not None:
            return {"status": "error", "error": f"The brief was not created: {licensed}"}

    if enabled and product_type not in enabled:
        if len(enabled) > 1:
            # There is no single type to fall back to, so the request is
            # refused instead of answered with the wrong thing.
            return {
                "status": "error",
                "error": f"the product type {product_type} is not in production",
            }
        product_type = enabled[0]

    words = description.strip()
    brief_id = db_client.insert_design_brief(
        concept=words,
        product_type=product_type,
        specific_products=[],
        audience="",
        visual_style="",
        confidence=0,
        reasoning="",
        llm_model=model,
        batch_id="custom",
        personal_use=personal_use,
        image_status="preparing",
    )
    watch = _open_watch(brief_id, personal_use, progress_client)
    _watch_phase(watch, "writing")

    messages = build_custom_brief_prompt(
        description, product_type, personal_use=personal_use, enabled=enabled
    )

    raw, error = create_answer(llm_completions, model, messages, "custom_brief", temperature=0)
    if error is not None:
        return _abandon(db_client, brief_id, watch, error)

    try:
        headers = raw.headers
        response = raw.parse()
        _log_llm_response(response, headers, "custom_brief")
        if not response.choices:
            return _abandon(db_client, brief_id, watch, "LLM returned empty choices")
        content = response.choices[0].message.content or ""
    except Exception as e:  # noqa: BLE001
        message = model_error_message(e, model)
        if is_model_missing_error(e):
            logger.error("Custom brief LLM call failed: %s", message)
        else:
            logger.warning("Custom brief LLM call failed: %s", e)
        return _abandon(db_client, brief_id, watch, message)

    brief, reason = read_custom_brief(content, enabled, personal_use)
    if brief is None:
        if reason is None:
            # The raw answer is the only way to see what the model actually sent.
            logger.error("The LLM custom brief response could not be read")
            logger.error("Raw LLM custom brief response: %s", content)
            return _abandon(db_client, brief_id, watch, "Failed to parse LLM response")
        # The answer was read, and the brief was not usable. Saying so is
        # better than blaming the parser (issue #232).
        logger.error("The LLM custom brief was dropped: %s", reason)
        return _abandon(db_client, brief_id, watch, f"The brief was not created: {reason}")

    design = None
    if personal_use:
        from store_listing.orchestration.personal_design_gemini import (
            research_personal_design_gemini,
        )

        _watch_phase(watch, "researching")
        design = research_personal_design_gemini(words, seed=_arrangement_seed(words))
        if design is not None:
            logger.info(
                "Personal brief %d researched via Gemini full-sentence path (%d subjects)",
                brief_id,
                design.search_query_count,
            )
        else:
            design = _research_personal(
                words,
                llm_completions=llm_completions,
                model=model,
                watch=watch,
            )
        if not research_has_sources(design):
            logger.error(
                "Personal research empty for brief %s (queries=%s notes=%s); retrying once",
                brief_id,
                design.search_query_count,
                design.note_count,
            )
            design = _research_personal(
                words,
                llm_completions=llm_completions,
                model=model,
                watch=watch,
            )
        if not research_has_sources(design):
            logger.error(
                "Personal research still empty for brief %s: errors=%s",
                brief_id,
                list(design.search_errors),
            )
            return _abandon(db_client, brief_id, watch, _EMPTY_RESEARCH)
        brief = replace(
            brief,
            concept=words,
            headline_text=design.lettering,
            scene_description=design.prompt,
            tagline_text=None,
            kicker_text=None,
            copy_mode=CAPTION if design.customer_lettering else NO_TEXT,
            personal_use=True,
            style_preset=requested_style or brief.style_preset,
        )
    else:
        brief = replace(
            brief,
            concept=words,
            headline_text=words,
            scene_description=words,
            tagline_text=None,
            kicker_text=None,
            personal_use=False,
            style_preset=requested_style or brief.style_preset,
        )

    _watch_phase(watch, "planning")
    # A single brief still gets a template, so the custom path does not always
    # come back as the same centred emblem (issue #238).
    brief = assign_batch_templates([brief], f"custom:{brief.concept}")[0]
    if brief.idea_device is None:
        # The model named no device we use; give the brief one anyway.
        brief = assign_batch_devices([brief], f"custom:{brief.concept}")[0]

    # A personal-use design is not going on a marketplace, so it gets no
    # storefront copy. A sellable custom brief gets the same listing pass as
    # the daily batch (issue #297).
    if not personal_use:
        from store_listing.orchestration.listing_copy import write_batch_listing_copy

        [brief], listing_issues = write_batch_listing_copy(llm_completions, model, [brief])
        if listing_issues:
            logger.warning("Listing copy fell back: %s", "; ".join(listing_issues))

    db_client.update_design_brief(
        brief_id,
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
        copy_mode=brief.copy_mode,
        kicker_text=brief.kicker_text,
        font_color=brief.font_color,
        scene_description=brief.scene_description,
        personal_use=personal_use,
        voice=brief.voice,
        font_family=brief.font_family,
        style_preset=brief.style_preset,
        text_plan=_stored_text_plan(brief, design, image_model=chosen_image_model),
        listing_description=brief.listing_description,
        label_plates=plates_to_value(brief.label_plates),
        image_status="pending",
    )

    result: dict[str, Any] = {
        "status": "success",
        "brief_id": brief_id,
        "personal_use": personal_use,
    }
    if chosen_image_model is not None:
        result["image_model"] = chosen_image_model
    if brief.style_preset:
        result["style_preset"] = brief.style_preset
    return result


def _research_personal(
    words: str,
    *,
    llm_completions: Any,
    model: str,
    watch: GenerationWatch | None,
) -> Any:
    """Search, look, and plan. Each step is a progress phase."""
    searches = {"count": 0}
    reads = {"count": 0}

    def _llm_complete(messages: list[dict[str, str]]) -> str:
        return _personal_answer(llm_completions, model, messages)

    def search(query: str) -> list[Any]:
        searches["count"] += 1
        _watch_phase(
            watch,
            "researching",
            searches["count"],
            max(searches["count"], MAX_QUERIES),
        )
        return search_personal(query, complete=_llm_complete, look=_vision_look)

    def complete(messages: list[dict[str, str]]) -> str:
        reads["count"] += 1
        if reads["count"] > 1:
            _watch_phase(watch, "planning")
        else:
            _watch_phase(watch, "researching")
        return _personal_answer(llm_completions, model, messages)

    def _vision_look(image: bytes) -> Any:
        return look_at_reference(image, client=get_vision_client(), model=get_vision_model())

    def look(image: bytes) -> Any:
        _watch_phase(watch, "studying")
        return look_at_reference(image, client=get_vision_client(), model=get_vision_model())

    return research_personal_design(
        words,
        complete=complete,
        search=search,
        seed=_arrangement_seed(words),
        look=look,
    )
