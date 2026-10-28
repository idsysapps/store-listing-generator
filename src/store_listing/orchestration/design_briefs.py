"""LLM-driven design brief generation from curated trend signals.

Synthesizes active seeds and their trend data into actionable design
briefs that a graphic designer can immediately execute.
"""

from __future__ import annotations

import json
import logging
import os
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, date, datetime
from typing import Any, Protocol

from store_listing.orchestration.art_direction import family_for, preset_for
from store_listing.orchestration.brief_mix import (
    BRIEF_MIX_REGENERATE_ONCE,
    MAX_BRIEFS_PER_THEME,
    MIN_EVERGREEN_BRIEFS,
    classify_brief_mix,
    enforce_brief_mix,
    enforce_style_mix,
    event_theme_for_query,
    evergreen_count,
)
from store_listing.orchestration.brief_prompts import (
    DESIGN_ONLY_RULES,
    DIVERSITY_RULES,
    art_direction_rules,
    copy_mode_rules,
    layout_rules,
    product_belongs_to,
    product_catalog,
    voice_rules,
)
from store_listing.orchestration.context_seeds import get_upcoming_events
from store_listing.orchestration.copy_modes import (
    CAPTION,
    HERO_LAYOUT_TYPES,
    NO_TEXT,
    TYPE_HERO,
    assign_batch_copy_modes,
    copy_mode_from_row,
    normalise_copy_mode,
    normalise_kicker,
)
from store_listing.orchestration.design_templates import (
    LEGACY_LAYOUTS,
    TEMPLATE_IDS,
    assign_templates,
    resolve_template,
)
from store_listing.orchestration.headline_voice import DEFAULT_VOICE, normalise_voice, screen_text
from store_listing.orchestration.idea_devices import (
    device_for,
    idea_device_rules,
    plan_device_slots,
)
from store_listing.orchestration.json_response import (
    LLMResponseParseError,
    extract_json_objects,
    merge_items,
)
from store_listing.orchestration.llm_client import (
    create_answer,
    is_model_missing_error,
    model_error_message,
)
from store_listing.orchestration.promotion import ORIGIN_TREND
from store_listing.orchestration.scene_text import strip_lettering
from store_listing.orchestration.seed_curation import PRODUCT_TAGS, _log_llm_response
from store_listing.orchestration.text_integration import (
    ATTACHABLE_CARRIERS,
    SURFACE_METHODS,
    PlannedText,
    TextPlan,
    build_text_plan_entries,
    build_text_plan_prompt,
    carrier_is_attached,
    check_text_plan_fit,
    check_text_plan_mix,
    infer_surface,
    read_text_plan_response,
    text_plan_from_value,
)

logger = logging.getLogger(__name__)

MAX_SEEDS_PER_BRIEF = int(os.environ.get("LLM_MAX_BRIEF_SEEDS", "20"))
MAX_BRIEFS = int(os.environ.get("LLM_MAX_BRIEFS", "8"))


def _text_plan_enabled() -> bool:
    """Whether the batch gets a second pass that plans how its text lives."""
    return os.environ.get("LLM_TEXT_PLAN", "true").lower() not in ("0", "false", "no")


def _text_plan_regenerate_once() -> bool:
    """Whether a batch whose text mix is poor is planned once more."""
    return os.environ.get("LLM_TEXT_PLAN_REGENERATE", "true").lower() not in (
        "0",
        "false",
        "no",
    )


# Whether a batch whose text was rejected is asked for again with the reason.
BRIEF_TEXT_REGENERATE_ONCE = os.environ.get("LLM_BRIEF_TEXT_REGENERATE", "true").lower() not in (
    "0",
    "false",
    "no",
)


def system_prompt(enabled: Sequence[str] | None = None) -> str:
    """The prompt for the brief batch, offering only what is in production.

    The catalog names the product types that are in production right now, so
    a store that focuses on DTF apparel is never offered mugs (issue #215).
    """
    types = list(enabled) if enabled else sorted(PRODUCT_TAGS)
    catalog = product_catalog(types)
    product_types = ", ".join(f"'{t}'" for t in types)
    return (
        "You are a print-on-demand design strategist. Your job is to synthesize trend "
        "signals into actionable design briefs that a graphic designer can immediately "
        "execute.\n\n"
        f"{catalog}\n"
        "For each brief:\n"
        "- concept: A specific, design-ready phrase or visual idea (not a category)\n"
        f"- product_type: exactly one of {product_types}\n"
        "- specific_products: array of 1-3 specific products from the catalog above\n"
        "- audience: who would buy this (demographics, interests)\n"
        "- visual_style: the colours and the details of this one design (colour "
        "palette, the details of the drawing). Do NOT name a style or a look in this "
        "field - the style_preset decides the look. Do NOT include any text/slogan "
        "either - text is rendered separately via the layout fields below.\n"
        "- confidence: 0-100, how confident you are this will sell\n"
        "- reasoning: why this concept is timely or evergreen and marketable (cite the "
        "trend signals)\n"
        "- source_seed_ids: array of seed IDs from the input that inspired this brief\n\n"
        f"{layout_rules(types)}\n"
        f"{copy_mode_rules()}\n"
        f"{voice_rules()}\n"
        f"{art_direction_rules()}\n"
        f"{idea_device_rules()}\n"
        f"{DESIGN_ONLY_RULES}\n"
        "MATERIALS — picking the right substrate:\n"
        "- recommended_materials: array of objects from the available_materials list, each with "
        "{external_id, name, color, substrate_type, unit_cost}. Pick 1-3 materials that best "
        "match this design's aesthetic and target audience. Consider color contrast with the "
        "design, substrate suitability, and unit cost for margin.\n"
        "- Only recommend a material whose product_type matches the brief.\n"
        "- If no materials are available for the product_type, omit recommended_materials.\n\n"
        "IMPORTANT: Do not suggest designs requiring licensed IP (movie characters, "
        "team logos, brand names). Create original concepts inspired by cultural moments.\n\n"
        "Return ONLY valid JSON with this exact structure:\n"
        '{"briefs": [{"concept": "...", "product_type": "...", "specific_products": [...], '
        '"audience": "...", "visual_style": "...", "confidence": 0-100, '
        '"reasoning": "...", "source_seed_ids": [id1, id2], '
        '"layout_type": "auto", "headline_text": "...", '
        '"tagline_text": "...", "copy_mode": "type_hero | caption | no_text", '
        '"kicker_text": "... or null", "font_color": "#000000", '
        '"voice": "deadpan | loud | cozy", '
        '"idea_device": "juxtaposition | transformation | scale | narrative | '
        'collection | pattern", '
        '"style_preset": "vintage_screenprint | seventies_retro | halftone_comic | '
        "riso_two_color | hand_inked_woodcut | kawaii_flat | art_nouveau | "
        'distressed_band_tee", '
        '"scene_description": "...", '
        '"recommended_materials": [{"external_id": "...", "name": "...", '
        '"color": "...", "substrate_type": "...", "unit_cost": 0.00}]}]}'
    )


@dataclass(frozen=True)
class BriefableCandidate:
    active_seed_id: int
    query: str
    source: str
    query_type: str
    score: int
    delta: int
    promotion_score: int
    product_tags: list[str] = field(default_factory=list)
    origin: str = ORIGIN_TREND
    theme: str | None = None
    seasonal_window_end: date | None = None


VALID_LAYOUT_TYPES: set[str] = {
    "auto",
    "full_bleed",
    "text_only",
    # The templates the pipeline assigns.
    *TEMPLATE_IDS,
    # The ids the briefs used before the templates existed.
    *LEGACY_LAYOUTS,
}

# The most words a headline may have. A typography-only design without a
# headline falls back to its concept, and a long concept is not a headline.
MAX_HEADLINE_WORDS = 6


def headline_for_text_only(concept: str, headline_text: str | None) -> str | None:
    """The headline of a typography-only design.

    A typography-only design is nothing but its text, so it needs a headline.
    A short concept makes a fine headline, a long one does not.
    """
    if headline_text and headline_text.strip():
        return headline_text
    words = concept.split()
    if 1 <= len(words) <= MAX_HEADLINE_WORDS:
        return concept
    return None


@dataclass(frozen=True)
class DesignBrief:
    concept: str
    product_type: str
    specific_products: list[str]
    audience: str
    visual_style: str
    confidence: int
    reasoning: str
    source_seed_ids: list[int]
    layout_type: str = "full_bleed"
    headline_text: str | None = None
    tagline_text: str | None = None
    # How the design carries its words: 'type_hero', 'caption' or 'no_text'.
    # The template stays geometry; this is whether the design uses a copy
    # treatment at all and which model draws it (issue #258).
    copy_mode: str = CAPTION
    # A short line above the hero word. Only the typographic hero path draws it.
    kicker_text: str | None = None
    font_color: str | None = None
    regeneration_feedback: str | None = None
    scene_description: str | None = None
    recommended_materials: list[dict[str, Any]] | None = None
    # The voice of the text of the design: 'deadpan', 'loud' or 'cozy'.
    voice: str = DEFAULT_VOICE
    # The family the text of the design is drawn in.
    font_family: str | None = None
    # The look the design is drawn in.
    style_preset: str | None = None
    # How the scene carries its idea: 'juxtaposition', 'transformation', ...
    idea_device: str | None = None
    # How the words live in the design: painted into the art, carved into a
    # stone, printed on a label. Written by the text-plan pass; None keeps the
    # design on the flat compositor treatment it had before (issue #246).
    text_plan: TextPlan | None = None


class DesignBriefDBClient(Protocol):
    def list_briefable_seeds(
        self, cooldown_days: int | None = None, limit: int | None = None
    ) -> list[BriefableCandidate]: ...

    def list_active_materials(
        self,
        product_type: str | None = None,
    ) -> list[dict[str, Any]]: ...

    def insert_design_brief(
        self,
        concept: str,
        product_type: str,
        specific_products: list[str],
        audience: str,
        visual_style: str,
        confidence: int,
        reasoning: str,
        llm_model: str,
        batch_id: str,
        layout_type: str = "full_bleed",
        headline_text: str | None = None,
        tagline_text: str | None = None,
        copy_mode: str = CAPTION,
        kicker_text: str | None = None,
        font_color: str | None = None,
        scene_description: str | None = None,
        personal_use: bool = False,
        recommended_materials: list[dict[str, Any]] | None = None,
        voice: str = DEFAULT_VOICE,
        font_family: str | None = None,
        style_preset: str | None = None,
        text_plan: dict[str, Any] | None = None,
    ) -> int: ...

    def insert_brief_source(self, brief_id: int, active_seed_id: int) -> int | None: ...


def diversity_rules(
    max_per_theme: int = MAX_BRIEFS_PER_THEME,
    min_evergreen: int = MIN_EVERGREEN_BRIEFS,
) -> str:
    """The diversity rules with the quotas filled in."""
    return DIVERSITY_RULES.format(max_per_theme=max_per_theme, min_evergreen=min_evergreen)


def build_brief_prompt(
    candidates: list[BriefableCandidate],
    context: list[dict[str, Any]],
    today: date,
    materials: list[dict[str, Any]] | None = None,
    extra_instructions: list[str] | None = None,
    enabled: Sequence[str] | None = None,
) -> list[dict[str, str]]:
    truncated = candidates[:MAX_SEEDS_PER_BRIEF]

    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for c in truncated:
        by_source[c.source].append(
            {
                "id": c.active_seed_id,
                "query": c.query,
                "query_type": c.query_type,
                "score": c.score,
                "delta": c.delta,
                "promotion_score": c.promotion_score,
                "product_tags": c.product_tags,
            }
        )

    payload: dict[str, Any] = {
        "date": today.isoformat(),
        # The events are named, but their seed lists are not: the seed lists
        # sat next to the signals and read as the instruction (issue #200).
        "upcoming_events": [
            {
                "name": event.get("name"),
                "days_until": event.get("days_until"),
                "theme": event.get("theme") or event.get("name"),
            }
            for event in context
        ],
        "max_briefs": MAX_BRIEFS,
        # One device per brief, so the batch covers the vocabulary instead of
        # the model writing the same still life eight times (issue #240).
        "brief_slots": [
            {"brief": index + 1, "idea_device": device}
            for index, device in enumerate(plan_device_slots(MAX_BRIEFS, today.isoformat()))
        ],
        "signals_by_source": dict(by_source),
    }

    if materials:
        payload["available_materials"] = materials

    user_content = json.dumps(payload, indent=2)

    system = system_prompt(enabled)
    if extra_instructions:
        system = f"{system}\n" + "\n".join(extra_instructions)
    system = f"{system}\n{diversity_rules()}"

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user_content},
    ]


def _brief_from(
    b: dict[str, Any],
    enabled: Sequence[str] | None = None,
    materials: list[dict[str, Any]] | None = None,
    rejections: list[str] | None = None,
) -> DesignBrief | None:
    """The brief in one answer object, or None when it is not usable."""
    concept = b.get("concept", "")
    if not concept:
        return None

    product_type = b.get("product_type", "")
    if product_type not in PRODUCT_TAGS:
        return None

    if enabled and product_type not in enabled:
        if len(enabled) > 1:
            # The model answered with a type that is not in production and
            # there is no single type to fall back to.
            logger.warning(
                "Skipping brief %r: the product type %s is not in production",
                concept,
                product_type,
            )
            return None
        # One type is in production, so the brief is that type.
        product_type = enabled[0]

    raw_products = b.get("specific_products", [])
    specific_products = (
        [p for p in raw_products if isinstance(p, str)] if isinstance(raw_products, list) else []
    )
    kept_products = [
        product for product in specific_products if product_belongs_to(product, product_type)
    ]
    if specific_products and not kept_products:
        logger.warning(
            "Brief %r named the products %s, which are not %s products",
            concept,
            specific_products,
            product_type,
        )
    specific_products = kept_products

    confidence = b.get("confidence", 0)
    if not isinstance(confidence, int):
        confidence = 0
    confidence = max(0, min(100, confidence))

    source_seed_ids = [i for i in b.get("source_seed_ids", []) if isinstance(i, int)]

    raw_layout = b.get("layout_type")
    layout_type = raw_layout if raw_layout in VALID_LAYOUT_TYPES else "full_bleed"
    # A missing layout is not the same as an explicit full_bleed: a model that
    # wrote a headline but forgot the layout wrote a caption, not art-only.
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
    copy_mode = normalise_copy_mode(
        raw_copy_mode,
        layout_type=layout_hint,
        headline_text=headline_text,
        tagline_text=tagline_text,
    )
    kicker_text = normalise_kicker(b.get("kicker_text"), copy_mode=copy_mode)
    if copy_mode == NO_TEXT:
        # A no-text design has no copy at all, even when the model ignored the
        # instruction and wrote some. The layout is forced art-only so the
        # renderer cannot accidentally composite the stale words.
        layout_type = "full_bleed"
        headline_text = None
        tagline_text = None
        kicker_text = None
    elif copy_mode == TYPE_HERO:
        # The hero path ignores the template, but the template is the fallback
        # when Qwen is off. A hero brief cannot live on a full-bleed or
        # text-only row.
        if layout_type in ("full_bleed", "text_only"):
            layout_type = "auto"
    elif (
        layout_type == "full_bleed"
        and (headline_text or tagline_text)
        and (raw_layout is None or (raw_copy_mode == CAPTION and raw_layout == "full_bleed"))
    ):
        # A caption with copy cannot be art-only. A missing layout is filled
        # in; an explicit caption is corrected. An explicit full_bleed with
        # stale copy was already treated as no_text above.
        layout_type = "auto"

    font_color = b.get("font_color")
    if font_color is not None and not isinstance(font_color, str):
        font_color = None

    scene_description = b.get("scene_description")
    if scene_description is not None and not isinstance(scene_description, str):
        scene_description = None
    if scene_description is not None:
        # The image model cannot spell, so a scene that quotes the words to
        # draw comes back misspelled - the bubble on the image of brief
        # #100 read "To get to the onthe sid" (issue #208). The scene
        # keeps the picture and loses the words.
        scene_description = strip_lettering(scene_description) or None
        if scene_description is None:
            logger.warning("Skipping brief %r: the scene description is only words", concept)
            return None

    if layout_type == "text_only":
        headline_text = headline_for_text_only(concept, headline_text)
        if headline_text is None:
            logger.warning("Skipping brief %r: a text_only design needs a headline", concept)
            return None

    voice = normalise_voice(b.get("voice") if isinstance(b.get("voice"), str) else None)
    # The device is how the scene carries its idea. A device we do not use is
    # no device; the batch slot fills it in later (issue #240).
    idea_device = b.get("idea_device")
    if not isinstance(idea_device, str):
        idea_device = None
    if device_for(idea_device) is None:
        idea_device = None
    # The look of the design. A brief that names a look we do not print has
    # no look, and the voice decides the type then (issue #227).
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
    # has to be good. A headline that names a brand, that lists three nouns or
    # that only restates the picture is dropped and the batch asks again
    # (issue #225). A no-text design has no text to screen.
    if copy_mode != NO_TEXT:
        reason = screen_text(
            headline_text,
            tagline=tagline_text,
            scene=scene_description,
            voice=voice,
        )
        if reason is not None:
            logger.warning("Skipping brief %r: %s", concept, reason)
            if rejections is not None:
                rejections.append(f"{reason} (headline: {headline_text!r})")
            return None

    recommended_materials = b.get("recommended_materials")
    if recommended_materials is not None and not isinstance(recommended_materials, list):
        recommended_materials = None
    if recommended_materials is not None:
        recommended_materials = _usable_materials(recommended_materials, product_type, materials)

    return DesignBrief(
        concept=concept,
        product_type=product_type,
        specific_products=specific_products,
        audience=str(b.get("audience", "")),
        visual_style=str(b.get("visual_style", "")),
        confidence=confidence,
        reasoning=str(b.get("reasoning", "")),
        source_seed_ids=source_seed_ids,
        layout_type=layout_type,
        headline_text=headline_text,
        tagline_text=tagline_text,
        copy_mode=copy_mode,
        kicker_text=kicker_text,
        font_color=font_color,
        scene_description=scene_description,
        recommended_materials=recommended_materials,
        voice=voice,
        font_family=font_family,
        style_preset=style_preset,
        idea_device=idea_device,
    )


def _usable_materials(
    recommended: list[Any],
    product_type: str,
    materials: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    """The recommended materials that we actually offered.

    The model can invent a material, and it can recommend a substrate that
    does not serve the product type of the brief. Both are dropped: a brief
    that names a material we do not carry cannot be printed on it
    (issue #215).

    Without a list of offered materials there is nothing to check against, so
    the recommendations are kept as they are.
    """
    if materials is None:
        return [entry for entry in recommended if isinstance(entry, dict)]
    if not materials:
        return []

    by_id = {m.get("external_id"): m for m in materials if m.get("external_id")}

    usable: list[dict[str, Any]] = []
    for entry in recommended:
        if not isinstance(entry, dict):
            continue
        material = by_id.get(entry.get("external_id"))
        if material is None:
            logger.warning(
                "Dropping a recommended material we did not offer: %s", entry.get("external_id")
            )
            continue
        types = material.get("product_type") or []
        if types and product_type not in types:
            logger.warning(
                "Dropping the recommended material %s: it does not serve %s",
                entry.get("external_id"),
                product_type,
            )
            continue
        usable.append(entry)
    return usable


def read_brief_response(
    text: str,
    enabled: Sequence[str] | None = None,
    materials: list[dict[str, Any]] | None = None,
    rejections: list[str] | None = None,
) -> tuple[list[DesignBrief], str | None]:
    """The briefs in the answer, and the reason the answer could not be read.

    The model does not always answer with only JSON: it appends a note,
    repeats the batch, or splits it in two. The answer is read value by
    value and the batches are merged, because a parser that read two values
    as one threw the whole batch away (issue #209).

    ``rejections`` collects the reasons the briefs were dropped for, so the
    batch can ask again with the reason in the prompt (issue #225).
    """
    objects = extract_json_objects(text)
    if not objects:
        return [], "no JSON object in the LLM response"

    if not any(isinstance(obj.get("briefs"), list) for obj in objects):
        return [], "the LLM response has no 'briefs' array"

    raw_briefs = [b for b in merge_items(objects, "briefs") if isinstance(b, dict)]

    result: list[DesignBrief] = []
    seen: set[tuple[str, str]] = set()
    for b in raw_briefs:
        brief = _brief_from(b, enabled, materials, rejections)
        if brief is None:
            continue
        # A repeated batch must not become two briefs.
        key = (brief.concept, brief.product_type)
        if key in seen:
            continue
        seen.add(key)
        result.append(brief)

    return result, None


def parse_brief_response(text: str) -> list[DesignBrief]:
    """The briefs in the answer, or an empty list when it cannot be read."""
    briefs, _ = read_brief_response(text)
    return briefs


def offered_materials(
    db_client: DesignBriefDBClient,
    product_type: str | None,
) -> list[dict[str, Any]]:
    """The materials the model is offered.

    The catalogue is filtered by the product type in production, because a
    brief of one type cannot be printed on the substrate of another
    (issue #215).

    When the filter matches nothing the model is offered every active
    material instead: cost-book publishes a material without a product type,
    so the typed list can be empty while the catalogue is full. An empty list
    would leave the briefs without a substrate and drop every recommendation
    (issue #236).
    """
    if not product_type:
        return db_client.list_active_materials()

    materials = db_client.list_active_materials(product_type=product_type)
    if materials:
        return materials

    logger.warning(
        "No material serves the product type %s, offering every active material instead",
        product_type,
    )
    return db_client.list_active_materials()


def generate_design_briefs(
    db_client: DesignBriefDBClient,
    llm_completions: Any,
    model: str,
    today: date | None = None,
    enabled: Sequence[str] | None = None,
) -> dict[str, Any]:
    today = today or datetime.now(tz=UTC).date()

    candidates = db_client.list_briefable_seeds(limit=MAX_SEEDS_PER_BRIEF)
    if not candidates:
        logger.warning("Design brief generation skipped: no briefable seeds")
        return {
            "status": "success",
            "briefs_created": 0,
            "briefable": 0,
            "reason": "no briefable seeds",
        }

    # The materials of the types in production: a brief of another type
    # cannot be printed on them (issue #215).
    material_type = enabled[0] if enabled and len(enabled) == 1 else None
    materials = offered_materials(db_client, material_type)
    context = get_upcoming_events(today)
    messages = build_brief_prompt(candidates, context, today, materials=materials, enabled=enabled)

    rejections: list[str] = []
    briefs, error = _request_briefs_with_retry(
        llm_completions, model, messages, enabled, materials, rejections
    )
    if briefs is None:
        return {"status": "error", "error": error or "LLM brief generation failed"}

    if rejections and BRIEF_TEXT_REGENERATE_ONCE:
        # The text of some briefs was rejected. Ask once more and say what was
        # wrong, because a brief with a bad headline is a design nobody wears
        # and a batch that is a batch is worth more than a batch that is short
        # (issue #225).
        logger.info("Asking for the briefs again: %d texts were rejected", len(rejections))
        retry_messages = build_brief_prompt(
            candidates,
            context,
            today,
            materials=materials,
            enabled=enabled,
            extra_instructions=[
                (
                    f"The previous answer had {len(rejections)} briefs whose text was "
                    f"rejected: {'; '.join(rejections[:5])}. Write new briefs whose headline "
                    "is a line a buyer would wear: it says something in the voice of the "
                    "brief, it is not a caption of the picture, it is not a list of nouns "
                    "and it names no brand or franchise."
                ),
            ],
        )
        retry_briefs, retry_error = _request_briefs(
            llm_completions, model, retry_messages, enabled, materials, []
        )
        if retry_error:
            logger.warning("The second brief attempt failed: %s", retry_error)
        if retry_briefs:
            briefs = _merge_briefs(briefs, retry_briefs)

    valid_ids = {c.active_seed_id for c in candidates}
    batch_id = today.isoformat()
    seed_themes = {
        c.active_seed_id: (c.theme or event_theme_for_query(c.query)) for c in candidates
    }

    kept, dropped = enforce_brief_mix(briefs, seed_themes)
    evergreen = evergreen_count(kept, seed_themes)
    logger.info(
        "Design brief batch: %d briefs, %d evergreen, %d dropped over the event quota",
        len(kept),
        evergreen,
        len(dropped),
    )

    # The shop sells eight looks and three voices, so a batch that came back
    # with the same look three times over looks like one design repeated
    # (issue #227).
    kept, style_dropped = enforce_style_mix(kept)
    if style_dropped:
        logger.info(
            "Dropped %d briefs over the look quota: %s",
            len(style_dropped),
            ", ".join(brief.concept for brief in style_dropped),
        )
        dropped = list(dropped) + list(style_dropped)

    if evergreen < MIN_EVERGREEN_BRIEFS and BRIEF_MIX_REGENERATE_ONCE:
        # The model answered with a season. Ask once more, and say what went
        # wrong, before shipping a batch that is not a batch (issue #200).
        retry_messages = build_brief_prompt(
            candidates,
            context,
            today,
            materials=materials,
            enabled=enabled,
            extra_instructions=[
                (
                    f"The previous answer was {evergreen} evergreen briefs out of "
                    f"{len(kept)}. Write at least {MIN_EVERGREEN_BRIEFS} briefs that are "
                    "not tied to a calendar event."
                ),
            ],
        )
        retry_briefs, retry_error = _request_briefs(
            llm_completions, model, retry_messages, enabled, materials
        )
        if retry_error:
            logger.warning("The second brief attempt failed: %s", retry_error)
        if retry_briefs:
            retry_kept, retry_dropped = enforce_brief_mix(retry_briefs, seed_themes)
            retry_evergreen = evergreen_count(retry_kept, seed_themes)
            if (len(retry_kept), retry_evergreen) > (len(kept), evergreen):
                kept, dropped = retry_kept, retry_dropped
                briefs = retry_briefs
                evergreen = retry_evergreen

    # The copy mode is a batch decision too: a batch with no typographic hero
    # gets one of its shortest captions promoted, and hero briefs are capped at
    # half. No-text is never forced (issue #258).
    kept = assign_batch_copy_modes(kept)
    # The geometry is rotated by the batch, not picked by the model: a batch
    # that the model answered with text_top for the eighth time still gets
    # eight different silhouettes (issue #238). This runs after the last
    # attempt, so a batch that was asked for again still carries templates.
    kept = assign_batch_templates(kept, batch_id)
    # The idea device is rotated too, so the batch is not eight still lifes
    # (issue #240).
    kept = assign_batch_devices(kept, batch_id)
    # Now that the templates and devices are known, the model decides how the
    # words live in each design. It sees the whole batch and plans the mix; the
    # code only checks the plan and renders it (issue #246).
    kept = plan_batch_text_integration(llm_completions, model, kept)

    if dropped:
        logger.info(
            "Dropped %d briefs over the event quota: %s",
            len(dropped),
            ", ".join(brief.concept for brief in dropped),
        )

    mix = classify_brief_mix(kept, seed_themes)
    logger.info("Design brief mix: %s", mix.summary())

    created = 0
    for brief in kept:
        brief_id = db_client.insert_design_brief(
            concept=brief.concept,
            product_type=brief.product_type,
            specific_products=brief.specific_products,
            audience=brief.audience,
            visual_style=brief.visual_style,
            confidence=brief.confidence,
            reasoning=brief.reasoning,
            llm_model=model,
            batch_id=batch_id,
            layout_type=brief.layout_type,
            headline_text=brief.headline_text,
            tagline_text=brief.tagline_text,
            copy_mode=brief.copy_mode,
            kicker_text=brief.kicker_text,
            font_color=brief.font_color,
            scene_description=brief.scene_description,
            recommended_materials=brief.recommended_materials,
            voice=brief.voice,
            font_family=brief.font_family,
            style_preset=brief.style_preset,
            text_plan=brief.text_plan.to_dict() if brief.text_plan else None,
        )
        linked = 0
        for seed_id in brief.source_seed_ids:
            if seed_id in valid_ids:
                db_client.insert_brief_source(brief_id=brief_id, active_seed_id=seed_id)
                linked += 1
        if not linked:
            logger.warning(
                "Design brief '%s' has no usable source seed (asked for %s)",
                brief.concept,
                brief.source_seed_ids,
            )
        created += 1

    logger.info("Design brief generation: created=%d from %d candidates", created, len(candidates))

    return {
        "status": "success",
        "briefs_created": created,
        "briefable": len(candidates),
        "seasonal": mix.seasonal,
        "evergreen": mix.evergreen,
        "dropped": len(dropped),
    }


def assign_batch_templates(briefs: Sequence[DesignBrief], batch_id: str) -> list[DesignBrief]:
    """The briefs with a template assigned by the batch.

    The model decides whether a brief has copy, no copy at all, or only copy.
    The geometry is rotated here, so a batch always covers the templates
    (issue #238).
    """
    return [
        replace(brief, layout_type=template_id)
        for brief, template_id in zip(briefs, assign_templates(briefs, batch_id), strict=True)
    ]


def assign_batch_devices(briefs: Sequence[DesignBrief], batch_id: str) -> list[DesignBrief]:
    """The briefs with an idea device assigned by the batch.

    The model writes each brief against a device slot, and the slot is what
    the scene was written for. The slot wins here so a batch covers the
    vocabulary even when the model repeated itself (issue #240).
    """
    return [
        replace(brief, idea_device=device)
        for brief, device in zip(briefs, plan_device_slots(len(briefs), batch_id), strict=True)
    ]


def _plan_entry_indexes(briefs: Sequence[DesignBrief]) -> list[int]:
    """The positions of the briefs the text planner sees.

    A full-bleed brief has no words, so it has no text plan; the other briefs
    keep their order, which is how the 1-based answer is matched back. A
    typographic hero draws its own words, so it is not sent either: a carrier
    written into its scene would be drawn by Qwen and break the emblem.
    """
    return [
        index
        for index, brief in enumerate(briefs)
        if brief.copy_mode != TYPE_HERO
        and brief.layout_type not in HERO_LAYOUT_TYPES
        and resolve_template(brief.layout_type, brief.headline_text, brief.tagline_text)
        != "full_bleed"
    ]


def plan_batch_text_integration(
    llm_completions: Any,
    model: str,
    briefs: Sequence[DesignBrief],
    extra_instructions: Sequence[str] | None = None,
) -> list[DesignBrief]:
    """The briefs with the model's text-integration plan attached.

    This is a second pass, after the batch has its templates and idea devices.
    It sees the template each brief will be drawn on, so the carrier it writes
    into the scene is in the place the design expects the words. One call per
    batch; a failed call leaves the briefs on the existing treatment.
    """
    result = list(briefs)
    if not _text_plan_enabled():
        return result

    indexes = _plan_entry_indexes(result)
    if not indexes:
        return result
    entries = build_text_plan_entries([result[index] for index in indexes])
    if not entries:
        return result

    planned = _request_text_plans(llm_completions, model, entries, extra_instructions)
    if planned is None:
        return result

    planned_briefs = _apply_text_plans(result, indexes, planned)
    mix_issues = check_text_plan_mix([entry.plan for entry in planned.values()])
    fit_issues = check_text_plan_fit(entries, planned)
    issues = fit_issues + mix_issues
    if issues:
        logger.info("The text plan needs another pass: %s", "; ".join(issues))
        if _text_plan_regenerate_once():
            feedback = (
                "The previous plan did not pass the checks: "
                + "; ".join(issues)
                + ". Fix every issue and keep the concept fit. When the batch has four "
                "or more briefs, use at least four methods, no method more than twice, "
                "at most one third carriers, and at least half with no carrier."
            )
            retry = _request_text_plans(llm_completions, model, entries, [feedback])
            if retry is not None:
                # A retry may answer for only some briefs; keep the first plan
                # for the ones it skipped.
                merged = {**planned, **retry}
                retry_fit = check_text_plan_fit(entries, merged)
                retry_mix = check_text_plan_mix([entry.plan for entry in merged.values()])
                if (len(retry_fit), len(retry_mix)) < (len(fit_issues), len(mix_issues)):
                    logger.info("The second text-plan pass improved the plan")
                    planned_briefs = _apply_text_plans(result, indexes, merged)
                else:
                    logger.warning(
                        "The second text-plan pass did not improve the plan: %s",
                        "; ".join(retry_fit + retry_mix),
                    )
    return planned_briefs


def _request_text_plans(
    llm_completions: Any,
    model: str,
    entries: list[dict[str, Any]],
    extra_instructions: Sequence[str] | None = None,
) -> dict[int, PlannedText] | None:
    """The text plans in the model's answer, or None when there are none.

    A planner that fails must not lose a batch: the briefs already exist and
    keep the treatment they had before this pass.
    """
    messages = build_text_plan_prompt(entries, extra_instructions)
    raw, error = create_answer(llm_completions, model, messages, "text_plans", temperature=0)
    if error is not None:
        logger.warning("The text-plan pass failed: %s", error)
        return None

    try:
        response = raw.parse()
        if not response.choices:
            logger.warning("The text-plan pass returned empty choices")
            return None
        content = response.choices[0].message.content or ""
    except Exception as e:  # noqa: BLE001
        logger.warning("The text-plan answer could not be read: %s", e)
        return None

    plans, parse_error = read_text_plan_response(content, len(entries))
    if parse_error is not None:
        logger.warning("The text-plan answer was unusable: %s", parse_error)
        logger.warning("Raw text-plan answer: %s", content)
        return None
    return plans


def _sanitise_text_plan(plan: TextPlan, product_type: str) -> TextPlan:
    """A plan that does not contradict the product or its own method.

    The planner is asked to fix these, but a batch must never render a
    garment mockup or a carved word with no surface. The fallback is the
    model's own non-carrier alternative, then painted; the brief keeps its
    words either way (issue #246).
    """
    if plan.needs_carrier and plan.surface == "none":
        plan = replace(
            plan,
            surface=infer_surface(plan.carrier, plan.carrier_description),
        )
    if product_type == "dtf_apparel" and plan.carrier == "apparel":
        return replace(
            plan,
            method=plan.non_carrier_method or "painted",
            carrier="none",
            carrier_description=None,
            region="any",
            surface="none",
        )
    if plan.method in SURFACE_METHODS and plan.carrier == "none":
        return replace(
            plan,
            method=plan.non_carrier_method or "painted",
            carrier="none",
            carrier_description=None,
            region="any",
            surface="none",
        )
    if plan.method == "typographic" and plan.carrier != "none":
        return replace(
            plan,
            carrier="none",
            carrier_description=None,
            region="any",
            surface="none",
        )
    return plan


def _apply_text_plans(
    briefs: list[DesignBrief],
    indexes: Sequence[int],
    plans: dict[int, PlannedText],
) -> list[DesignBrief]:
    """The briefs with the plans and revised carrier scenes applied."""
    result = list(briefs)
    for number, planned in plans.items():
        if number < 1 or number > len(indexes):
            continue
        index = indexes[number - 1]
        brief = result[index]
        plan = _sanitise_text_plan(planned.plan, brief.product_type)
        scene = brief.scene_description
        if planned.scene_description:
            scene = planned.scene_description
        if plan != planned.plan:
            logger.warning(
                "Adjusted an unusable text plan for brief %r: %s/%s -> %s/%s",
                brief.concept,
                planned.plan.method,
                planned.plan.carrier,
                plan.method,
                plan.carrier,
            )
            if not plan.needs_carrier:
                scene = brief.scene_description
        if plan.carrier in ATTACHABLE_CARRIERS and not carrier_is_attached(plan, scene):
            logger.warning(
                "Dropping an unattached %s carrier for brief %r; the words fall "
                "back to %s in the template zone",
                plan.carrier,
                brief.concept,
                plan.non_carrier_method or "painted",
            )
            plan = replace(
                plan,
                method=plan.non_carrier_method or "painted",
                carrier="none",
                carrier_description=None,
                region="any",
                surface="none",
            )
            scene = brief.scene_description
        result[index] = replace(
            brief,
            text_plan=plan,
            scene_description=scene,
        )
    return result


class TextPlanDBClient(Protocol):
    def list_briefs_by_batch(self, batch_id: str) -> list[dict[str, Any]]: ...

    def update_brief_text_plan(
        self,
        brief_id: int,
        text_plan: dict[str, Any] | None,
        scene_description: str | None,
    ) -> None: ...


def _brief_from_row(row: dict[str, Any]) -> DesignBrief:
    """A brief from a stored row, for a replan pass."""
    copy_mode = copy_mode_from_row(row)
    layout_type = str(row.get("layout_type") or "full_bleed")
    headline_text = row.get("headline_text")
    tagline_text = row.get("tagline_text")
    kicker_text = row.get("kicker_text") if copy_mode == TYPE_HERO else None
    if copy_mode == NO_TEXT:
        layout_type = "full_bleed"
        headline_text = None
        tagline_text = None
        kicker_text = None
    elif (copy_mode == TYPE_HERO and layout_type in ("full_bleed", "text_only")) or (
        copy_mode == CAPTION
        and layout_type == "full_bleed"
        and (headline_text or tagline_text or row.get("text_plan"))
    ):
        layout_type = "auto"
    return DesignBrief(
        concept=str(row.get("concept") or ""),
        product_type=str(row.get("product_type") or "dtf_apparel"),
        specific_products=[],
        audience=str(row.get("audience") or ""),
        visual_style=str(row.get("visual_style") or ""),
        confidence=0,
        reasoning="",
        source_seed_ids=[],
        layout_type=layout_type,
        headline_text=headline_text,
        tagline_text=tagline_text,
        copy_mode=copy_mode,
        kicker_text=kicker_text,
        font_color=row.get("font_color"),
        regeneration_feedback=row.get("regeneration_feedback"),
        scene_description=row.get("scene_description"),
        recommended_materials=row.get("recommended_materials"),
        voice=str(row.get("voice") or DEFAULT_VOICE),
        font_family=row.get("font_family"),
        style_preset=row.get("style_preset"),
        text_plan=text_plan_from_value(row.get("text_plan")),
    )


def replan_batch(
    llm_completions: Any,
    model: str,
    db_client: TextPlanDBClient,
    batch_id: str,
    extra_instructions: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Rewrite the text plans of an existing batch and store the changes.

    The image renderer only executes a stored plan, so a batch that was
    written before the plan existed (or that got a poor plan) needs this pass
    before its images can be re-rendered. Briefs the planner cannot improve
    keep their current plan.
    """
    rows = db_client.list_briefs_by_batch(batch_id)
    if not rows:
        return {"status": "skipped", "batch_id": batch_id, "reason": "no briefs in batch"}

    briefs = [_brief_from_row(row) for row in rows]
    planned = plan_batch_text_integration(llm_completions, model, briefs, extra_instructions)
    updated = 0
    for row, before, after in zip(rows, briefs, planned, strict=True):
        if after.text_plan is None:
            continue
        if (
            after.text_plan == before.text_plan
            and after.scene_description == before.scene_description
        ):
            continue
        db_client.update_brief_text_plan(
            row["id"],
            after.text_plan.to_dict(),
            after.scene_description,
        )
        updated += 1
    return {
        "status": "success",
        "batch_id": batch_id,
        "briefs": len(briefs),
        "updated": updated,
    }


def _merge_briefs(first: list[DesignBrief], second: list[DesignBrief]) -> list[DesignBrief]:
    """Two batches of briefs as one, without the repeats.

    The second attempt answers for the same seeds, so it can repeat a brief
    the first answer already gave.
    """
    merged = list(first)
    seen = {(brief.concept, brief.product_type) for brief in first}
    for brief in second:
        key = (brief.concept, brief.product_type)
        if key in seen:
            continue
        seen.add(key)
        merged.append(brief)
    return merged


def _request_briefs_with_retry(
    llm_completions: Any,
    model: str,
    messages: list[dict[str, str]],
    enabled: Sequence[str] | None = None,
    materials: list[dict[str, Any]] | None = None,
    rejections: list[str] | None = None,
) -> tuple[list[DesignBrief] | None, str | None]:
    """Ask for a batch of briefs, and ask again when the answer was unreadable.

    A parse failure used to be reported as a successful batch of zero briefs,
    so the whole run was lost without a word in the logs (issue #209). The
    model gets one more chance, and if the second answer is unreadable too
    the run fails.
    """
    briefs: list[DesignBrief] | None = None
    error: str | None = None
    for attempt in (1, 2):
        try:
            briefs, error = _request_briefs(
                llm_completions, model, messages, enabled, materials, rejections
            )
            break
        except LLMResponseParseError as e:
            error = str(e)
            if attempt == 1:
                logger.info("Asking for the briefs again after an unreadable answer: %s", e)
    return briefs, error


def _request_briefs(
    llm_completions: Any,
    model: str,
    messages: list[dict[str, str]],
    enabled: Sequence[str] | None = None,
    materials: list[dict[str, Any]] | None = None,
    rejections: list[str] | None = None,
) -> tuple[list[DesignBrief] | None, str | None]:
    """Ask the model for a batch of briefs: (briefs, error).

    An answer that cannot be read raises `LLMResponseParseError`: the model
    answered, but the answer was not JSON, and losing the batch silently is
    what the parse failure looked like (issue #209).
    """
    raw, error = create_answer(llm_completions, model, messages, "briefs")
    if error is not None:
        return None, error

    try:
        headers = raw.headers
        response = raw.parse()
        _log_llm_response(response, headers, "briefs")
        if not response.choices:
            logger.warning(
                "LLM brief generation returned empty choices: model=%s, id=%s, body=%s",
                getattr(response, "model", None),
                getattr(response, "id", None),
                getattr(raw, "text", None),
            )
            return None, "LLM returned empty choices"
        content = response.choices[0].message.content or ""
    except Exception as e:  # noqa: BLE001
        message = model_error_message(e, model)
        if is_model_missing_error(e):
            logger.error("LLM brief generation failed: %s", message)
        else:
            logger.warning("LLM brief generation call failed: %s", e)
        return None, message

    briefs, parse_error = read_brief_response(content, enabled, materials, rejections)
    if parse_error is not None:
        logger.error("The LLM brief response could not be read: %s", parse_error)
        # The raw answer is the only way to see what the model actually sent.
        logger.error("Raw LLM brief response: %s", content)
        raise LLMResponseParseError(parse_error)

    return briefs, None
