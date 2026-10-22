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
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from typing import Any, Protocol

from store_listing.orchestration.brief_mix import (
    BRIEF_MIX_REGENERATE_ONCE,
    MAX_BRIEFS_PER_THEME,
    MIN_EVERGREEN_BRIEFS,
    classify_brief_mix,
    enforce_brief_mix,
    event_theme_for_query,
    evergreen_count,
)
from store_listing.orchestration.brief_prompts import (
    DESIGN_ONLY_RULES,
    DIVERSITY_RULES,
    layout_rules,
    product_belongs_to,
    product_catalog,
    voice_rules,
)
from store_listing.orchestration.context_seeds import get_upcoming_events
from store_listing.orchestration.headline_voice import DEFAULT_VOICE, normalise_voice, screen_text
from store_listing.orchestration.json_response import (
    LLMResponseParseError,
    extract_json_objects,
    merge_items,
)
from store_listing.orchestration.llm_client import is_model_missing_error, model_error_message
from store_listing.orchestration.promotion import ORIGIN_TREND
from store_listing.orchestration.scene_text import strip_lettering
from store_listing.orchestration.seed_curation import PRODUCT_TAGS, _log_llm_response
from store_listing.orchestration.typography import family_for_voice

logger = logging.getLogger(__name__)

MAX_SEEDS_PER_BRIEF = int(os.environ.get("LLM_MAX_BRIEF_SEEDS", "20"))
MAX_BRIEFS = int(os.environ.get("LLM_MAX_BRIEFS", "8"))
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
        "- visual_style: art direction notes (color palette, illustration style, "
        "typography). Do NOT include any text/slogan in this field — text is rendered "
        "separately via the layout fields below.\n"
        "- confidence: 0-100, how confident you are this will sell\n"
        "- reasoning: why this concept is timely or evergreen and marketable (cite the "
        "trend signals)\n"
        "- source_seed_ids: array of seed IDs from the input that inspired this brief\n\n"
        f"{layout_rules(types)}\n"
        f"{voice_rules()}\n"
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
        '"layout_type": "text_top", "headline_text": "...", '
        '"tagline_text": "...", "font_color": "#000000", '
        '"voice": "deadpan | loud | cozy", '
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


VALID_LAYOUT_TYPES: set[str] = {"full_bleed", "text_top", "text_top_bottom", "text_only"}

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
    font_color: str | None = None
    regeneration_feedback: str | None = None
    scene_description: str | None = None
    recommended_materials: list[dict[str, Any]] | None = None
    # The voice of the text of the design: 'deadpan', 'loud' or 'cozy'.
    voice: str = DEFAULT_VOICE
    # The family the text of the design is drawn in.
    font_family: str | None = None


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
        font_color: str | None = None,
        scene_description: str | None = None,
        personal_use: bool = False,
        recommended_materials: list[dict[str, Any]] | None = None,
        voice: str = DEFAULT_VOICE,
        font_family: str | None = None,
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
    # The family of the type is decided here, not by the model: the model picks
    # a tone, and the type follows the art direction (issue #226).
    font_family = family_for_voice(voice)

    # The text of a design is the part the buyer reads, so it is the part that
    # has to be good. A headline that names a brand, that lists three nouns or
    # that only restates the picture is dropped and the batch asks again
    # (issue #225).
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
        font_color=font_color,
        scene_description=scene_description,
        recommended_materials=recommended_materials,
        voice=voice,
        font_family=font_family,
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
    if material_type:
        materials = db_client.list_active_materials(product_type=material_type)
    else:
        materials = db_client.list_active_materials()
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
            font_color=brief.font_color,
            scene_description=brief.scene_description,
            recommended_materials=brief.recommended_materials,
            voice=brief.voice,
            font_family=brief.font_family,
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
    try:
        raw = llm_completions.with_raw_response.create(model=model, messages=messages)
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
