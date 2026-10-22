"""LLM-driven seed curation: evaluate candidates for design readiness.

Per-candidate evaluation with structured reasoning. Each candidate gets
a PROMOTE / PIVOT / REJECT decision with brief reasoning, product tags,
and design style suggestions.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from typing import Any, Final, Protocol

from store_listing.orchestration.context_seeds import SEASONAL_EVENT_THEME, get_upcoming_events
from store_listing.orchestration.json_response import extract_json_objects, merge_items
from store_listing.orchestration.llm_client import (
    LLM_MAX_TOKENS,
    is_model_missing_error,
    model_error_message,
)
from store_listing.orchestration.promotion import (
    DISCOVERY_SOURCE,
    MAX_ACTIVE_SEEDS,
    MIN_BRIEFABLE_SEEDS,
    ORIGIN_DISCOVERY,
    ORIGIN_EVENT,
    ORIGIN_PIVOT,
    ORIGIN_TREND,
    ActiveSeed,
    SeedCandidate,
    enforce_cap,
)

logger = logging.getLogger(__name__)

PRODUCT_TAGS: Final[set[str]] = {"dtf_apparel", "sublimation", "sticker_vinyl"}
VALID_ACTIONS: Final[set[str]] = {"PROMOTE", "PIVOT", "REJECT"}


def _log_llm_response(response: Any, headers: Any, label: str) -> None:
    usage = getattr(response, "usage", None)
    tokens_in = getattr(usage, "prompt_tokens", None) if usage else None
    tokens_out = getattr(usage, "completion_tokens", None) if usage else None
    model = getattr(response, "model", None)
    # The reason the answer ended. An answer that was cut off says "length",
    # and without it a cut answer and a broken answer look the same
    # (issue #232).
    choices = getattr(response, "choices", None)
    finish_reason = getattr(choices[0], "finish_reason", None) if choices else None
    limit = headers.get("x-ratelimit-limit") if headers else None
    remaining = headers.get("x-ratelimit-remaining") if headers else None
    reset = headers.get("x-ratelimit-reset") if headers else None
    logger.info(
        "LLM %s: model=%s, tokens_in=%s, tokens_out=%s, finish=%s, ratelimit=%s/%s, reset=%s",
        label,
        model,
        tokens_in,
        tokens_out,
        finish_reason,
        remaining,
        limit,
        reset,
    )


SYSTEM_PROMPT = (
    "You are an expert e-commerce product strategist and apparel graphic designer "
    "evaluating trending search terms.\n\n"
    "Your task is to analyze incoming trend candidates and classify each into a "
    "merchandise decision:\n"
    '- "PROMOTE": Strong visual hook, clear target audience, unique phrase or meme, '
    "low copyright risk. A designer could immediately create artwork for this.\n"
    '- "PIVOT": The underlying trend is good, but the raw text needs a creative '
    "angle or rephrasing to work visually or avoid copyright. Suggest a specific "
    "pivot seed.\n"
    '- "REJECT": Visually untranslatable, generic category, noise, tutorial/how-to, '
    "or direct trademark/IP infringement.\n\n"
    "We produce designs for these products ONLY:\n"
    "- DTF apparel: t-shirts, hoodies, sweatshirts, tank tops, hats, tote bags, "
    "baby onesies\n"
    "- Sublimation: mugs, tumblers, phone cases, mouse pads, coasters, puzzles, "
    "ornaments, wall art, pillows, blankets, socks\n"
    "- Stickers and vinyl decals\n\n"
    'Valid product_tags: "dtf_apparel", "sublimation", "sticker_vinyl"\n\n'
    "RULES:\n"
    "1. Keep reasoning to ONE short sentence per candidate (under 20 words).\n"
    "2. For PROMOTE, assign product_tags and suggest a design style.\n"
    "3. For PIVOT, provide a specific pivot seed and product_tags.\n"
    "4. Flag trademark/copyright risks.\n"
    "5. Respond ONLY with a valid JSON object. No text before or after the JSON."
)

MAX_CANDIDATES = int(os.environ.get("LLM_MAX_CANDIDATES", "10"))


@dataclass(frozen=True)
class CurationEvaluation:
    id: int
    action: str
    reasoning: str = ""
    product_tags: list[str] = field(default_factory=list)
    design_style: str | None = None
    pivot_to: str | None = None


@dataclass(frozen=True)
class EventSeed:
    seed: str
    product_tags: list[str] = field(default_factory=list)
    design_style: str | None = None


@dataclass
class CurationResult:
    evaluations: list[CurationEvaluation] = field(default_factory=list)
    event_seeds: list[EventSeed] = field(default_factory=list)


class CurationDBClient(Protocol):
    def list_pending_candidates(self) -> list[SeedCandidate]: ...

    def list_active_seeds(self) -> list[ActiveSeed]: ...

    def list_briefable_seed_ids(self) -> list[int]: ...

    def cross_seed_counts(self) -> dict[str, int]: ...

    def insert_active_seed(
        self,
        query: str,
        promotion_score: int,
        candidate_id: int | None,
        origin: str = ...,
        theme: str | None = ...,
        seasonal_window_end: date | None = ...,
    ) -> int | None: ...

    def promote_candidate(self, candidate_id: int) -> int: ...

    def reject_candidate(self, candidate_id: int) -> int: ...

    def archive_active_seed(self, query: str) -> int: ...

    def insert_curation_log(
        self,
        candidate_id: int | None,
        action: str,
        reasoning: str | None,
        event_context: str | None,
        llm_model: str | None,
    ) -> int: ...

    def insert_product_tags(self, active_seed_id: int, tags: list[str]) -> int: ...


def build_curation_prompt(
    candidates: list[SeedCandidate],
    active_seeds: list[ActiveSeed],
    cross_seed_counts: dict[str, int],
    context: list[dict[str, Any]],
    today: date,
) -> list[dict[str, str]]:
    trends = [
        {
            "id": c.id,
            "term": c.query,
            "source": c.source,
            "score": c.score,
            "delta": c.delta,
        }
        for c in candidates[:MAX_CANDIDATES]
    ]

    events_str = ", ".join(f"{e['name']} ({e['days_until']}d)" for e in context) or "none"

    seeds_str = ", ".join(s.query for s in active_seeds[:30]) or "none"

    user_content = (
        "Analyze the following trend batch and output your response in JSON format.\n\n"
        f"[CONTEXT]\nDate: {today.isoformat()}\n"
        f"Upcoming events: {events_str}\n"
        f"Current active seeds: {seeds_str}\n\n"
        "[INPUT DATA]\n" + json.dumps({"trends": trends}, indent=2) + "\n\n"
        "[OUTPUT FORMAT]\n"
        "Return a JSON object structured exactly as follows:\n"
        "{\n"
        '  "evaluations": [\n'
        "    {\n"
        '      "id": 123,\n'
        '      "reasoning": "brief evaluation",\n'
        '      "action": "PROMOTE | PIVOT | REJECT",\n'
        '      "product_tags": ["dtf_apparel"],\n'
        '      "design_style": "string or null",\n'
        '      "pivot_to": "string or null"\n'
        "    }\n"
        "  ],\n"
        '  "event_seeds": [\n'
        '    {"seed": "keyword", "product_tags": ["dtf_apparel"], '
        '"design_style": "style"}\n'
        "  ]\n"
        "}"
    )

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def _filter_tags(raw: Any) -> list[str]:
    if not isinstance(raw, list):
        return []
    return [t for t in raw if isinstance(t, str) and t in PRODUCT_TAGS]


def read_curation_response(text: str) -> tuple[CurationResult, str | None]:
    """The decisions in the answer, and the reason the answer could not be read.

    The model does not always answer with only JSON: it appends a note,
    repeats the answer, or splits it in two. The answer is read value by
    value and the decisions are merged, because a parser that read two values
    as one threw the whole answer away (issue #209).
    """
    objects = extract_json_objects(text)
    if not objects:
        return CurationResult(), "no JSON object in the LLM response"

    if not any(
        isinstance(obj.get("evaluations"), list) or isinstance(obj.get("event_seeds"), list)
        for obj in objects
    ):
        return CurationResult(), "the LLM response has no 'evaluations' array"

    evaluations: list[CurationEvaluation] = []
    seen_ids: set[int] = set()
    for ev in merge_items(objects, "evaluations"):
        if not isinstance(ev, dict):
            continue
        cid = ev.get("id")
        action = ev.get("action", "").upper()
        if not isinstance(cid, int) or action not in VALID_ACTIONS:
            continue
        # A repeated answer must not decide the same candidate twice.
        if cid in seen_ids:
            continue
        seen_ids.add(cid)
        evaluations.append(
            CurationEvaluation(
                id=cid,
                action=action,
                reasoning=str(ev.get("reasoning", "")),
                product_tags=_filter_tags(ev.get("product_tags", [])),
                design_style=ev.get("design_style"),
                pivot_to=ev.get("pivot_to"),
            )
        )

    event_seeds: list[EventSeed] = []
    seen_seeds: set[str] = set()
    for e in merge_items(objects, "event_seeds"):
        if not isinstance(e, dict):
            continue
        seed = e.get("seed", "")
        if not seed or seed in seen_seeds:
            continue
        seen_seeds.add(seed)
        event_seeds.append(
            EventSeed(
                seed=seed,
                product_tags=_filter_tags(e.get("product_tags", [])),
                design_style=e.get("design_style"),
            )
        )

    return CurationResult(evaluations=evaluations, event_seeds=event_seeds), None


def parse_curation_response(text: str) -> CurationResult:
    """The decisions in the answer, or an empty result when it cannot be read."""
    result, _ = read_curation_response(text)
    return result


def curate_seeds(
    db_client: CurationDBClient,
    llm_completions: Any,
    model: str,
    today: date | None = None,
) -> dict[str, Any]:
    today = today or datetime.now(tz=UTC).date()

    all_candidates = db_client.list_pending_candidates()
    if not all_candidates:
        return {
            "status": "success",
            "promoted": 0,
            "rejected": 0,
            "pivots": 0,
            "event_seeds": 0,
            "remaining": 0,
            "briefable": len(db_client.list_briefable_seed_ids()),
        }

    candidates = all_candidates[:MAX_CANDIDATES]
    remaining = max(0, len(all_candidates) - len(candidates))

    active_seeds = db_client.list_active_seeds()
    cross_counts = db_client.cross_seed_counts()
    context = get_upcoming_events(today)

    messages = build_curation_prompt(candidates, active_seeds, cross_counts, context, today)

    try:
        raw = llm_completions.with_raw_response.create(
            model=model,
            messages=messages,
            temperature=0,
            max_tokens=LLM_MAX_TOKENS,
        )
        headers = raw.headers
        response = raw.parse()
        _log_llm_response(response, headers, "curation")
        if not response.choices:
            logger.warning(
                "LLM curation returned empty choices: model=%s, id=%s, body=%s",
                getattr(response, "model", None),
                getattr(response, "id", None),
                getattr(raw, "text", None),
            )
            return {"status": "error", "error": "LLM returned empty choices"}
        content = response.choices[0].message.content or ""
    except Exception as e:  # noqa: BLE001
        error_str = str(e)
        if "429" in error_str:
            logger.warning("LLM curation rate-limited: %s", e)
            return {"status": "rate_limited", "error": error_str}
        message = model_error_message(e, model)
        if is_model_missing_error(e):
            logger.error("LLM curation call failed: %s", message)
        else:
            logger.warning("LLM curation call failed: %s", e)
        return {"status": "error", "error": message}

    result, parse_error = read_curation_response(content)
    if parse_error is not None:
        # The raw answer is the only way to see what the model actually sent.
        logger.error("The LLM curation response could not be read: %s", parse_error)
        logger.error("Raw LLM curation response: %s", content)
        return {"status": "error", "error": parse_error}

    by_id = {c.id: c for c in candidates}

    promoted = 0
    rejected = 0
    pivot_count = 0

    for ev in result.evaluations:
        if ev.id not in by_id:
            continue
        candidate = by_id[ev.id]

        if ev.action == "PROMOTE":
            seed_id = db_client.insert_active_seed(
                query=candidate.query,
                promotion_score=candidate.promotion_score,
                candidate_id=ev.id,
                origin=(ORIGIN_DISCOVERY if candidate.source == DISCOVERY_SOURCE else ORIGIN_TREND),
            )
            db_client.promote_candidate(ev.id)
            db_client.insert_curation_log(
                candidate_id=ev.id,
                action="promote",
                reasoning=ev.reasoning,
                event_context=None,
                llm_model=model,
            )
            if seed_id is not None and ev.product_tags:
                db_client.insert_product_tags(active_seed_id=seed_id, tags=ev.product_tags)
            promoted += 1

        elif ev.action == "PIVOT":
            pivot_seed = ev.pivot_to or candidate.query
            db_client.reject_candidate(ev.id)
            inserted = db_client.insert_active_seed(
                query=pivot_seed,
                promotion_score=0,
                candidate_id=None,
                origin=ORIGIN_PIVOT,
            )
            db_client.insert_curation_log(
                candidate_id=ev.id,
                action="pivot",
                reasoning=ev.reasoning,
                event_context=None,
                llm_model=model,
            )
            if inserted is not None and ev.product_tags:
                db_client.insert_product_tags(active_seed_id=inserted, tags=ev.product_tags)
            pivot_count += 1

        elif ev.action == "REJECT":
            db_client.reject_candidate(ev.id)
            db_client.insert_curation_log(
                candidate_id=ev.id,
                action="reject",
                reasoning=ev.reasoning,
                event_context=None,
                llm_model=model,
            )
            rejected += 1

    event_count = 0
    for es in result.event_seeds:
        inserted = db_client.insert_active_seed(
            query=es.seed,
            promotion_score=0,
            candidate_id=None,
            origin=ORIGIN_EVENT,
            theme=SEASONAL_EVENT_THEME,
        )
        if inserted is not None:
            db_client.insert_curation_log(
                candidate_id=None,
                action="event_inject",
                reasoning=None,
                event_context=es.seed,
                llm_model=model,
            )
            if es.product_tags:
                db_client.insert_product_tags(active_seed_id=inserted, tags=es.product_tags)
            event_count += 1

    incoming = promoted + pivot_count + event_count
    briefable_ids = db_client.list_briefable_seed_ids()
    if incoming > 0:
        active_seeds = db_client.list_active_seeds()
        archives = enforce_cap(
            active_seeds,
            0,
            max_active=MAX_ACTIVE_SEEDS,
            briefable_ids=set(briefable_ids),
            reserve=MIN_BRIEFABLE_SEEDS,
        )
        for query in archives:
            db_client.archive_active_seed(query)

    logger.info(
        "LLM curation: promoted=%d rejected=%d pivots=%d events=%d remaining=%d "
        "briefable=%d reserve=%d",
        promoted,
        rejected,
        pivot_count,
        event_count,
        remaining,
        len(briefable_ids),
        MIN_BRIEFABLE_SEEDS,
    )

    return {
        "status": "success",
        "promoted": promoted,
        "rejected": rejected,
        "pivots": pivot_count,
        "event_seeds": event_count,
        "remaining": remaining,
        "briefable": len(briefable_ids),
    }
