"""LLM-driven seed curation: evaluate candidates for design readiness.

Replaces pure algorithmic promotion with semantic judgment from an
OpenAI-compatible LLM (Groq free tier now, local vLLM later).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Any, Protocol

from store_listing.orchestration.context_seeds import get_upcoming_events
from store_listing.orchestration.promotion import ActiveSeed, SeedCandidate, enforce_cap

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a print-on-demand product strategist. Your job is to curate seed "
    "keywords that will drive the next harvest cycle toward specific, design-ready "
    "product concepts.\n\n"
    "Promote seeds that a designer could immediately create artwork for. "
    "Reject generic categories and noise. Suggest pivots that narrow broad seeds "
    "into specific niches with buyer appeal.\n\n"
    "Also suggest event_seeds — new seeds inspired by the current context "
    "(holidays, weather, sports, trending events) that no harvester would discover "
    "on its own. These should be specific, design-ready concepts tied to the moment.\n\n"
    "IMPORTANT: Do not suggest seeds that would require licensed IP (movie characters, "
    "team logos, brand names). Suggest concepts inspired by the cultural moment, "
    "not the IP itself.\n\n"
    "Return ONLY valid JSON with this exact structure:\n"
    '{"promote": [candidate_ids], "reject": [candidate_ids], '
    '"pivot": [{"from": "broad seed", "to": "specific seed", "reason": "why"}], '
    '"event_seeds": [{"seed": "keyword", "reason": "why", "event": "event_name"}], '
    '"reasoning": {"candidate_id": "explanation"}}'
)

MAX_CANDIDATES = 50


@dataclass(frozen=True)
class PivotSuggestion:
    from_seed: str
    to_seed: str
    reason: str


@dataclass(frozen=True)
class EventSeed:
    seed: str
    reason: str
    event: str


@dataclass
class CurationResult:
    promote: list[int] = field(default_factory=list)
    reject: list[int] = field(default_factory=list)
    pivot: list[PivotSuggestion] = field(default_factory=list)
    event_seeds: list[EventSeed] = field(default_factory=list)
    reasoning: dict[str, str] = field(default_factory=dict)


class CurationDBClient(Protocol):
    def list_pending_candidates(self) -> list[SeedCandidate]: ...

    def list_active_seeds(self) -> list[ActiveSeed]: ...

    def cross_seed_counts(self) -> dict[str, int]: ...

    def insert_active_seed(
        self, query: str, promotion_score: int, candidate_id: int | None
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


def build_curation_prompt(
    candidates: list[SeedCandidate],
    active_seeds: list[ActiveSeed],
    cross_seed_counts: dict[str, int],
    context: list[dict[str, Any]],
    today: date,
) -> list[dict[str, str]]:
    candidate_data = [
        {
            "id": c.id,
            "query": c.query,
            "source": c.source,
            "query_type": c.query_type,
            "score": c.score,
            "delta": c.delta,
            "promotion_score": c.promotion_score,
            "cross_source_count": cross_seed_counts.get(c.query, 0),
        }
        for c in candidates[:MAX_CANDIDATES]
    ]

    user_content = json.dumps(
        {
            "date": today.isoformat(),
            "upcoming_events": context,
            "current_seeds": [s.query for s in active_seeds],
            "candidates": candidate_data,
        },
        indent=2,
    )

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def parse_curation_response(text: str) -> CurationResult:
    try:
        start = text.find("{")
        end = text.rfind("}") + 1
        if start < 0 or end <= start:
            logger.warning("No JSON object found in LLM response")
            return CurationResult()
        data = json.loads(text[start:end])
    except json.JSONDecodeError as e:
        logger.warning("Failed to parse LLM curation response: %s", e)
        return CurationResult()

    pivots = [
        PivotSuggestion(
            from_seed=p.get("from", ""),
            to_seed=p.get("to", ""),
            reason=p.get("reason", ""),
        )
        for p in data.get("pivot", [])
        if isinstance(p, dict) and p.get("to")
    ]
    event_seeds = [
        EventSeed(
            seed=e.get("seed", ""),
            reason=e.get("reason", ""),
            event=e.get("event", ""),
        )
        for e in data.get("event_seeds", [])
        if isinstance(e, dict) and e.get("seed")
    ]
    reasoning = data.get("reasoning", {})
    if not isinstance(reasoning, dict):
        reasoning = {}

    return CurationResult(
        promote=[i for i in data.get("promote", []) if isinstance(i, int)],
        reject=[i for i in data.get("reject", []) if isinstance(i, int)],
        pivot=pivots,
        event_seeds=event_seeds,
        reasoning={str(k): str(v) for k, v in reasoning.items()},
    )


def curate_seeds(
    db_client: CurationDBClient,
    llm_completions: Any,
    model: str,
    today: date | None = None,
) -> dict[str, Any]:
    today = today or datetime.now(tz=timezone.utc).date()

    candidates = db_client.list_pending_candidates()
    if not candidates:
        return {"status": "success", "promoted": 0, "rejected": 0, "pivots": 0, "event_seeds": 0}

    active_seeds = db_client.list_active_seeds()
    cross_counts = db_client.cross_seed_counts()
    context = get_upcoming_events(today)

    messages = build_curation_prompt(candidates, active_seeds, cross_counts, context, today)

    try:
        response = llm_completions.create(model=model, messages=messages)
        content = response.choices[0].message.content or ""
    except Exception as e:  # noqa: BLE001
        logger.warning("LLM curation call failed: %s", e)
        return {"status": "error", "error": str(e)}

    result = parse_curation_response(content)
    by_id = {c.id: c for c in candidates}

    promoted = 0
    for cid in result.promote:
        if cid not in by_id:
            continue
        candidate = by_id[cid]
        db_client.insert_active_seed(
            query=candidate.query,
            promotion_score=candidate.promotion_score,
            candidate_id=cid,
        )
        db_client.promote_candidate(cid)
        db_client.insert_curation_log(
            candidate_id=cid,
            action="promote",
            reasoning=result.reasoning.get(str(cid)),
            event_context=None,
            llm_model=model,
        )
        promoted += 1

    rejected = 0
    for cid in result.reject:
        if cid not in by_id:
            continue
        db_client.reject_candidate(cid)
        db_client.insert_curation_log(
            candidate_id=cid,
            action="reject",
            reasoning=result.reasoning.get(str(cid)),
            event_context=None,
            llm_model=model,
        )
        rejected += 1

    pivot_count = 0
    for pivot in result.pivot:
        inserted = db_client.insert_active_seed(
            query=pivot.to_seed, promotion_score=0, candidate_id=None
        )
        if inserted is not None:
            db_client.insert_curation_log(
                candidate_id=None,
                action="pivot",
                reasoning=f"{pivot.from_seed} -> {pivot.to_seed}: {pivot.reason}",
                event_context=None,
                llm_model=model,
            )
            pivot_count += 1

    event_count = 0
    for es in result.event_seeds:
        inserted = db_client.insert_active_seed(query=es.seed, promotion_score=0, candidate_id=None)
        if inserted is not None:
            db_client.insert_curation_log(
                candidate_id=None,
                action="event_inject",
                reasoning=es.reason,
                event_context=es.event,
                llm_model=model,
            )
            event_count += 1

    incoming = promoted + pivot_count + event_count
    if incoming > 0:
        active_seeds = db_client.list_active_seeds()
        archives = enforce_cap(active_seeds, 0)
        for query in archives:
            db_client.archive_active_seed(query)

    logger.info(
        "LLM curation: promoted=%d rejected=%d pivots=%d events=%d",
        promoted,
        rejected,
        pivot_count,
        event_count,
    )

    return {
        "status": "success",
        "promoted": promoted,
        "rejected": rejected,
        "pivots": pivot_count,
        "event_seeds": event_count,
    }
