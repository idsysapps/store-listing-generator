"""Periodic reranking of design briefs against current trend signals."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import date
from typing import Any, Final, Protocol

from store_listing.orchestration.seed_curation import _log_llm_response

logger = logging.getLogger(__name__)

SYSTEM_PROMPT: Final[str] = (
    "You are a print-on-demand trend analyst. You are re-evaluating existing design "
    "briefs against current trend data to determine if their confidence scores should "
    "change.\n\n"
    "For each brief, consider:\n"
    "- Are the trends that inspired it still rising, plateauing, or fading?\n"
    "- Is the timing still right (seasonal relevance, event proximity)?\n"
    "- Have new competing trends emerged that make this concept less unique?\n"
    "- Has search volume or social engagement changed?\n\n"
    "Return a JSON object with evaluations for EVERY brief provided:\n"
    '{"evaluations": [\n'
    "  {\n"
    '    "brief_id": 1,\n'
    '    "new_confidence": 0-100,\n'
    '    "direction": "up | down | hold",\n'
    '    "reasoning": "why the score changed or stayed"\n'
    "  }\n"
    "]}\n\n"
    "Guidelines:\n"
    "- 'up': trend momentum is stronger than when the brief was created\n"
    "- 'down': trend is fading, season passed, or competition increased\n"
    "- 'hold': trend is stable, confidence should stay roughly the same\n"
    "- Be conservative — only move scores significantly if the data clearly supports it\n"
    "- A brief past its seasonal window should drop sharply\n\n"
    "Respond ONLY with valid JSON."
)


@dataclass(frozen=True)
class RerankableBrief:
    id: int
    concept: str
    product_type: str
    audience: str
    confidence: int
    original_confidence: int
    reasoning: str
    batch_id: str


class RerankDBClient(Protocol):
    def list_briefs_for_reranking(self) -> list[RerankableBrief]: ...

    def list_top_active_seed_signals(self) -> list[dict[str, Any]]: ...

    def update_brief_confidence(
        self, *, brief_id: int, confidence: int, reasoning: str
    ) -> None: ...


def build_rerank_prompt(
    briefs: list[RerankableBrief],
    trends: list[dict[str, Any]],
    today: date,
) -> list[dict[str, str]]:
    brief_data = [
        {
            "brief_id": b.id,
            "concept": b.concept,
            "product_type": b.product_type,
            "audience": b.audience,
            "current_confidence": b.confidence,
            "original_confidence": b.original_confidence,
            "original_reasoning": b.reasoning,
            "created_batch": b.batch_id,
        }
        for b in briefs
    ]

    user_content = json.dumps(
        {
            "date": today.isoformat(),
            "briefs": brief_data,
            "current_trend_signals": trends,
        },
        indent=2,
    )

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


VALID_DIRECTIONS: set[str] = {"up", "down", "hold"}


def parse_rerank_response(text: str) -> list[dict[str, Any]]:
    try:
        start = text.find("{")
        end = text.rfind("}") + 1
        if start < 0 or end <= start:
            return []
        data = json.loads(text[start:end])
    except json.JSONDecodeError:
        logger.warning("Failed to parse rerank response")
        return []

    evaluations = data.get("evaluations", [])
    if not isinstance(evaluations, list):
        return []

    result: list[dict[str, Any]] = []
    for ev in evaluations:
        if not isinstance(ev, dict):
            continue

        brief_id = ev.get("brief_id")
        if not isinstance(brief_id, int):
            continue

        new_confidence = ev.get("new_confidence", 0)
        if not isinstance(new_confidence, int):
            new_confidence = 0
        new_confidence = max(0, min(100, new_confidence))

        direction = ev.get("direction", "hold")
        if direction not in VALID_DIRECTIONS:
            direction = "hold"

        reasoning = str(ev.get("reasoning", ""))

        result.append(
            {
                "brief_id": brief_id,
                "new_confidence": new_confidence,
                "direction": direction,
                "reasoning": reasoning,
            }
        )

    return result


def rerank_briefs(
    db_client: RerankDBClient,
    llm_completions: Any,
    model: str,
    today: date | None = None,
) -> dict[str, Any]:
    from datetime import UTC, datetime

    today = today or datetime.now(tz=UTC).date()

    briefs = db_client.list_briefs_for_reranking()
    if not briefs:
        return {"status": "success", "reranked": 0}

    trends = db_client.list_top_active_seed_signals()
    messages = build_rerank_prompt(briefs, trends, today)

    try:
        raw = llm_completions.with_raw_response.create(
            model=model,
            messages=messages,
            temperature=0,
            max_tokens=4096,
        )
        headers = raw.headers
        response = raw.parse()
        _log_llm_response(response, headers, "reranking")
        if not response.choices:
            return {"status": "error", "error": "LLM returned empty choices"}
        content = response.choices[0].message.content or ""
    except Exception as e:  # noqa: BLE001
        logger.warning("Reranking LLM call failed: %s", e)
        return {"status": "error", "error": str(e)}

    evaluations = parse_rerank_response(content)
    valid_ids = {b.id for b in briefs}

    updated = 0
    for ev in evaluations:
        if ev["brief_id"] not in valid_ids:
            continue
        db_client.update_brief_confidence(
            brief_id=ev["brief_id"],
            confidence=ev["new_confidence"],
            reasoning=ev["reasoning"],
        )
        updated += 1

    logger.info("Brief reranking: updated=%d of %d briefs", updated, len(briefs))
    return {"status": "success", "reranked": updated}
