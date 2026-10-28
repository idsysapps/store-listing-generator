"""The LLM copy editor for the text a design carries.

The text is the part of a t-shirt a buyer reads, so it is the part that has to
be good. Mechanical rules cannot tell a sellable line from a scene description,
so the editorial decision belongs to the model: it decides whether to keep,
drop or rewrite a line, and whether the design already stands alone.

The code in this module only builds the context, reads the answer and applies
it. It deliberately has no word-overlap or preposition rules: those are not a
substitute for taste.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any, Final

from store_listing.orchestration.copy_modes import NO_TEXT
from store_listing.orchestration.design_briefs import DesignBrief
from store_listing.orchestration.json_response import extract_json_objects, merge_items
from store_listing.orchestration.llm_client import create_answer

logger = logging.getLogger(__name__)

__all__ = [
    "CopyBlockReview",
    "CopyReview",
    "apply_copy_reviews",
    "build_copy_review_prompt",
    "copy_review_enabled",
    "parse_copy_review_response",
    "request_copy_reviews",
    "review_batch_copy",
]

KEEP: Final[str] = "keep"
DROP: Final[str] = "drop"
REWRITE: Final[str] = "rewrite"
REJECT: Final[str] = "reject"

_BLOCK_ACTIONS: Final[frozenset[str]] = frozenset({KEEP, DROP, REWRITE})
_HEADLINE_ACTIONS: Final[frozenset[str]] = frozenset({KEEP, REWRITE})
_OVERALL_ACTIONS: Final[frozenset[str]] = frozenset({KEEP, REJECT})

_SYSTEM_PROMPT: Final[str] = """You are the creative director for a print-on-demand t-shirt store.
You review the copy of a design before any image is generated. Your job is to
decide whether each line makes the shirt more wearable, or whether it should be
dropped or rewritten.

DESIGN PRINCIPLES
- A buyer wears an identity, a joke, an occasion or an attitude, not a product
  description.
- The illustration shows the scene. Do not explain what the image already
  shows.
- One idea, one joke, one identity. Extra text dilutes the design.
- The design must work at three distances: as a thumbnail, from across the
  room, and up close.
- Ask: would a stranger wear this without the product description?
- Specificity beats a broad pun. A line for a niche, an inside joke, a
  profession or a subculture is stronger than a line for everyone.
- A headline must work as the main line a person wears.
- A kicker sets up the hero; it must not repeat it.
- A tagline is optional. When the design already stands alone, DROP is the
  best answer. Only keep or rewrite a tagline when it adds a second joke,
  identity, occasion or brand-like line.
- A rewrite is not a way to keep text for its own sake. If the design does not
  need a second line, drop it.

Return one review for every brief. Use these actions:
- headline: KEEP or REWRITE. If no headline can work, set overall_action to
  REJECT.
- kicker: KEEP, DROP or REWRITE.
- tagline: KEEP, DROP or REWRITE.
- overall_action: KEEP or REJECT.
- design_stands_alone: true when the concept, kicker, hero and illustration
  already deliver the idea without a tagline.
- Give a short reason for every decision and a confidence from 0 to 1.

Return ONLY valid JSON with this shape:
{"reviews":[{"index":1,"design_stands_alone":true,"headline":{"action":"KEEP","rewrite":null,"reason":"..."},"kicker":{"action":"KEEP","rewrite":null,"reason":"..."},"tagline":{"action":"DROP","rewrite":null,"reason":"..."},"overall_action":"KEEP","overall_reason":"...","confidence":0.9}]}
"""

_RE_REVIEW_PROMPT: Final[str] = """This is a re-review of copy that the previous pass rewrote.
Judge the current candidate as a final line. You may still KEEP or DROP a
block. Do not rewrite it again: a line that still needs another rewrite is a
line that should be dropped (or a brief that should be REJECTed when the
headline is the problem)."""


def copy_review_enabled() -> bool:
    """Whether the brief pipeline asks the model to edit the copy."""
    value = os.environ.get("LLM_COPY_REVIEW", "true").strip().lower()
    return value not in ("0", "false", "no")


@dataclass(frozen=True)
class CopyBlockReview:
    """The editor's decision about one text block."""

    action: str
    rewrite: str | None = None
    reason: str = ""


@dataclass(frozen=True)
class CopyReview:
    """The editor's decision about one brief."""

    index: int
    design_stands_alone: bool
    headline: CopyBlockReview
    kicker: CopyBlockReview
    tagline: CopyBlockReview
    overall_action: str = KEEP
    overall_reason: str = ""
    confidence: float = 0.0


def _brief_payload(index: int, brief: DesignBrief) -> dict[str, Any]:
    return {
        "index": index,
        "copy_mode": brief.copy_mode,
        "concept": brief.concept,
        "product_type": brief.product_type,
        "audience": brief.audience,
        "kicker": brief.kicker_text,
        "headline": brief.headline_text,
        "tagline": brief.tagline_text,
        "scene": brief.scene_description,
    }


def build_copy_review_prompt(
    briefs: Sequence[DesignBrief],
    *,
    extra_instructions: Sequence[str] | None = None,
    re_review: bool = False,
) -> list[dict[str, str]]:
    """The messages that ask the local model to edit the copy."""
    entries = [
        _brief_payload(index, brief)
        for index, brief in enumerate(briefs, start=1)
        if brief.copy_mode != NO_TEXT
    ]
    system = _SYSTEM_PROMPT
    if re_review:
        system = f"{system}\n\n{_RE_REVIEW_PROMPT}"
    if extra_instructions:
        system = f"{system}\n\n" + "\n".join(extra_instructions)

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps({"briefs": entries}, ensure_ascii=False)},
    ]


def _block_review(value: Any, allowed: frozenset[str]) -> CopyBlockReview | None:
    if not isinstance(value, dict):
        return None
    raw_action = value.get("action")
    if not isinstance(raw_action, str):
        return None
    action = raw_action.strip().lower()
    if action not in allowed:
        return None
    raw_rewrite = value.get("rewrite")
    if raw_rewrite is not None and not isinstance(raw_rewrite, str):
        return None
    rewrite = raw_rewrite.strip() if isinstance(raw_rewrite, str) else None
    if action == REWRITE and not rewrite:
        return None
    raw_reason = value.get("reason")
    reason = raw_reason.strip() if isinstance(raw_reason, str) else ""
    return CopyBlockReview(action=action, rewrite=rewrite, reason=reason)


def _review_from(value: Any) -> CopyReview | None:
    if not isinstance(value, dict):
        return None
    index = value.get("index")
    if not isinstance(index, int) or index < 1:
        return None

    headline = _block_review(value.get("headline"), _HEADLINE_ACTIONS)
    kicker = _block_review(value.get("kicker"), _BLOCK_ACTIONS)
    tagline = _block_review(value.get("tagline"), _BLOCK_ACTIONS)
    if headline is None or kicker is None or tagline is None:
        return None

    raw_overall = value.get("overall_action", KEEP)
    if not isinstance(raw_overall, str):
        return None
    overall_action = raw_overall.strip().lower()
    if overall_action not in _OVERALL_ACTIONS:
        return None

    raw_confidence = value.get("confidence", 0.0)
    try:
        confidence = float(raw_confidence)
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))

    raw_reason = value.get("overall_reason")
    overall_reason = raw_reason.strip() if isinstance(raw_reason, str) else ""

    return CopyReview(
        index=index,
        design_stands_alone=bool(value.get("design_stands_alone", False)),
        headline=headline,
        kicker=kicker,
        tagline=tagline,
        overall_action=overall_action,
        overall_reason=overall_reason,
        confidence=confidence,
    )


def parse_copy_review_response(
    text: str,
    *,
    expected_indexes: set[int],
) -> tuple[list[CopyReview], str | None]:
    """The reviews the model wrote, and why the answer was not complete."""
    objects = extract_json_objects(text)
    items = merge_items(objects, "reviews")
    if not items:
        return [], "the answer has no 'reviews' array"

    by_index: dict[int, CopyReview] = {}
    for item in items:
        review = _review_from(item)
        if review is None or review.index not in expected_indexes:
            continue
        by_index[review.index] = review

    missing = expected_indexes - set(by_index)
    if missing:
        return list(by_index.values()), f"missing reviews for indexes {sorted(missing)}"
    return [by_index[index] for index in sorted(by_index)], None


def _apply_block(
    value: str | None,
    review: CopyBlockReview,
) -> str | None:
    if review.action == DROP:
        return None
    if review.action == REWRITE:
        return review.rewrite
    return value


def apply_copy_reviews(
    briefs: Sequence[DesignBrief],
    reviews: Sequence[CopyReview],
) -> list[DesignBrief]:
    """The briefs with the editor's decisions applied."""
    by_index = {review.index: review for review in reviews}
    result: list[DesignBrief] = []
    for index, brief in enumerate(briefs, start=1):
        review = by_index.get(index)
        if review is None:
            result.append(brief)
            continue
        if review.overall_action == REJECT:
            logger.info(
                "Dropping brief %r after the copy review: %s",
                brief.concept,
                review.overall_reason or "the editor rejected the copy",
            )
            continue

        result.append(
            replace(
                brief,
                headline_text=_apply_block(brief.headline_text, review.headline),
                kicker_text=_apply_block(brief.kicker_text, review.kicker),
                tagline_text=_apply_block(brief.tagline_text, review.tagline),
            )
        )
    return result


def _answer_content(raw: Any) -> tuple[str | None, str | None]:
    try:
        response = raw.parse()
        if not response.choices:
            return None, "the model returned no choices"
        content = response.choices[0].message.content or ""
        return content, None
    except Exception as error:  # noqa: BLE001 - report the failure, do not hide it
        return None, str(error)


def _has_rewrite(reviews: Sequence[CopyReview]) -> bool:
    return any(
        block.action == REWRITE
        for review in reviews
        for block in (review.headline, review.kicker, review.tagline)
    )


def request_copy_reviews(
    llm_completions: Any,
    model: str,
    briefs: Sequence[DesignBrief],
    *,
    extra_instructions: Sequence[str] | None = None,
    re_review: bool = False,
) -> tuple[list[CopyReview], list[str]]:
    """One editor call: the reviews and any reason the answer was unusable."""
    if not any(brief.copy_mode != NO_TEXT for brief in briefs):
        return [], []

    messages = build_copy_review_prompt(
        briefs,
        extra_instructions=extra_instructions,
        re_review=re_review,
    )
    raw, error = create_answer(
        llm_completions,
        model,
        messages,
        "copy_review",
        temperature=0,
    )
    if error is not None:
        return [], [f"copy review failed: {error}"]

    content, read_error = _answer_content(raw)
    if read_error is not None or content is None:
        return [], [f"copy review failed: {read_error or 'empty answer'}"]

    expected = {index for index, brief in enumerate(briefs, start=1) if brief.copy_mode != NO_TEXT}
    reviews, parse_error = parse_copy_review_response(content, expected_indexes=expected)
    if parse_error is not None:
        return [], [f"copy review was incomplete: {parse_error}"]
    return reviews, []


def review_batch_copy(
    llm_completions: Any,
    model: str,
    briefs: Sequence[DesignBrief],
    *,
    extra_instructions: Sequence[str] | None = None,
    max_passes: int = 2,
    min_confidence: float = 0.6,
) -> tuple[list[DesignBrief], list[str]]:
    """Ask the local model to edit the copy, then apply its decisions.

    A rewrite is sent through one more review. If the model still wants to
    rewrite it, the next decision is applied as written; the caller only gets
    the final briefs, so the renderer never has to know about the review.

    A low-confidence review is not applied. The original copy is returned
    with the reason, so a person can look at it instead of acting on an
    uncertain edit.
    """
    original = list(briefs)
    if not any(brief.copy_mode != NO_TEXT for brief in original):
        return original, []

    issues: list[str] = []
    current = original
    passes = max(1, max_passes)
    for pass_number in range(passes):
        reviews, review_issues = request_copy_reviews(
            llm_completions,
            model,
            current,
            extra_instructions=extra_instructions,
            re_review=pass_number > 0,
        )
        if review_issues:
            return original, review_issues

        uncertain = [review for review in reviews if review.confidence < min_confidence]
        if uncertain:
            indexes = ", ".join(str(review.index) for review in uncertain)
            return original, [
                f"copy review confidence was below {min_confidence:.2f} for briefs {indexes}"
            ]

        current = apply_copy_reviews(current, reviews)
        if not current or not _has_rewrite(reviews) or pass_number + 1 >= passes:
            break

    return current, issues
