"""Generate a creative shirt concept when the customer names a character but not a direction.

When someone submits "Flash the sloth from Zootopia" with no creative hook, this module
asks the LLM to brainstorm N concepts, scores each, and returns the highest-scoring one
as an enriched sentence the pipeline can process normally.

The winning concept is formatted as:
  "{original sentence} - {title} - "{tagline}""

e.g. "Flash the sloth from Zootopia - Sloth Running Team - "We'll get there when we get there""

The pipeline's quoted_lettering() picks up the tagline; the title becomes the design concept.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable

from store_listing.orchestration.json_response import extract_json_objects
from store_listing.orchestration.personal_design import quoted_lettering

logger = logging.getLogger(__name__)

_CONCEPT_COUNT = 3

_CONCEPT_SYSTEM = (
    "You are a creative shirt designer. Given a character or subject the customer named, "
    f"invent {_CONCEPT_COUNT} clever, funny, or charming shirt design concepts that suit their personality. "
    "Think about their famous traits, catchphrases, job, or running jokes. "
    "Return JSON only: "
    '{"concepts": [{"title": "2-5 word shirt concept", '
    '"tagline": "witty one-liner that fits the character", '
    '"score": 0-100}]}. '
    "Score on cleverness, freshness, and immediate wearability — 90+ means someone would "
    "buy it instantly. "
    "Example for 'Flash the sloth from Zootopia': "
    '{"title": "Sloth Running Team", '
    '"tagline": "We\'ll get there when we get there", "score": 92}'
)

# Separators that signal the customer already has a creative direction
_CONCEPT_SEPARATORS = re.compile(r"\s*[-–—,]\s*", re.UNICODE)


def needs_concept(sentence: str) -> bool:
    """Whether the customer's sentence lacks a creative design concept.

    Returns True when the sentence names a subject but provides no title,
    no quoted tagline, and no dash/comma-separated concept direction.
    Returns False when the customer has already given the creative hook.
    """
    if quoted_lettering(sentence):
        return False
    parts = _CONCEPT_SEPARATORS.split(sentence.strip())
    if len(parts) <= 1:
        return True
    # Additional parts separated by dash/comma are a creative concept UNLESS they
    # are pure theme modifiers ending with a theme suffix word (e.g. "Thanksgiving themed",
    # "safari style"). A real concept ("Sloth Running Team") ends with a content word.
    _THEME_SUFFIX = {"themed", "theme", "style", "edition", "version", "inspired", "mode"}
    for part in parts[1:]:
        words = part.lower().split()
        if not words:
            continue
        if words[-1] not in _THEME_SUFFIX:
            return False  # Ends with a content word — this IS a creative concept
    return True  # All additional parts are theme modifiers → no concept yet


def generate_shirt_concept(
    sentence: str,
    *,
    complete: Callable[[list[dict[str, str]]], str],
) -> str:
    """Return an enriched sentence with the highest-scoring creative concept appended.

    When the sentence already has a concept direction (dash-separated or quoted),
    returns it unchanged. On LLM failure, returns the original sentence so the
    pipeline always has something to work with.
    """
    if not needs_concept(sentence):
        return sentence

    try:
        raw = complete(
            [
                {"role": "system", "content": _CONCEPT_SYSTEM},
                {"role": "user", "content": f"Character/subject: {sentence}"},
            ]
        )
        concepts = _parse_concepts(raw)
        if not concepts:
            return sentence
        best = max(concepts, key=lambda c: c.get("score", 0))
        title = str(best.get("title") or "").strip()
        tagline = str(best.get("tagline") or "").strip()
        if not title:
            return sentence
        enriched = f"{sentence} - {title}"
        if tagline:
            enriched = f'{enriched} - "{tagline}"'
        logger.info(
            "Concept generation picked %r (score=%s) for %r",
            title,
            best.get("score"),
            sentence,
        )
        return enriched
    except Exception:  # noqa: BLE001
        logger.warning("Concept generation failed for %r", sentence)
        return sentence


def _parse_concepts(text: str) -> list[dict]:
    for obj in extract_json_objects(text):
        if not isinstance(obj, dict):
            continue
        concepts = obj.get("concepts")
        if isinstance(concepts, list):
            return [c for c in concepts if isinstance(c, dict) and c.get("title")]
    return []
