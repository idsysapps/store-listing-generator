"""Tests for the shirt concept generation step."""

from __future__ import annotations

import json
from collections.abc import Callable

from store_listing.orchestration.concept_generation import (
    generate_shirt_concept,
    needs_concept,
)


def _complete(concepts: list[dict]) -> Callable[[list[dict]], str]:
    payload = {"concepts": concepts}

    def _fn(messages: list[dict]) -> str:
        return json.dumps(payload)

    return _fn


# ---------------------------------------------------------------------------
# needs_concept — detects concept-less briefs
# ---------------------------------------------------------------------------


def test_bare_character_name_needs_concept() -> None:
    assert needs_concept("Flash the sloth from Zootopia") is True


def test_character_with_dash_concept_does_not_need_concept() -> None:
    assert needs_concept("Flash the sloth - Sloth Running Team") is False


def test_quoted_tagline_does_not_need_concept() -> None:
    assert needs_concept('Flash the sloth "We\'ll get there when we get there"') is False


def test_character_with_theme_still_needs_concept() -> None:
    # A theme modifier alone is not a creative concept
    assert needs_concept("Flash the sloth - Thanksgiving themed") is True


def test_character_with_named_concept_does_not_need_concept() -> None:
    assert needs_concept("Mickey Mouse - Sloth Running Team") is False


# ---------------------------------------------------------------------------
# generate_shirt_concept
# ---------------------------------------------------------------------------


def test_highest_scoring_concept_is_returned() -> None:
    concepts = [
        {"title": "Slow and Steady", "tagline": "The turtle would be jealous", "score": 70},
        {
            "title": "Sloth Running Team",
            "tagline": "We'll get there when we get there",
            "score": 92,
        },
        {"title": "Born to be Slow", "tagline": "Living life in the slow lane", "score": 65},
    ]
    result = generate_shirt_concept(
        "Flash the sloth from Zootopia",
        complete=_complete(concepts),
    )

    assert "Sloth Running Team" in result
    assert "We'll get there when we get there" in result


def test_original_sentence_returned_when_concept_not_needed() -> None:
    sentence = "Flash the sloth - Sloth Running Team"
    result = generate_shirt_concept(sentence, complete=_complete([]))
    assert result == sentence


def test_original_sentence_returned_on_llm_failure() -> None:
    def bad_complete(messages: list) -> str:
        return "not json"

    result = generate_shirt_concept(
        "Flash the sloth from Zootopia",
        complete=bad_complete,
    )
    assert result == "Flash the sloth from Zootopia"


def test_original_sentence_returned_when_no_concepts() -> None:
    result = generate_shirt_concept(
        "Flash the sloth from Zootopia",
        complete=_complete([]),
    )
    assert result == "Flash the sloth from Zootopia"


def test_winning_concept_formatted_as_customer_sentence() -> None:
    """The winning concept produces a sentence the pipeline can process."""
    concepts = [
        {
            "title": "Sloth Running Team",
            "tagline": "We'll get there when we get there",
            "score": 92,
        },
    ]
    result = generate_shirt_concept(
        "Flash the sloth from Zootopia",
        complete=_complete(concepts),
    )
    # Should include the character and the concept direction
    assert "Flash" in result or "sloth" in result.lower()
    assert "Sloth Running Team" in result
