"""Tests for the copy-mode contract of a design brief (issue #258).

A brief carries its idea in one of three ways:

- ``type_hero``: a stacked typographic emblem whose hero word is the picture,
- ``caption``: artwork with a line drawn beside it,
- ``no_text``: an illustration that explains itself.

The mode is separate from the template: the template is geometry, the mode is
whether the design has words at all and who draws them.
"""

from __future__ import annotations

from typing import Any

import pytest

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
from store_listing.orchestration.design_briefs import DesignBrief


def _brief(**overrides: Any) -> DesignBrief:
    defaults: dict[str, Any] = {
        "concept": "Groovy Ghost",
        "product_type": "dtf_apparel",
        "specific_products": [],
        "audience": "halloween shoppers",
        "visual_style": "retro",
        "confidence": 80,
        "reasoning": "test",
        "source_seed_ids": [],
        "layout_type": "auto",
        "headline_text": "GROOVY",
        "tagline_text": "GROOVY GHOST VIBES",
        "scene_description": "a cute ghost",
        "copy_mode": CAPTION,
    }
    defaults.update(overrides)
    return DesignBrief(**defaults)


class TestNormaliseCopyMode:
    @pytest.mark.parametrize("raw", [TYPE_HERO, CAPTION, NO_TEXT])
    def test_a_known_mode_is_kept(self, raw: str) -> None:
        assert (
            normalise_copy_mode(
                raw,
                layout_type="auto",
                headline_text="GROOVY",
                tagline_text=None,
            )
            == raw
        )

    def test_an_art_only_layout_defaults_to_no_text(self) -> None:
        assert (
            normalise_copy_mode(
                None,
                layout_type="full_bleed",
                headline_text=None,
                tagline_text=None,
            )
            == NO_TEXT
        )

    def test_a_hero_layout_defaults_to_type_hero(self) -> None:
        assert (
            normalise_copy_mode(
                None,
                layout_type="type_hero",
                headline_text="GROOVY",
                tagline_text=None,
            )
            == TYPE_HERO
        )

    def test_an_ordinary_layout_with_copy_defaults_to_caption(self) -> None:
        assert (
            normalise_copy_mode(
                None,
                layout_type="auto",
                headline_text="GROOVY",
                tagline_text=None,
            )
            == CAPTION
        )

    def test_a_brief_without_copy_is_no_text(self) -> None:
        assert (
            normalise_copy_mode(
                None,
                layout_type="auto",
                headline_text=None,
                tagline_text=None,
            )
            == NO_TEXT
        )

    def test_a_hero_without_a_headline_falls_back_to_caption(self) -> None:
        assert (
            normalise_copy_mode(
                TYPE_HERO,
                layout_type="auto",
                headline_text=None,
                tagline_text="A short descriptor",
            )
            == CAPTION
        )

    def test_a_hero_without_any_copy_falls_back_to_no_text(self) -> None:
        assert (
            normalise_copy_mode(
                TYPE_HERO,
                layout_type="auto",
                headline_text=None,
                tagline_text=None,
            )
            == NO_TEXT
        )

    def test_a_caption_without_copy_falls_back_to_no_text(self) -> None:
        assert (
            normalise_copy_mode(
                CAPTION,
                layout_type="auto",
                headline_text=None,
                tagline_text=None,
            )
            == NO_TEXT
        )

    def test_a_text_only_layout_is_still_a_caption(self) -> None:
        assert (
            normalise_copy_mode(
                None,
                layout_type="text_only",
                headline_text="JUST A SLOGAN",
                tagline_text=None,
            )
            == CAPTION
        )

    def test_a_text_only_layout_without_copy_is_still_a_caption(self) -> None:
        # The parser falls back to the concept for a text-only brief, so the
        # mode must not turn it into a no-text row before that happens.
        assert (
            normalise_copy_mode(
                None,
                layout_type="text_only",
                headline_text=None,
                tagline_text=None,
            )
            == CAPTION
        )


class TestCopyModeFromRow:
    def test_a_stored_mode_wins(self) -> None:
        row = {
            "copy_mode": NO_TEXT,
            "layout_type": "auto",
            "headline_text": "STALE HEADLINE",
            "tagline_text": None,
        }

        assert copy_mode_from_row(row) == NO_TEXT

    def test_a_legacy_art_only_row_is_no_text(self) -> None:
        row = {
            "layout_type": "full_bleed",
            "headline_text": None,
            "tagline_text": None,
        }

        assert copy_mode_from_row(row) == NO_TEXT

    def test_a_legacy_hero_row_is_type_hero(self) -> None:
        row = {
            "layout_type": "wordmark_stack",
            "headline_text": "GROOVY",
            "tagline_text": None,
        }

        assert copy_mode_from_row(row) == TYPE_HERO

    def test_a_legacy_text_row_is_a_caption(self) -> None:
        row = {
            "layout_type": "text_top",
            "headline_text": "GROOVY",
            "tagline_text": None,
        }

        assert copy_mode_from_row(row) == CAPTION

    def test_a_legacy_row_with_a_text_plan_is_a_caption(self) -> None:
        # A plan was written to put words in the picture, so the row is not
        # art-only even when it predates the copy_mode column.
        row = {
            "text_plan": {"method": "carved", "carrier": "headstone"},
            "scene_description": "a quiet graveyard",
        }

        assert copy_mode_from_row(row) == CAPTION

    def test_every_hero_layout_is_known(self) -> None:
        assert HERO_LAYOUT_TYPES == frozenset({"type_hero", "wordmark_stack"})


class TestKicker:
    def test_a_short_kicker_is_kept_for_a_hero(self) -> None:
        assert normalise_kicker("  in   my spooky era ", copy_mode=TYPE_HERO) == (
            "in my spooky era"
        )

    def test_a_kicker_is_dropped_for_a_caption(self) -> None:
        assert normalise_kicker("in my spooky era", copy_mode=CAPTION) is None

    def test_a_kicker_is_dropped_for_a_no_text_design(self) -> None:
        assert normalise_kicker("in my spooky era", copy_mode=NO_TEXT) is None

    def test_a_kicker_longer_than_six_words_is_dropped(self) -> None:
        assert normalise_kicker("one two three four five six seven", copy_mode=TYPE_HERO) is None

    def test_a_non_string_kicker_is_dropped(self) -> None:
        assert normalise_kicker(42, copy_mode=TYPE_HERO) is None

    def test_an_empty_kicker_is_none(self) -> None:
        assert normalise_kicker("   ", copy_mode=TYPE_HERO) is None


class TestAssignBatchCopyModes:
    def test_an_empty_batch_stays_empty(self) -> None:
        assert assign_batch_copy_modes([]) == []

    def test_a_single_custom_brief_is_not_forced_into_the_hero_style(self) -> None:
        brief = _brief()

        result = assign_batch_copy_modes([brief])

        assert result[0].copy_mode == CAPTION

    def test_a_batch_without_a_hero_promotes_a_short_caption(self) -> None:
        briefs = [
            _brief(
                concept="Long Caption",
                headline_text="A LONG CAPTION LINE THAT IS NOT A HERO WORD",
                confidence=99,
            ),
            _brief(concept="Short Hero", headline_text="BOO JEE", confidence=70),
        ]

        result = assign_batch_copy_modes(briefs)

        assert [brief.copy_mode for brief in result] == [CAPTION, TYPE_HERO]

    def test_a_brief_without_a_scene_is_not_promoted(self) -> None:
        briefs = [
            _brief(concept="One", headline_text="BOO", scene_description=None),
            _brief(concept="Two", headline_text="JEE", scene_description=None),
        ]

        result = assign_batch_copy_modes(briefs)

        assert [brief.copy_mode for brief in result] == [CAPTION, CAPTION]

    def test_a_brief_with_a_long_headline_is_not_promoted(self) -> None:
        briefs = [
            _brief(concept="One", headline_text="BOO"),
            _brief(concept="Two", headline_text="A CAPTION OF FIVE WORDS"),
        ]

        result = assign_batch_copy_modes(briefs)

        # The first brief is short enough; the second is not.
        assert result[0].copy_mode == TYPE_HERO
        assert result[1].copy_mode == CAPTION

    def test_hero_briefs_are_capped_at_half_the_batch(self) -> None:
        briefs = [
            _brief(concept="A", confidence=10, copy_mode=TYPE_HERO, kicker_text="A KICKER"),
            _brief(concept="B", confidence=20, copy_mode=TYPE_HERO, kicker_text="B KICKER"),
            _brief(concept="C", confidence=30, copy_mode=TYPE_HERO, kicker_text="C KICKER"),
            _brief(concept="D", confidence=40, copy_mode=TYPE_HERO, kicker_text="D KICKER"),
        ]

        result = assign_batch_copy_modes(briefs)

        assert [brief.copy_mode for brief in result] == [CAPTION, CAPTION, TYPE_HERO, TYPE_HERO]
        assert result[0].kicker_text is None
        assert result[1].kicker_text is None
        assert result[2].kicker_text == "C KICKER"

    def test_demoting_a_hero_layout_clears_the_layout_too(self) -> None:
        briefs = [
            _brief(
                concept="A",
                confidence=10,
                layout_type="wordmark_stack",
                copy_mode=TYPE_HERO,
            ),
            _brief(concept="B", confidence=20, copy_mode=TYPE_HERO),
            _brief(concept="C", confidence=90),
        ]

        result = assign_batch_copy_modes(briefs)

        assert result[0].copy_mode == CAPTION
        assert result[0].layout_type == "auto"
        assert result[1].copy_mode == TYPE_HERO

    def test_no_text_briefs_are_left_alone(self) -> None:
        briefs = [
            _brief(concept="A", headline_text=None, layout_type="full_bleed", copy_mode=NO_TEXT),
            _brief(concept="B", headline_text="BOO"),
        ]

        result = assign_batch_copy_modes(briefs)

        assert result[0].copy_mode == NO_TEXT
        assert result[1].copy_mode == TYPE_HERO
