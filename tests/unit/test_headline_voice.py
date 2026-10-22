"""Unit tests for the voice of the text a brief carries.

The text is what the buyer reads, and the model writes it. A headline that
describes the picture ("Thanksgiving in the Magic Kingdom") or that lists three
nouns ("Feast, fun, and fireworks") is not a line anybody wears, so the gate
drops the brief and the batch asks again (issue #225).

The gate is deliberately mechanical: a word list cannot tell a good line from a
bad one, so the rules are about the shape of the text - a list, a repetition, a
brand name - and the voice itself is taught by the prompt.
"""

from store_listing.orchestration.headline_voice import (
    DEFAULT_VOICE,
    VOICES,
    normalise_voice,
    screen_text,
)


class TestVoices:
    def test_the_store_sells_three_voices(self) -> None:
        assert VOICES == ("deadpan", "loud", "cozy")

    def test_an_unknown_voice_falls_back_to_the_default(self) -> None:
        assert normalise_voice("shouty") == DEFAULT_VOICE

    def test_a_known_voice_is_kept(self) -> None:
        assert normalise_voice("cozy") == "cozy"

    def test_an_empty_voice_falls_back_to_the_default(self) -> None:
        assert normalise_voice(None) == DEFAULT_VOICE


class TestScreenText:
    def test_a_line_is_accepted(self) -> None:
        assert (
            screen_text(
                "Gobble Til The Drawbridge Drops",
                tagline="Feast at the old castle",
                scene="a stone castle with a wide gate and lanterns",
                voice="loud",
            )
            is None
        )

    def test_a_headline_that_names_a_brand_is_rejected(self) -> None:
        reason = screen_text(
            "Thanksgiving in the Magic Kingdom",
            scene="a fairytale castle with autumn trees",
            voice="cozy",
        )
        assert reason is not None
        assert "brand" in reason.lower()

    def test_a_noun_list_is_rejected(self) -> None:
        reason = screen_text(
            "Feast, fun, and fireworks",
            scene="a castle with fireworks above it",
            voice="cozy",
        )
        assert reason is not None
        assert "list" in reason.lower()

    def test_two_items_are_not_a_list(self) -> None:
        # "Rock and Roll" is a line, not a list of things.
        assert (
            screen_text(
                "Rock and Roll",
                scene="a pair of headphones with sound waves",
                voice="deadpan",
            )
            is None
        )

    def test_a_headline_that_only_restates_the_picture_is_rejected(self) -> None:
        reason = screen_text(
            "A Castle With Autumn Trees",
            scene="a stone castle with autumn trees and falling leaves",
            voice="deadpan",
        )
        assert reason is not None
        assert "picture" in reason.lower()

    def test_a_tagline_that_repeats_the_headline_is_rejected(self) -> None:
        reason = screen_text(
            "Gobble Mode",
            tagline="Gobble Mode All Day Long",
            scene="a bird with a bib holding a drumstick",
            voice="loud",
        )
        assert reason is not None
        assert "tagline" in reason.lower()

    def test_a_too_long_headline_is_rejected(self) -> None:
        reason = screen_text(
            "The Very Best Day For A Sandwich In The Whole Wide World",
            scene="a sandwich on a plate",
            voice="deadpan",
        )
        assert reason is not None
        assert "long" in reason.lower()

    def test_a_deadpan_headline_of_pure_sentiment_is_rejected(self) -> None:
        # A deadpan line says something. "Warm love and happy memories" says
        # nothing.
        reason = screen_text(
            "Warm Love And Happy Memories",
            scene="two cats under a blanket",
            voice="deadpan",
        )
        assert reason is not None

    def test_the_same_sentiment_is_allowed_in_the_cozy_voice(self) -> None:
        assert (
            screen_text(
                "Warm Love And Happy Memories",
                scene="two cats under a blanket",
                voice="cozy",
            )
            is None
        )

    def test_an_empty_headline_is_not_screened(self) -> None:
        # A design with no text is legitimate - the gate is about the words.
        assert screen_text(None, scene="a cat in a hat") is None
        assert screen_text("") is None
