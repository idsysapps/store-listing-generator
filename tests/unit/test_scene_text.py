"""Tests for the scene sanitation: a scene must not ask the image model to spell.

Brief #100 asked for "a grinning pumpkin with a speech bubble saying 'To get
to the other side!'" and the image model drew the bubble with the words
"onthe sid" (issue #208).
"""

from __future__ import annotations

from store_listing.orchestration.scene_text import (
    has_lettering,
    lettering_cues,
    strip_lettering,
)

PUMPKIN_SCENE = (
    "A grinning pumpkin with a speech bubble saying 'To get to the other side!' "
    "with a small ghost peeking out"
)


class TestHasLettering:
    def test_a_speech_bubble_with_a_quote_is_lettering(self) -> None:
        assert has_lettering(PUMPKIN_SCENE) is True

    def test_a_plain_scene_is_not_lettering(self) -> None:
        assert has_lettering("a skeleton doing yoga on a beach towel") is False

    def test_a_quoted_phrase_is_lettering(self) -> None:
        assert has_lettering('a banner reading "OPEN 24 HOURS"') is True

    def test_a_sign_that_says_something_is_lettering(self) -> None:
        assert has_lettering("a wooden sign that says WELCOME by the door") is True

    def test_a_scene_that_mentions_a_book_title_is_not_lettering(self) -> None:
        # A title is a word in a sentence, not a word in the picture.
        assert has_lettering("a stack of old novels with a title on the spine") is False


class TestLetteringCues:
    def test_it_names_the_carrying_text(self) -> None:
        cues = lettering_cues(PUMPKIN_SCENE)

        assert any("speech bubble" in cue for cue in cues)
        assert any("other side" in cue for cue in cues)

    def test_a_clean_scene_has_no_cues(self) -> None:
        assert lettering_cues("a skeleton doing yoga on a beach towel") == []


class TestStripLettering:
    def test_it_removes_the_quoted_punchline(self) -> None:
        cleaned = strip_lettering(PUMPKIN_SCENE)

        assert "other side" not in cleaned
        assert "speech bubble" not in cleaned

    def test_it_keeps_the_picture(self) -> None:
        cleaned = strip_lettering(PUMPKIN_SCENE)

        assert "pumpkin" in cleaned
        assert "ghost" in cleaned

    def test_it_leaves_a_clean_scene_alone(self) -> None:
        scene = "a skeleton doing yoga on a beach towel"

        assert strip_lettering(scene) == scene

    def test_it_removes_a_sign_that_says_something(self) -> None:
        cleaned = strip_lettering('a shop sign that says "OPEN 24 HOURS" above a wooden door')

        assert "OPEN" not in cleaned
        assert "sign" not in cleaned
        assert "wooden door" in cleaned

    def test_it_removes_a_banner_with_a_quote(self) -> None:
        cleaned = strip_lettering('a cafe awning with the words "COFFEE" on it')

        assert "COFFEE" not in cleaned
        assert "cafe awning" in cleaned

    def test_it_does_not_leave_a_doubled_word_behind(self) -> None:
        cleaned = strip_lettering("a cat with a speech bubble saying 'hi' and a hat")

        assert "with with" not in cleaned
        assert "  " not in cleaned

    def test_a_scene_that_is_only_text_becomes_empty(self) -> None:
        assert strip_lettering("the words 'TO GET TO THE OTHER SIDE'").strip() == ""

    def test_it_keeps_a_scene_that_merely_looks_like_a_sign(self) -> None:
        scene = "a cat sitting next to a wooden signpost in a field"

        assert strip_lettering(scene) == scene
