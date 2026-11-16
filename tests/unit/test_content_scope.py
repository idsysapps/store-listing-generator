"""The merchandise screen keeps a design inside the theme library.

Halloween costumes are allow-listed. A practice or an advocacy is not, and
putting a costume word next to one does not bring it in. The match is a whole
phrase, so a word that merely contains one is kept.
"""

import pytest

from store_listing.orchestration.content_scope import content_rejection, excluded_theme
from store_listing.orchestration.context_seeds import EVERGREEN_SEED_RULES, SEASONAL_SEED_RULES


class TestExcludedTheme:
    def test_an_empty_text_is_in_the_library(self) -> None:
        assert excluded_theme("") is None
        assert excluded_theme(None) is None

    def test_a_family_and_a_hobby_are_in_the_library(self) -> None:
        assert excluded_theme("a family rosary on the kitchen table") is None
        assert excluded_theme("pickleball Saturday") is None
        assert excluded_theme("Loud And Proud") is None

    @pytest.mark.parametrize(
        "text",
        [
            "witch humor shirt",
            "skeleton coffee mug",
            "spooky mug",
            "pumpkin everything",
            "trick or treat bag",
            "a haunted house and a black cat",
            "jack-o-lantern grin",
            "vampire costume",
            "zombie fun run",
            "hocus pocus on a broomstick",
        ],
    )
    def test_halloween_themes_are_allow_listed(self, text: str) -> None:
        assert excluded_theme(text) is None

    def test_a_halloween_label_does_not_carry_an_occult_practice(self) -> None:
        assert excluded_theme("halloween tarot shirt") == "an occult or esoteric practice"
        assert excluded_theme("halloween witchcraft altar") == "an occult or esoteric practice"

    def test_a_halloween_label_does_not_carry_explicit_content(self) -> None:
        assert excluded_theme("sexy halloween costume") == "sexual or explicit content"

    def test_a_pro_life_line_is_in_the_library(self) -> None:
        assert excluded_theme("a pro life onesie for the new baby") is None

    def test_an_occult_practice_is_excluded(self) -> None:
        assert excluded_theme("tarot reading shirt") == "an occult or esoteric practice"

    def test_sexual_content_is_excluded(self) -> None:
        assert excluded_theme("a sexy halloween costume") == "sexual or explicit content"

    def test_advocacy_is_excluded(self) -> None:
        assert excluded_theme("a pro choice rally shirt") == "abortion or contraception advocacy"

    def test_an_identity_theme_is_excluded(self) -> None:
        assert excluded_theme("a same-sex wedding design") == "an LGBTQ+ theme"

    def test_blasphemy_is_excluded(self) -> None:
        assert excluded_theme("a god is dead poster") == "blasphemy"

    def test_a_word_that_merely_contains_a_theme_is_kept(self) -> None:
        assert excluded_theme("unisex tee shirt") is None
        assert excluded_theme("Essex county fair") is None

    def test_the_reason_names_the_field(self) -> None:
        reason = content_rejection(("scene", "a tarot spread"), ("headline", "Family Rosary"))

        assert reason == "the scene depicts an occult or esoteric practice"

    def test_the_calendar_seeds_stay_in_the_library(self) -> None:
        seeds = [seed for rule in SEASONAL_SEED_RULES.values() for seed in rule.seeds]
        seeds.extend(seed for rule in EVERGREEN_SEED_RULES.values() for seed in rule.seeds)

        excluded = [seed for seed in seeds if excluded_theme(seed)]

        assert excluded == []
