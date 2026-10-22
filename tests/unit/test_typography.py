"""Tests for the typography of a design.

Every design was drawn in one font, one weight, centred, with no tracking and
a uniform outline around it. That outline is the "meme font" look, and it is
the biggest reason the artwork reads as amateur (issue #226).
"""

from __future__ import annotations

from store_listing.orchestration.typography import (
    DEFAULT_FONT,
    FONTS,
    accent_for,
    family_for_voice,
    font_path,
    treatment_for_voice,
)


class TestTheFontRegistry:
    def test_every_font_is_on_disk(self) -> None:
        for family in FONTS:
            assert font_path(family).is_file(), f"the font file of {family} is missing"

    def test_every_font_carries_its_licence(self) -> None:
        for family, spec in FONTS.items():
            licence = font_path(family).parent / spec.licence
            assert licence.is_file(), f"the licence of {family} is missing"

    def test_a_font_that_is_not_registered_falls_back(self) -> None:
        assert font_path("comic-sans").parent == font_path(DEFAULT_FONT).parent


class TestTheVoiceDecidesTheType:
    def test_every_voice_has_a_font(self) -> None:
        for voice in ("deadpan", "loud", "cozy"):
            assert family_for_voice(voice) in FONTS

    def test_an_unknown_voice_falls_back(self) -> None:
        assert family_for_voice("shouty") in FONTS

    def test_every_voice_has_a_treatment(self) -> None:
        for voice in ("deadpan", "loud", "cozy"):
            treatment = treatment_for_voice(voice)
            assert treatment.style in ("shadow", "block", "halo")

    def test_the_loud_voice_sets_a_word_apart(self) -> None:
        # Two colours and a wide tracking: that is what makes a line shout.
        assert treatment_for_voice("loud").accent_word is not None
        assert treatment_for_voice("loud").tracking > treatment_for_voice("deadpan").tracking

    def test_the_cozy_voice_is_not_shouting(self) -> None:
        assert not treatment_for_voice("cozy").uppercase
        assert treatment_for_voice("cozy").accent_word is None


class TestTheAccentColour:
    def test_a_dark_text_gets_a_different_accent(self) -> None:
        assert accent_for("#000000") != "#000000"

    def test_a_light_text_gets_a_different_accent(self) -> None:
        assert accent_for("#FFFFFF") != "#FFFFFF"

    def test_the_accent_is_the_same_for_every_call(self) -> None:
        assert accent_for("#101010") == accent_for("#101010")


class TestTheFontFiles:
    def test_the_fonts_are_the_ones_we_chose(self) -> None:
        labels = {spec.label for spec in FONTS.values()}
        for label in (
            "Bebas Neue",
            "Anton",
            "Alfa Slab One",
            "Pacifico",
            "Bungee",
            "Rye",
            "Abril Fatface",
        ):
            assert label in labels

    def test_a_font_is_a_real_font_file(self) -> None:
        path = font_path("bebas")
        assert path.suffix == ".ttf"
        assert path.stat().st_size > 1000
