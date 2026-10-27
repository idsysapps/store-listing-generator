"""Tests for the art direction of a design.

Every brief came out as the same flat vector clip-art, because the prompt
bolted a fixed tail onto whatever the model wrote in `visual_style`. The art
direction is a closed vocabulary now (issue #227).
"""

from __future__ import annotations

from store_listing.orchestration.art_direction import (
    PRESET_IDS,
    PRESETS,
    family_for,
    negative_for,
    preset_for,
    prompt_for,
    tagline_family_for,
)
from store_listing.orchestration.typography import FONTS


class TestTheVocabulary:
    def test_there_are_eight_looks(self) -> None:
        assert len(PRESETS) == 8

    def test_the_looks_are_the_ones_we_chose(self) -> None:
        assert set(PRESET_IDS) == {
            "vintage_screenprint",
            "seventies_retro",
            "halftone_comic",
            "riso_two_color",
            "hand_inked_woodcut",
            "kawaii_flat",
            "art_nouveau",
            "distressed_band_tee",
        }

    def test_every_look_has_a_prompt_and_a_negative(self) -> None:
        for preset in PRESETS.values():
            assert preset.prompt
            assert preset.negative

    def test_every_look_has_a_palette(self) -> None:
        for preset in PRESETS.values():
            assert len(preset.palette) >= 3
            for colour in preset.palette:
                assert colour.startswith("#")
                assert len(colour) == 7

    def test_every_look_has_a_type_that_exists(self) -> None:
        for preset in PRESETS.values():
            assert preset.headline_family in FONTS
            assert preset.tagline_family in FONTS

    def test_a_look_that_is_not_sold_is_not_found(self) -> None:
        assert preset_for("cubism") is None
        assert preset_for(None) is None
        assert preset_for("") is None


class TestTheLookCannotPaintTheBackground:
    def test_no_look_asks_for_a_paper(self) -> None:
        # A paper texture would survive the DTF cutout as a grey rectangle.
        for preset in PRESETS.values():
            lowered = preset.prompt.lower()
            for word in ("paper", "surface", "background", "texture"):
                assert word not in lowered, f"{preset.id} asks for {word}"

    def test_the_comic_look_does_not_use_a_halftone_screen(self) -> None:
        # The hard dot grid read as little squares all over the art.
        preset = preset_for("halftone_comic")
        assert preset is not None

        assert "halftone" not in preset.prompt.lower()
        assert preset.texture != "halftone"
        assert preset.texture is None, "the look adds no screen of its own"
        assert preset.despeckle, "the generated art is despeckled before the type"

    def test_the_negative_keeps_the_background_out(self) -> None:
        negative = negative_for(preset_for("riso_two_color"))

        assert "paper texture" in negative
        assert "background texture" in negative

    def test_the_negative_of_an_unknown_look_is_still_a_negative(self) -> None:
        negative = negative_for(None)

        assert "lettering" in negative


class TestTheTypeOfTheLook:
    def test_the_look_decides_the_family(self) -> None:
        preset = preset_for("halftone_comic")
        assert preset is not None

        assert family_for(preset, "deadpan") == preset.headline_family
        assert tagline_family_for(preset, "deadpan") == preset.tagline_family

    def test_the_voice_decides_the_family_without_a_look(self) -> None:
        assert family_for(None, "loud") != family_for(None, "cozy")

    def test_the_prompt_of_an_unknown_look_is_empty(self) -> None:
        assert prompt_for(None) == ""
