"""Tests for the Qwen-Image typographic hero path (issue #254).

The Phase 0 bake-off showed that Qwen-Image can render a kicker and a hero
word together with the subject weaving through the letters. It cannot also
render the descriptor, so the descriptor stays out of this path.
"""

from __future__ import annotations

from typing import Any

import pytest

from store_listing.orchestration.comfyui_client import ComfyUIError, ComfyUIResult
from store_listing.orchestration.design_briefs import DesignBrief
from store_listing.orchestration.text_validation import StrictTextResult
from store_listing.orchestration.typographic_hero import (
    HeroCopy,
    TypographicHeroGenerator,
    build_typographic_hero_prompt,
    hero_brief_ids,
    hero_candidate_count,
    hero_copy_from_brief,
    is_typographic_hero_brief,
    typographic_hero_enabled,
)


def _brief(
    *,
    headline_text: str | None = "GROOVY",
    tagline_text: str | None = None,
    scene_description: str | None = "a cute ghost",
    layout_type: str = "full_bleed",
    copy_mode: str = "caption",
    kicker_text: str | None = None,
) -> DesignBrief:
    return DesignBrief(
        concept="a groovy ghost",
        product_type="sublimation",
        specific_products=[],
        audience="halloween shoppers",
        visual_style="retro",
        confidence=80,
        reasoning="test",
        source_seed_ids=[],
        layout_type=layout_type,
        headline_text=headline_text,
        tagline_text=tagline_text,
        copy_mode=copy_mode,
        kicker_text=kicker_text,
        scene_description=scene_description,
    )


def _brief_with_kicker(kicker: str | None, **overrides: Any) -> DesignBrief:
    return _brief(kicker_text=kicker, **overrides)


def _result(image: bytes = b"image", seed: int | None = 1) -> ComfyUIResult:
    return ComfyUIResult(image_bytes=image, prompt_id="prompt-1", seed=seed)


class _FakeComfy:
    def __init__(self, results: list[object]) -> None:
        self._results = list(results)
        self.calls: list[dict] = []

    def generate_qwen_image(self, **kwargs: object) -> ComfyUIResult:
        self.calls.append(dict(kwargs))
        result = self._results.pop(0)
        if isinstance(result, Exception):
            raise result
        assert isinstance(result, ComfyUIResult)
        return result


class _FakeValidator:
    def __init__(self, outcomes: list[bool]) -> None:
        self._outcomes = list(outcomes)
        self.calls: list[tuple[bytes, tuple[str, ...]]] = []

    def validate(self, image_bytes: bytes, *, expected: list[str]) -> StrictTextResult:
        self.calls.append((image_bytes, tuple(expected)))
        matched = self._outcomes.pop(0)
        return StrictTextResult(
            matched=matched,
            expected=list(expected),
            observed="",
            score=1.0 if matched else 0.0,
        )


class TestBuildPrompt:
    def test_carries_the_hero_and_the_kicker(self) -> None:
        prompt = build_typographic_hero_prompt(
            HeroCopy(
                hero="GROOVY",
                kicker="IN MY SPOOKY ERA",
                subject="a cute groovy ghost",
            )
        )

        assert 'the word "GROOVY"' in prompt
        assert 'the words "IN MY SPOOKY ERA"' in prompt
        assert "a cute groovy ghost weaving through and overlapping" in prompt
        assert "hard offset shadow" in prompt
        assert "plain white background" in prompt

    def test_never_writes_the_descriptor(self) -> None:
        prompt = build_typographic_hero_prompt(
            HeroCopy(
                hero="GROOVY",
                kicker="IN MY SPOOKY ERA",
                descriptor="GROOVY GHOST VIBES",
                subject="a cute ghost",
            )
        )

        assert "GROOVY GHOST VIBES" not in prompt

    def test_a_wordmark_without_a_subject_has_no_interlock(self) -> None:
        prompt = build_typographic_hero_prompt(HeroCopy(hero="GROOVY"))

        assert "A die-cut sticker wordmark" in prompt
        assert "weaving through" not in prompt
        assert "At the top" not in prompt

    def test_a_kicker_is_optional(self) -> None:
        prompt = build_typographic_hero_prompt(HeroCopy(hero="GROOVY", subject="a ghost"))

        assert "At the top" not in prompt
        assert 'the word "GROOVY"' in prompt

    def test_strips_quotes_from_the_copy(self) -> None:
        prompt = build_typographic_hero_prompt(
            HeroCopy(hero='GRO"OVY', kicker='SAY "BOO"', subject="a ghost")
        )

        assert '"GROOVY"' in prompt
        assert '"SAY BOO"' in prompt
        assert '""' not in prompt

    def test_is_deterministic(self) -> None:
        copy = HeroCopy(hero="GROOVY", kicker="IN MY SPOOKY ERA", subject="a ghost")
        assert build_typographic_hero_prompt(copy) == build_typographic_hero_prompt(copy)

    def test_an_empty_hero_is_rejected(self) -> None:
        with pytest.raises(ValueError):
            HeroCopy(hero="   ")


class TestHeroCopy:
    def test_text_blocks_are_the_kicker_and_the_hero(self) -> None:
        copy = HeroCopy(
            hero="GROOVY",
            kicker="IN MY SPOOKY ERA",
            descriptor="GROOVY GHOST VIBES",
        )

        assert copy.text_blocks == ("IN MY SPOOKY ERA", "GROOVY")

    def test_text_blocks_without_a_kicker(self) -> None:
        assert HeroCopy(hero="GROOVY").text_blocks == ("GROOVY",)


class TestHeroCopyFromBrief:
    def test_maps_the_brief_fields(self) -> None:
        brief = _brief_with_kicker(
            "IN MY SPOOKY ERA",
            headline_text="GROOVY",
            tagline_text="GROOVY GHOST VIBES",
            scene_description="a cute ghost with a speech bubble saying 'BOO'",
        )

        copy = hero_copy_from_brief(brief)

        assert copy.hero == "GROOVY"
        assert copy.kicker == "IN MY SPOOKY ERA"
        assert copy.descriptor == "GROOVY GHOST VIBES"
        assert copy.subject == "a cute ghost"

    def test_a_brief_without_a_kicker_is_still_valid(self) -> None:
        brief = _brief(
            headline_text="GROOVY",
            tagline_text=None,
            scene_description="a ghost",
        )

        copy = hero_copy_from_brief(brief)

        assert copy.kicker is None
        assert copy.descriptor is None
        assert copy.text_blocks == ("GROOVY",)

    def test_a_brief_without_a_headline_is_rejected(self) -> None:
        brief = _brief(headline_text=None, tagline_text="x", scene_description="a ghost")

        with pytest.raises(ValueError):
            hero_copy_from_brief(brief)

    def test_maps_the_persisted_kicker(self) -> None:
        brief = _brief(
            headline_text="BOO JEE",
            kicker_text="IN MY SPOOKY ERA",
            copy_mode="type_hero",
            scene_description="a ghost weaving through the letters",
        )

        copy = hero_copy_from_brief(brief)

        assert copy.kicker == "IN MY SPOOKY ERA"


class TestFeatureFlag:
    def test_is_off_by_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("TYPOGRAPHIC_HERO_ENABLED", raising=False)
        assert typographic_hero_enabled() is False

    @pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on"])
    def test_truthy_values_enable_it(self, monkeypatch: pytest.MonkeyPatch, value: str) -> None:
        monkeypatch.setenv("TYPOGRAPHIC_HERO_ENABLED", value)
        assert typographic_hero_enabled() is True

    @pytest.mark.parametrize("value", ["0", "false", "off", ""])
    def test_falsy_values_leave_it_off(self, monkeypatch: pytest.MonkeyPatch, value: str) -> None:
        monkeypatch.setenv("TYPOGRAPHIC_HERO_ENABLED", value)
        assert typographic_hero_enabled() is False


class TestTypographicHeroGenerator:
    def _generator(
        self, results: list[object], outcomes: list[bool], *, candidates: int = 2
    ) -> tuple[TypographicHeroGenerator, _FakeComfy, _FakeValidator]:
        comfy = _FakeComfy(results)
        validator = _FakeValidator(outcomes)
        generator = TypographicHeroGenerator(
            comfyui_client=comfy, validator=validator, candidates=candidates
        )
        return generator, comfy, validator

    def test_returns_the_first_candidate_that_passes(self) -> None:
        generator, comfy, validator = self._generator([_result()], [True])

        result = generator.generate(
            HeroCopy(hero="GROOVY", kicker="IN MY SPOOKY ERA", subject="a ghost"), seed=123
        )

        assert result.image_bytes == b"image"
        assert result.attempts == 1
        assert result.seed == 123
        assert result.error is None
        assert len(comfy.calls) == 1
        assert comfy.calls[0]["seed"] == 123
        assert validator.calls[0][1] == ("IN MY SPOOKY ERA", "GROOVY")

    def test_retries_with_the_next_seed_when_the_gate_fails(self) -> None:
        generator, comfy, _ = self._generator(
            [_result(b"first"), _result(b"second")], [False, True]
        )

        result = generator.generate(HeroCopy(hero="GROOVY"), seed=100)

        assert result.image_bytes == b"second"
        assert result.attempts == 2
        assert result.seed == 101
        assert [call["seed"] for call in comfy.calls] == [100, 101]

    def test_fails_after_the_candidate_budget_is_spent(self) -> None:
        generator, comfy, _ = self._generator(
            [_result(b"first"), _result(b"second")], [False, False]
        )

        result = generator.generate(HeroCopy(hero="GROOVY"), seed=100)

        assert result.image_bytes is None
        assert result.attempts == 2
        assert result.error is not None
        assert "text gate" in result.error
        assert len(comfy.calls) == 2

    def test_a_comfyui_error_does_not_stop_the_next_candidate(self) -> None:
        generator, comfy, _ = self._generator(
            [ComfyUIError("server is busy"), _result(b"second")], [True], candidates=2
        )

        result = generator.generate(HeroCopy(hero="GROOVY"), seed=10)

        assert result.image_bytes == b"second"
        assert result.attempts == 2
        assert len(comfy.calls) == 2

    def test_all_comfyui_errors_are_reported(self) -> None:
        generator, _, _ = self._generator(
            [ComfyUIError("down"), ComfyUIError("still down")], [], candidates=2
        )

        result = generator.generate(HeroCopy(hero="GROOVY"), seed=10)

        assert result.image_bytes is None
        assert result.error is not None
        assert "down" in result.error

    def test_the_descriptor_is_not_sent_to_the_model(self) -> None:
        generator, comfy, _ = self._generator([_result()], [True])

        generator.generate(
            HeroCopy(
                hero="GROOVY",
                kicker="IN MY SPOOKY ERA",
                descriptor="GROOVY GHOST VIBES",
                subject="a ghost",
            ),
            seed=1,
        )

        assert "GROOVY GHOST VIBES" not in comfy.calls[0]["prompt"]

    def test_candidates_must_be_at_least_one(self) -> None:
        with pytest.raises(ValueError):
            TypographicHeroGenerator(
                comfyui_client=_FakeComfy([]), validator=_FakeValidator([]), candidates=0
            )


class TestHeroSelection:
    def test_a_dedicated_layout_opts_a_brief_in(self) -> None:
        brief = _brief(layout_type="type_hero")
        assert is_typographic_hero_brief({"id": 1}, brief) is True

    def test_the_stacked_wordmark_layout_opts_a_brief_in(self) -> None:
        brief = _brief(layout_type="wordmark_stack")
        assert is_typographic_hero_brief({"id": 1}, brief) is True

    def test_an_ordinary_brief_is_not_opted_in(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("TYPOGRAPHIC_HERO_BRIEF_IDS", raising=False)
        brief = _brief(layout_type="full_bleed")
        assert is_typographic_hero_brief({"id": 1}, brief) is False

    def test_a_trial_batch_can_opt_a_brief_in_by_id(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TYPOGRAPHIC_HERO_BRIEF_IDS", "1, 3")
        brief = _brief(layout_type="full_bleed")
        assert is_typographic_hero_brief({"id": 3}, brief) is True
        assert is_typographic_hero_brief({"id": 2}, brief) is False

    def test_a_brief_without_a_headline_is_never_a_hero(self) -> None:
        brief = _brief(headline_text=None, layout_type="type_hero")
        assert is_typographic_hero_brief({"id": 1}, brief) is False

    def test_a_copy_mode_hero_opts_a_brief_in(self) -> None:
        brief = _brief(layout_type="auto", copy_mode="type_hero")
        assert is_typographic_hero_brief({"id": 1}, brief) is True

    def test_a_no_text_brief_is_never_a_hero(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TYPOGRAPHIC_HERO_BRIEF_IDS", "1")
        brief = _brief(layout_type="type_hero", copy_mode="no_text")
        assert is_typographic_hero_brief({"id": 1}, brief) is False

    def test_brief_ids_ignore_junk(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TYPOGRAPHIC_HERO_BRIEF_IDS", "1, 2,abc, 3")
        assert hero_brief_ids() == frozenset({1, 2, 3})

    def test_the_candidate_count_defaults_to_two(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("TYPOGRAPHIC_HERO_CANDIDATES", raising=False)
        assert hero_candidate_count() == 2

    @pytest.mark.parametrize(("value", "expected"), [("3", 3), ("0", 1), ("nope", 2)])
    def test_the_candidate_count_is_configurable(
        self, monkeypatch: pytest.MonkeyPatch, value: str, expected: int
    ) -> None:
        monkeypatch.setenv("TYPOGRAPHIC_HERO_CANDIDATES", value)
        assert hero_candidate_count() == expected
