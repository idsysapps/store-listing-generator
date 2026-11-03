"""Tests for model-drawn text on plans the compositor cannot execute (#133).

A "painted" plan asks for the words to live inside the artwork. The compositor
can only overlay a font, so these briefs go to Qwen-Image, which draws the
scene and the line in one pass. The strict OCR gate rejects a misspelling.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from store_listing.orchestration.comfyui_client import ComfyUIError, ComfyUIResult
from store_listing.orchestration.design_briefs import DesignBrief
from store_listing.orchestration.model_text import (
    ModelTextCopy,
    ModelTextGenerator,
    build_model_text_prompt,
    is_model_text_brief,
    model_text_copy_from_brief,
    model_text_enabled,
)
from store_listing.orchestration.text_integration import TextPlan
from store_listing.orchestration.text_validation import StrictTextResult


def _brief(
    *,
    headline_text: str | None = "Forever In My Heart",
    tagline_text: str | None = "All Souls Day",
    scene_description: str | None = (
        "a black labrador silhouette dissolving into soft, glowing white smoke"
    ),
    copy_mode: str = "caption",
    method: str = "painted",
    carrier: str = "none",
    reason: str = "painted directly into the soft, glowing smoke and light",
    ink: str = "light",
    voice: str = "cozy",
    visual_style: str = "vintage screenprint",
    product_type: str = "dtf_apparel",
) -> DesignBrief:
    return DesignBrief(
        concept="a gentle farewell to a black labrador",
        product_type=product_type,
        specific_products=[],
        audience="dog owners",
        visual_style=visual_style,
        confidence=80,
        reasoning="test",
        source_seed_ids=[],
        layout_type="mid_ribbon",
        headline_text=headline_text,
        tagline_text=tagline_text,
        copy_mode=copy_mode,
        voice=voice,
        scene_description=scene_description,
        text_plan=TextPlan(
            method=method,
            carrier=carrier,
            reason=reason,
            ink=ink,
        ),
    )


class _FakeComfy:
    def __init__(self, results: list[object]) -> None:
        self._results = list(results)
        self.calls: list[dict[str, Any]] = []

    def generate_qwen_image(self, **kwargs: Any) -> ComfyUIResult:
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
            observed="Forever In My Heart" if matched else "Forevr In My Hart",
            score=1.0 if matched else 0.0,
        )


class TestDecision:
    def test_a_painted_carrier_free_brief_uses_the_model(self) -> None:
        assert is_model_text_brief(_brief()) is True

    def test_a_painted_plan_with_a_carrier_keeps_the_compositor(self) -> None:
        assert is_model_text_brief(_brief(carrier="headstone")) is False

    def test_a_printed_plan_keeps_the_compositor(self) -> None:
        assert is_model_text_brief(_brief(method="printed")) is False

    def test_a_no_text_brief_never_uses_the_model(self) -> None:
        assert is_model_text_brief(_brief(copy_mode="no_text")) is False

    def test_a_brief_without_a_plan_keeps_the_compositor(self) -> None:
        brief = replace(_brief(), text_plan=None)
        assert is_model_text_brief(brief) is False

    def test_a_brief_without_a_headline_never_uses_the_model(self) -> None:
        assert is_model_text_brief(_brief(headline_text=None)) is False


class TestCopy:
    def test_carries_the_headline_and_the_plan_style(self) -> None:
        copy = model_text_copy_from_brief(_brief())

        assert copy.headline == "Forever In My Heart"
        assert copy.descriptor == "All Souls Day"
        assert "soft, warm, hand-painted" in copy.style
        assert "vintage screenprint" in copy.style
        assert "painted directly into the soft, glowing smoke" in copy.integration
        assert "light-colored" in copy.ink
        assert "solid pure magenta" in copy.background

    def test_a_sublimation_brief_gets_a_white_background(self) -> None:
        copy = model_text_copy_from_brief(_brief(product_type="sublimation"))

        assert "plain white background" in copy.background

    def test_text_blocks_are_the_headline_only(self) -> None:
        copy = model_text_copy_from_brief(_brief())

        assert copy.text_blocks == ("Forever In My Heart",)


class TestPrompt:
    def test_a_dtf_text_design_has_no_default_sticker_border(self) -> None:
        copy = model_text_copy_from_brief(_brief())

        prompt = build_model_text_prompt(copy)

        assert copy.sticker is False
        assert "screen-printed apparel graphic" in prompt.lower()
        assert "no white sticker border" in prompt.lower()
        assert "die-cut sticker" not in prompt.lower()

    def test_a_dtf_text_design_gets_the_no_sticker_negative(self) -> None:
        comfy = _FakeComfy([ComfyUIResult(image_bytes=b"good")])
        generator = ModelTextGenerator(
            comfyui_client=comfy, validator=_FakeValidator([True]), candidates=1
        )

        generator.generate(model_text_copy_from_brief(_brief()), seed=1)

        assert "white sticker border" in comfy.calls[0]["negative_prompt"]

    def test_carries_the_line_the_scene_and_the_style(self) -> None:
        prompt = build_model_text_prompt(model_text_copy_from_brief(_brief()))

        assert 'the words "Forever In My Heart"' in prompt
        assert "black labrador silhouette" in prompt
        assert "soft, warm, hand-painted" in prompt
        assert "painted directly into the soft, glowing smoke" in prompt
        assert "fully visible and readable" in prompt
        assert "no other text" in prompt
        assert "no quotation marks" in prompt
        assert "no extra punctuation" in prompt

    def test_never_writes_the_descriptor(self) -> None:
        prompt = build_model_text_prompt(model_text_copy_from_brief(_brief()))

        assert "All Souls Day" not in prompt

    def test_uses_the_dtf_magenta_background(self) -> None:
        prompt = build_model_text_prompt(model_text_copy_from_brief(_brief()))

        assert "solid pure magenta background" in prompt
        assert "plain white background" not in prompt


class TestEnabled:
    def test_off_by_default(self, monkeypatch: Any) -> None:
        monkeypatch.delenv("MODEL_TEXT_ENABLED", raising=False)
        assert model_text_enabled() is False


class TestGenerator:
    def test_retries_a_misspelling_and_keeps_the_good_render(self) -> None:
        comfy = _FakeComfy(
            [
                ComfyUIResult(image_bytes=b"bad", seed=1),
                ComfyUIResult(image_bytes=b"good", seed=2),
            ]
        )
        validator = _FakeValidator([False, True])
        generator = ModelTextGenerator(comfyui_client=comfy, validator=validator, candidates=2)

        result = generator.generate(model_text_copy_from_brief(_brief()), seed=10)

        assert result.image_bytes == b"good"
        assert result.attempts == 2
        assert result.error is None
        assert validator.calls == [
            (b"bad", ("Forever In My Heart",)),
            (b"good", ("Forever In My Heart",)),
        ]
        assert comfy.calls[0]["seed"] == 10
        assert comfy.calls[1]["seed"] == 11

    def test_refuses_when_every_candidate_misses(self) -> None:
        comfy = _FakeComfy([ComfyUIResult(image_bytes=b"bad")])
        validator = _FakeValidator([False])
        generator = ModelTextGenerator(comfyui_client=comfy, validator=validator, candidates=1)

        result = generator.generate(model_text_copy_from_brief(_brief()), seed=10)

        assert result.image_bytes is None
        assert "text gate" in (result.error or "")

    def test_survives_a_comfyui_error_and_tries_the_next_candidate(self) -> None:
        comfy = _FakeComfy([ComfyUIError("boom"), ComfyUIResult(image_bytes=b"good")])
        validator = _FakeValidator([True])
        generator = ModelTextGenerator(comfyui_client=comfy, validator=validator, candidates=2)

        result = generator.generate(model_text_copy_from_brief(_brief()), seed=1)

        assert result.image_bytes == b"good"
        assert result.attempts == 2

    def test_a_bare_copy_still_builds_a_prompt(self) -> None:
        prompt = build_model_text_prompt(ModelTextCopy(headline="HELLO", subject="a quiet cat"))

        assert 'the words "HELLO"' in prompt
        assert "a quiet cat" in prompt
