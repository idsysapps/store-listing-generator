"""Unit tests for custom design brief generation from user descriptions."""

import json
import logging
from unittest.mock import MagicMock

import pytest

from store_listing.orchestration.custom_brief import (
    build_custom_brief_prompt,
    create_custom_brief,
    parse_custom_brief_response,
)
from store_listing.orchestration.llm_client import LLM_MAX_TOKENS


class _ApiError(Exception):
    """An error that carries an HTTP status, like the OpenAI SDK raises."""

    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code


def _brief_response(
    concept: str = "First Baseball Game Tee",
    product_type: str = "dtf_apparel",
    specific_products: list[str] | None = None,
    audience: str = "Parents of young children",
    visual_style: str = "playful cartoon style",
    confidence: int = 85,
    reasoning: str = "Personal milestone design",
    layout_type: str = "text_top",
    headline_text: str | None = "First Game Day!",
    tagline_text: str | None = None,
    font_color: str | None = "#FFFFFF",
    scene_description: str | None = "a small child in a baseball cap holding a glove",
    voice: str | None = None,
    copy_mode: str = "caption",
    kicker_text: str | None = None,
) -> str:
    return json.dumps(
        {
            "brief": {
                "concept": concept,
                "product_type": product_type,
                "specific_products": specific_products or ["t-shirts"],
                "audience": audience,
                "visual_style": visual_style,
                "confidence": confidence,
                "reasoning": reasoning,
                "layout_type": layout_type,
                "headline_text": headline_text,
                "tagline_text": tagline_text,
                "font_color": font_color,
                "scene_description": scene_description,
                "voice": voice,
                "copy_mode": copy_mode,
                "kicker_text": kicker_text,
            }
        }
    )


class TestBuildCustomBriefPrompt:
    def test_includes_user_description(self) -> None:
        desc = "shirt for my daughters first baseball game"
        messages = build_custom_brief_prompt(desc, "dtf_apparel")
        user_msg = messages[-1]["content"]
        assert desc in user_msg

    def test_includes_product_type(self) -> None:
        messages = build_custom_brief_prompt("test idea", "sublimation")
        user_msg = messages[-1]["content"]
        assert "sublimation" in user_msg

    def test_system_prompt_mentions_single_brief(self) -> None:
        messages = build_custom_brief_prompt("test idea", "dtf_apparel")
        system_msg = messages[0]["content"]
        assert "single" in system_msg.lower() or "one" in system_msg.lower()

    def test_system_prompt_includes_layout_fields(self) -> None:
        messages = build_custom_brief_prompt("test idea", "dtf_apparel")
        system_msg = messages[0]["content"]
        assert "headline_text" in system_msg
        assert "scene_description" in system_msg
        assert "layout_type" in system_msg

    def test_system_prompt_asks_for_print_artwork_not_a_product(self) -> None:
        # The custom brief prompt never had the print-artwork rule, so a
        # description of a bag design produced an image of a bag (issue #195).
        messages = build_custom_brief_prompt("a bag for halloween", "dtf_apparel")
        system_msg = messages[0]["content"].lower()

        assert "not a photo of a product" in system_msg
        assert "never mention a bag" in system_msg

    def test_system_prompt_says_the_concept_is_not_the_product(self) -> None:
        messages = build_custom_brief_prompt("a bag for halloween", "dtf_apparel")
        system_msg = messages[0]["content"].lower()

        assert "not the product" in system_msg

    def test_system_prompt_offers_a_typography_only_layout(self) -> None:
        messages = build_custom_brief_prompt("just a slogan", "dtf_apparel")

        assert "text_only" in messages[0]["content"]

    def test_system_prompt_offers_the_three_copy_modes(self) -> None:
        messages = build_custom_brief_prompt("a groovy ghost", "dtf_apparel")

        system = messages[0]["content"]
        assert "copy_mode" in system
        assert "type_hero" in system
        assert "caption" in system
        assert "no_text" in system
        assert "kicker_text" in system

    def test_system_prompt_asks_for_sellable_copy(self) -> None:
        messages = build_custom_brief_prompt("a groovy ghost", "dtf_apparel")

        system = messages[0]["content"]
        assert "would a stranger wear" in system.lower()
        assert "Through A Kitchen" in system
        assert "Kitchen Commute" in system

    def test_returns_system_and_user_messages(self) -> None:
        messages = build_custom_brief_prompt("test", "dtf_apparel")
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"


class TestParseCustomBriefResponse:
    def test_parses_valid_response(self) -> None:
        brief = parse_custom_brief_response(_brief_response())
        assert brief is not None
        assert brief.concept == "First Baseball Game Tee"
        assert brief.product_type == "dtf_apparel"

    def test_parses_headline_and_layout(self) -> None:
        brief = parse_custom_brief_response(
            _brief_response(layout_type="text_top", headline_text="Game Day!")
        )
        assert brief is not None
        assert brief.layout_type == "text_top"
        assert brief.headline_text == "Game Day!"

    def test_parses_scene_description(self) -> None:
        brief = parse_custom_brief_response(
            _brief_response(scene_description="a baseball diamond at sunset")
        )
        assert brief is not None
        assert brief.scene_description == "a baseball diamond at sunset"

    def test_a_scene_that_asks_for_words_loses_them(self) -> None:
        # The image model cannot spell, so a scene that quotes the words to
        # draw comes back misspelled (issue #208).
        brief = parse_custom_brief_response(
            _brief_response(
                scene_description="a mascot with a speech bubble saying 'GO TEAM' in the stands"
            )
        )
        assert brief is not None
        assert brief.scene_description is not None
        assert "GO TEAM" not in brief.scene_description
        assert "speech bubble" not in brief.scene_description
        assert "mascot" in brief.scene_description

    def test_returns_none_on_invalid_json(self) -> None:
        assert parse_custom_brief_response("not json at all") is None

    def test_returns_none_on_empty_concept(self) -> None:
        assert parse_custom_brief_response(_brief_response(concept="")) is None

    def test_returns_none_on_invalid_product_type(self) -> None:
        assert parse_custom_brief_response(_brief_response(product_type="invalid")) is None

    def test_clamps_confidence(self) -> None:
        brief = parse_custom_brief_response(_brief_response(confidence=150))
        assert brief is not None
        assert brief.confidence == 100

    def test_defaults_layout_to_full_bleed(self) -> None:
        brief = parse_custom_brief_response(_brief_response(layout_type="invalid"))
        assert brief is not None
        assert brief.layout_type == "full_bleed"

    def test_handles_markdown_wrapped_json(self) -> None:
        raw = "```json\n" + _brief_response() + "\n```"
        brief = parse_custom_brief_response(raw)
        assert brief is not None
        assert brief.concept == "First Baseball Game Tee"

    def test_parses_specific_products(self) -> None:
        brief = parse_custom_brief_response(
            _brief_response(specific_products=["t-shirts", "hoodies"])
        )
        assert brief is not None
        assert brief.specific_products == ["t-shirts", "hoodies"]

    def test_parses_tagline(self) -> None:
        brief = parse_custom_brief_response(
            _brief_response(
                layout_type="text_top_bottom",
                headline_text="Game Day!",
                tagline_text="Her first swing",
            )
        )
        assert brief is not None
        assert brief.tagline_text == "Her first swing"

    def test_parses_a_hero_copy_mode_and_kicker(self) -> None:
        brief = parse_custom_brief_response(
            _brief_response(
                concept="Groovy Ghost",
                layout_type="auto",
                copy_mode="type_hero",
                headline_text="BOO JEE",
                kicker_text="IN MY SPOOKY ERA",
                scene_description="a ghost weaving through the letters",
            )
        )
        assert brief is not None
        assert brief.copy_mode == "type_hero"
        assert brief.kicker_text == "IN MY SPOOKY ERA"
        assert brief.layout_type == "auto"

    def test_parses_a_no_text_brief_without_copy(self) -> None:
        brief = parse_custom_brief_response(
            _brief_response(
                concept="Ghost In A Pumpkin",
                layout_type="text_top",
                copy_mode="no_text",
                headline_text="BOO",
                tagline_text="A GHOST",
                kicker_text="SPOOKY",
                scene_description="a ghost hiding inside a pumpkin",
            )
        )
        assert brief is not None
        assert brief.copy_mode == "no_text"
        assert brief.layout_type == "full_bleed"
        assert brief.headline_text is None
        assert brief.tagline_text is None
        assert brief.kicker_text is None

    def test_a_kicker_on_a_caption_is_dropped(self) -> None:
        brief = parse_custom_brief_response(
            _brief_response(
                copy_mode="caption",
                headline_text="BOO JEE",
                kicker_text="IN MY SPOOKY ERA",
            )
        )
        assert brief is not None
        assert brief.copy_mode == "caption"
        assert brief.kicker_text is None

    def test_parses_a_typography_only_brief(self) -> None:
        brief = parse_custom_brief_response(
            _brief_response(
                concept="First Game Day",
                layout_type="text_only",
                headline_text="First Game Day",
                scene_description=None,
            )
        )
        assert brief is not None
        assert brief.layout_type == "text_only"
        assert brief.scene_description is None

    def test_a_typography_only_brief_falls_back_to_a_short_concept(self) -> None:
        brief = parse_custom_brief_response(
            _brief_response(
                concept="First Game Day",
                layout_type="text_only",
                headline_text=None,
                scene_description=None,
            )
        )
        assert brief is not None
        assert brief.headline_text == "First Game Day"

    def test_a_typography_only_brief_without_a_usable_headline_is_rejected(self) -> None:
        brief = parse_custom_brief_response(
            _brief_response(
                concept="A concept that is far too long to be a headline for a shirt",
                layout_type="text_only",
                headline_text=None,
                scene_description=None,
            )
        )
        assert brief is None


class TestCreateCustomBrief:
    def _mock_db(self) -> MagicMock:
        db = MagicMock()
        db.insert_design_brief.return_value = 42
        return db

    def _mock_llm(self, content: str, finish_reason: str = "stop") -> MagicMock:
        llm = MagicMock()
        raw_response = MagicMock()
        response = MagicMock()
        choice = MagicMock()
        choice.message.content = content
        choice.finish_reason = finish_reason
        response.choices = [choice]
        raw_response.parse.return_value = response
        raw_response.headers = {}
        llm.with_raw_response.create.return_value = raw_response
        return llm

    def test_inserts_brief_and_returns_id(self) -> None:
        db = self._mock_db()
        llm = self._mock_llm(_brief_response())
        result = create_custom_brief(db, llm, "model-x", "shirt for baseball game", "dtf_apparel")
        assert result["status"] == "success"
        assert result["brief_id"] == 42
        db.insert_design_brief.assert_called_once()

    def test_the_copy_review_drops_a_scene_tagline(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LLM_COPY_REVIEW", "true")
        db = self._mock_db()
        brief_llm = self._mock_llm(
            _brief_response(
                concept="Night Train",
                layout_type="type_hero",
                headline_text="NIGHT TRAIN",
                kicker_text="GNOME ON THE MOVE",
                tagline_text="THROUGH A KITCHEN",
                scene_description="a garden gnome driving a night train through a kitchen",
                copy_mode="type_hero",
            )
        )
        review_llm = self._mock_llm(
            json.dumps(
                {
                    "reviews": [
                        {
                            "index": 1,
                            "design_stands_alone": True,
                            "headline": {
                                "action": "KEEP",
                                "rewrite": None,
                                "reason": "Reads as the main line.",
                            },
                            "kicker": {
                                "action": "KEEP",
                                "rewrite": None,
                                "reason": "Sets up the hero.",
                            },
                            "tagline": {
                                "action": "DROP",
                                "rewrite": None,
                                "reason": "The design stands alone.",
                            },
                            "overall_action": "KEEP",
                            "overall_reason": "One clear joke.",
                            "confidence": 0.9,
                        }
                    ]
                }
            )
        )
        llm = MagicMock()
        llm.with_raw_response.create.side_effect = [
            brief_llm.with_raw_response.create.return_value,
            review_llm.with_raw_response.create.return_value,
        ]

        result = create_custom_brief(db, llm, "model-x", "gnome train", "dtf_apparel")

        assert result["status"] == "success"
        assert db.insert_design_brief.call_args.kwargs["tagline_text"] is None
        assert llm.with_raw_response.create.call_count == 2

    def test_passes_correct_fields_to_db(self) -> None:
        db = self._mock_db()
        llm = self._mock_llm(
            _brief_response(
                concept="Baseball Tee",
                audience="Parents",
                visual_style="cartoon",
                confidence=85,
                headline_text="Game Day!",
                scene_description="child with bat",
            )
        )
        create_custom_brief(db, llm, "model-x", "baseball shirt", "dtf_apparel")
        call_kwargs = db.insert_design_brief.call_args
        assert call_kwargs.kwargs["concept"] == "Baseball Tee"
        assert call_kwargs.kwargs["audience"] == "Parents"
        assert call_kwargs.kwargs["headline_text"] == "Game Day!"
        assert call_kwargs.kwargs["scene_description"] == "child with bat"
        assert call_kwargs.kwargs["llm_model"] == "model-x"
        assert call_kwargs.kwargs["batch_id"] == "custom"
        assert call_kwargs.kwargs["copy_mode"] == "caption"
        assert call_kwargs.kwargs["kicker_text"] is None

    def test_stores_a_hero_copy_mode_and_kicker(self) -> None:
        db = self._mock_db()
        llm = self._mock_llm(
            _brief_response(
                concept="Groovy Ghost",
                layout_type="auto",
                copy_mode="type_hero",
                headline_text="BOO JEE",
                kicker_text="IN MY SPOOKY ERA",
                scene_description="a ghost weaving through the letters",
            )
        )

        create_custom_brief(db, llm, "model-x", "groovy ghost", "dtf_apparel")

        call_kwargs = db.insert_design_brief.call_args.kwargs
        assert call_kwargs["copy_mode"] == "type_hero"
        assert call_kwargs["kicker_text"] == "IN MY SPOOKY ERA"

    def test_returns_error_on_llm_failure(self) -> None:
        db = self._mock_db()
        llm = MagicMock()
        llm.with_raw_response.create.side_effect = RuntimeError("API down")
        result = create_custom_brief(db, llm, "model-x", "test idea", "dtf_apparel")
        assert result["status"] == "error"
        db.insert_design_brief.assert_not_called()

    def test_a_model_the_endpoint_does_not_serve_is_reported_clearly(self) -> None:
        # The cluster pointed at a model the DGX proxy does not serve, so
        # every custom brief call failed with a generic LLM error (issue #194).
        db = self._mock_db()
        llm = MagicMock()
        llm.with_raw_response.create.side_effect = _ApiError(
            400,
            "Unknown model: qwen2.5. Available: ['qwen38_briefs', 'qwen_vl_ocr']",
        )

        result = create_custom_brief(
            db, llm, "qwen2.5:32b-instruct-q4_K_M", "test idea", "dtf_apparel"
        )

        assert result["status"] == "error"
        assert "qwen38_briefs" in result["error"]
        db.insert_design_brief.assert_not_called()

    def test_reads_a_response_the_model_appended_something_to(self) -> None:
        # The model answered with the brief and then another object; the
        # parser read two values as one and lost the brief (issue #209).
        db = self._mock_db()
        llm = self._mock_llm(_brief_response() + "\n" + json.dumps({"notes": {"summary": "x"}}))

        result = create_custom_brief(db, llm, "model-x", "test idea", "dtf_apparel")

        assert result["status"] == "success"
        assert result["brief_id"] == 42

    def test_a_brief_of_a_type_that_is_not_in_production_is_coerced(self) -> None:
        # A store that focuses on DTF apparel still gets a brief, and it is a
        # DTF brief (issue #215).
        db = self._mock_db()
        llm = self._mock_llm(_brief_response(product_type="sublimation"))

        result = create_custom_brief(
            db,
            llm,
            "model-x",
            "test idea",
            "sublimation",
            enabled=["dtf_apparel"],
        )

        assert result["status"] == "success"
        assert db.insert_design_brief.call_args.kwargs["product_type"] == "dtf_apparel"

    def test_a_brief_of_a_type_that_is_not_in_production_is_refused(self) -> None:
        # With more than one type in production there is no single type to
        # fall back to, so the request is refused (issue #215).
        db = self._mock_db()
        llm = self._mock_llm(_brief_response())

        result = create_custom_brief(
            db,
            llm,
            "model-x",
            "test idea",
            "sublimation",
            enabled=["dtf_apparel", "sticker_vinyl"],
        )

        assert result["status"] == "error"
        db.insert_design_brief.assert_not_called()

    def test_the_prompt_offers_only_the_types_in_production(self) -> None:
        db = self._mock_db()
        llm = self._mock_llm(_brief_response())

        create_custom_brief(db, llm, "model-x", "test idea", "dtf_apparel", enabled=["dtf_apparel"])

        system = llm.with_raw_response.create.call_args.kwargs["messages"][0]["content"].lower()
        assert "sublimation" not in system
        assert "sticker_vinyl" not in system

    def test_returns_error_on_empty_choices(self) -> None:
        db = self._mock_db()
        llm = MagicMock()
        raw_response = MagicMock()
        response = MagicMock()
        response.choices = []
        raw_response.parse.return_value = response
        raw_response.headers = {}
        llm.with_raw_response.create.return_value = raw_response
        result = create_custom_brief(db, llm, "model-x", "test idea", "dtf_apparel")
        assert result["status"] == "error"

    def test_returns_error_on_unparseable_response(self) -> None:
        db = self._mock_db()
        llm = self._mock_llm("not valid json")
        result = create_custom_brief(db, llm, "model-x", "test idea", "dtf_apparel")
        assert result["status"] == "error"
        db.insert_design_brief.assert_not_called()

    def test_uses_temperature_zero(self) -> None:
        db = self._mock_db()
        llm = self._mock_llm(_brief_response())
        create_custom_brief(db, llm, "model-x", "test", "dtf_apparel")
        call_kwargs = llm.with_raw_response.create.call_args.kwargs
        assert call_kwargs.get("temperature") == 0

    def test_personal_use_false_by_default(self) -> None:
        db = self._mock_db()
        llm = self._mock_llm(_brief_response())
        result = create_custom_brief(db, llm, "model-x", "test", "dtf_apparel")
        assert result["personal_use"] is False
        call_kwargs = db.insert_design_brief.call_args.kwargs
        assert call_kwargs["personal_use"] is False

    def test_personal_use_passed_to_db(self) -> None:
        db = self._mock_db()
        llm = self._mock_llm(_brief_response())
        result = create_custom_brief(
            db, llm, "model-x", "disney themed shirt", "dtf_apparel", personal_use=True
        )
        assert result["status"] == "success"
        assert result["personal_use"] is True
        call_kwargs = db.insert_design_brief.call_args.kwargs
        assert call_kwargs["personal_use"] is True


class TestPersonalUsePrompt:
    def test_standard_prompt_forbids_ip(self) -> None:
        messages = build_custom_brief_prompt("test idea", "dtf_apparel", personal_use=False)
        system_msg = messages[0]["content"]
        assert "licensed IP" in system_msg or "Do not use licensed" in system_msg

    def test_personal_use_prompt_allows_ip(self) -> None:
        messages = build_custom_brief_prompt(
            "disney princess shirt", "dtf_apparel", personal_use=True
        )
        system_msg = messages[0]["content"]
        assert "PERSONAL USE ONLY" in system_msg
        assert "MAY reference" in system_msg

    def test_personal_use_prompt_does_not_forbid_ip(self) -> None:
        messages = build_custom_brief_prompt(
            "disney princess shirt", "dtf_apparel", personal_use=True
        )
        system_msg = messages[0]["content"]
        assert "Do not use licensed IP" not in system_msg

    def test_personal_use_still_returns_valid_structure(self) -> None:
        messages = build_custom_brief_prompt(
            "disney shirt for family trip", "dtf_apparel", personal_use=True
        )
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert "disney shirt for family trip" in messages[1]["content"]


class TestTheVoiceOfTheCustomBrief:
    def test_the_prompt_asks_for_a_voice(self) -> None:
        messages = build_custom_brief_prompt("test idea", "dtf_apparel")
        system_msg = messages[0]["content"]
        assert "voice" in system_msg.lower()
        for voice in ("deadpan", "loud", "cozy"):
            assert voice in system_msg.lower()

    def test_the_voice_is_parsed(self) -> None:
        response = _brief_response(headline_text="Gobble Til The Drawbridge Drops", voice="loud")

        brief = parse_custom_brief_response(response)

        assert brief is not None
        assert brief.voice == "loud"

    def test_a_headline_that_names_a_brand_is_dropped(self) -> None:
        # The design that started this (issue #225).
        response = _brief_response(headline_text="Thanksgiving in the Magic Kingdom")

        assert parse_custom_brief_response(response) is None

    def test_a_personal_use_brief_may_name_a_franchise(self) -> None:
        # A brief for personal use is allowed to be about a franchise.
        response = _brief_response(headline_text="A Weekend At The Magic Kingdom")

        brief = parse_custom_brief_response(response, personal_use=True)

        assert brief is not None
        assert brief.headline_text == "A Weekend At The Magic Kingdom"

    def test_a_noun_list_is_dropped(self) -> None:
        response = _brief_response(headline_text="Feast, fun, and fireworks")

        assert parse_custom_brief_response(response) is None


class TestACutAnswer:
    """A model that runs out of room answers half way (issue #232).

    The briefs model is Qwen3, and it thinks before it answers. The thinking
    comes out of the same token budget as the answer, so a brief that is cut
    off is a brief that is lost - and the log said the response could not be
    parsed, which is not what happened.
    """

    def _cut_answer(self) -> str:
        return (
            "{\n"
            '"brief": {\n'
            '"concept": "Thanksgiving in Disney World",\n'
            '"product_type": "dtf_apparel",\n'
            '"specific_products": ["t-shirts"],\n'
            '"audience": "Disney enthusiasts and families",\n'
            '"visual_style": "Vibrant, playful and nostalgic",\n'
            '"confidence": 90,\n'
            '"reasoning": "A festive mashup of two traditions.",\n'
            '"layout_type": "text_top",\n'
            '"headline_text": "Thanksgiving in Disney World",\n'
            '"tagline_text": null,\n'
            '"font_color": "#FFFFFF",\n'
            '"voice": "loud",\n'
            '"style_preset": "distressed_band_tee",'
        )

    def test_the_call_asks_the_model_for_no_thinking(self) -> None:
        db = MagicMock()
        llm = MagicMock()
        raw_response = MagicMock()
        response = MagicMock()
        choice = MagicMock()
        choice.message.content = _brief_response()
        choice.finish_reason = "stop"
        response.choices = [choice]
        raw_response.parse.return_value = response
        raw_response.headers = {}
        llm.with_raw_response.create.return_value = raw_response

        create_custom_brief(db, llm, "qwen38_briefs", "test idea", "dtf_apparel")

        call_kwargs = llm.with_raw_response.create.call_args.kwargs
        assert call_kwargs["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}

    def test_a_model_that_does_not_think_is_left_alone(self) -> None:
        db = MagicMock()
        llm = MagicMock()
        raw_response = MagicMock()
        response = MagicMock()
        choice = MagicMock()
        choice.message.content = _brief_response()
        choice.finish_reason = "stop"
        response.choices = [choice]
        raw_response.parse.return_value = response
        raw_response.headers = {}
        llm.with_raw_response.create.return_value = raw_response

        create_custom_brief(db, llm, "llama-3.3-70b-versatile", "test idea", "dtf_apparel")

        call_kwargs = llm.with_raw_response.create.call_args.kwargs
        assert "extra_body" not in call_kwargs

    def test_the_call_asks_for_a_token_budget(self) -> None:
        db = MagicMock()
        llm = MagicMock()
        raw_response = MagicMock()
        response = MagicMock()
        choice = MagicMock()
        choice.message.content = _brief_response()
        choice.finish_reason = "stop"
        response.choices = [choice]
        raw_response.parse.return_value = response
        raw_response.headers = {}
        llm.with_raw_response.create.return_value = raw_response

        create_custom_brief(db, llm, "qwen38_briefs", "test idea", "dtf_apparel")

        call_kwargs = llm.with_raw_response.create.call_args.kwargs
        assert call_kwargs["max_tokens"] == LLM_MAX_TOKENS

    def test_a_cut_answer_still_produces_a_brief(self) -> None:
        db = MagicMock()
        db.insert_design_brief.return_value = 42
        llm = MagicMock()
        raw_response = MagicMock()
        response = MagicMock()
        choice = MagicMock()
        choice.message.content = self._cut_answer()
        choice.finish_reason = "length"
        response.choices = [choice]
        raw_response.parse.return_value = response
        raw_response.headers = {}
        llm.with_raw_response.create.return_value = raw_response

        result = create_custom_brief(
            db,
            llm,
            "qwen38_briefs",
            "thanksgiving disney shirt",
            "dtf_apparel",
            personal_use=True,
        )

        assert result["status"] == "success"
        assert result["brief_id"] == 42
        call_kwargs = db.insert_design_brief.call_args.kwargs
        assert call_kwargs["concept"] == "Thanksgiving in Disney World"
        assert call_kwargs["voice"] == "loud"
        assert call_kwargs["style_preset"] == "distressed_band_tee"

    def test_a_brief_that_was_rejected_says_why(self) -> None:
        # The brief was read and dropped, which is not the same as an answer
        # that could not be read.
        db = MagicMock()
        llm = MagicMock()
        raw_response = MagicMock()
        response = MagicMock()
        choice = MagicMock()
        choice.message.content = _brief_response(headline_text="Thanksgiving in Disney World")
        choice.finish_reason = "stop"
        response.choices = [choice]
        raw_response.parse.return_value = response
        raw_response.headers = {}
        llm.with_raw_response.create.return_value = raw_response

        result = create_custom_brief(
            db, llm, "qwen38_briefs", "thanksgiving disney shirt", "dtf_apparel"
        )

        assert result["status"] == "error"
        assert "licensed brand" in result["error"]
        db.insert_design_brief.assert_not_called()

    def test_an_answer_without_json_says_it_could_not_be_read(self) -> None:
        db = MagicMock()
        llm = MagicMock()
        raw_response = MagicMock()
        response = MagicMock()
        choice = MagicMock()
        choice.message.content = "I am afraid I cannot do that."
        choice.finish_reason = "stop"
        response.choices = [choice]
        raw_response.parse.return_value = response
        raw_response.headers = {}
        llm.with_raw_response.create.return_value = raw_response

        result = create_custom_brief(db, llm, "qwen38_briefs", "test idea", "dtf_apparel")

        assert result["status"] == "error"
        assert result["error"] == "Failed to parse LLM response"

    def test_the_finish_reason_is_logged(self, caplog: pytest.LogCaptureFixture) -> None:
        db = MagicMock()
        llm = MagicMock()
        raw_response = MagicMock()
        response = MagicMock()
        choice = MagicMock()
        choice.message.content = self._cut_answer()
        choice.finish_reason = "length"
        response.choices = [choice]
        raw_response.parse.return_value = response
        raw_response.headers = {}
        llm.with_raw_response.create.return_value = raw_response

        with caplog.at_level(logging.INFO):
            create_custom_brief(
                db,
                llm,
                "qwen38_briefs",
                "thanksgiving disney shirt",
                "dtf_apparel",
                personal_use=True,
            )

        assert "finish=length" in caplog.text
