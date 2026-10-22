"""Unit tests for custom design brief generation from user descriptions."""

import json
from unittest.mock import MagicMock

from store_listing.orchestration.custom_brief import (
    build_custom_brief_prompt,
    create_custom_brief,
    parse_custom_brief_response,
)


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

    def _mock_llm(self, content: str) -> MagicMock:
        llm = MagicMock()
        raw_response = MagicMock()
        response = MagicMock()
        choice = MagicMock()
        choice.message.content = content
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
