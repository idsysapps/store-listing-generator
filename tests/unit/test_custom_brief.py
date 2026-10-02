"""Unit tests for custom design brief generation from user descriptions."""

import json
from unittest.mock import MagicMock

from store_listing.orchestration.custom_brief import (
    build_custom_brief_prompt,
    create_custom_brief,
    parse_custom_brief_response,
)


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
