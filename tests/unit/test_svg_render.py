"""Unit tests for LLM-driven SVG design generation from design briefs."""

from unittest.mock import MagicMock

from store_listing.rendering.svg_design import (
    DesignBriefInput,
    build_svg_prompt,
    generate_design_svgs,
    parse_svg_response,
    validate_svg,
)


def _brief(
    brief_id: int = 1,
    concept: str = "Dad Jokes Loading...",
    product_type: str = "dtf_apparel",
    specific_products: list[str] | None = None,
    visual_style: str = "bold slab serif, black on white, retro loading bar",
) -> DesignBriefInput:
    return DesignBriefInput(
        brief_id=brief_id,
        concept=concept,
        product_type=product_type,
        specific_products=specific_products or ["t-shirt"],
        visual_style=visual_style,
    )


MINIMAL_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 1000">'
    '<text x="400" y="500" text-anchor="middle" font-size="72"'
    ' font-family="sans-serif" fill="#000000">Dad Jokes Loading...</text>'
    "</svg>"
)


class TestBuildSvgPrompt:
    def test_returns_system_and_user_messages(self) -> None:
        messages = build_svg_prompt(_brief())

        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"

    def test_system_prompt_mentions_svg(self) -> None:
        messages = build_svg_prompt(_brief())

        assert "SVG" in messages[0]["content"]

    def test_user_message_contains_brief_fields(self) -> None:
        brief = _brief(concept="Funny Mug Quote", visual_style="script font, pastel palette")
        messages = build_svg_prompt(brief)

        user_content = messages[1]["content"]
        assert "Funny Mug Quote" in user_content
        assert "script font, pastel palette" in user_content

    def test_user_message_contains_product_type(self) -> None:
        brief = _brief(product_type="sublimation")
        messages = build_svg_prompt(brief)

        user_content = messages[1]["content"]
        assert "sublimation" in user_content

    def test_user_message_contains_specific_products(self) -> None:
        brief = _brief(specific_products=["mug", "tumbler"])
        messages = build_svg_prompt(brief)

        user_content = messages[1]["content"]
        assert "mug" in user_content
        assert "tumbler" in user_content


class TestValidateSvg:
    def test_accepts_well_formed_svg(self) -> None:
        assert validate_svg(MINIMAL_SVG) is True

    def test_rejects_non_xml(self) -> None:
        assert validate_svg("this is not xml") is False

    def test_rejects_non_svg_root(self) -> None:
        assert validate_svg("<div>not svg</div>") is False

    def test_rejects_empty_string(self) -> None:
        assert validate_svg("") is False

    def test_rejects_svg_with_script_tags(self) -> None:
        dangerous = (
            '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 1000">'
            '<script>alert("xss")</script>'
            "</svg>"
        )
        assert validate_svg(dangerous) is False

    def test_rejects_svg_with_event_handlers(self) -> None:
        dangerous = (
            '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 1000">'
            '<text onclick="alert(1)">test</text>'
            "</svg>"
        )
        assert validate_svg(dangerous) is False

    def test_accepts_svg_with_common_elements(self) -> None:
        svg = (
            '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 1000">'
            '<rect x="0" y="0" width="800" height="1000" fill="#ffffff"/>'
            '<text x="400" y="300" font-size="48" fill="#333">Hello</text>'
            '<circle cx="400" cy="600" r="50" fill="red"/>'
            '<path d="M10 10 L50 50" stroke="black"/>'
            "</svg>"
        )
        assert validate_svg(svg) is True


class TestParseSvgResponse:
    def test_extracts_svg_from_markdown_code_block(self) -> None:
        response = f"Here is the design:\n```svg\n{MINIMAL_SVG}\n```\nLet me know!"

        result = parse_svg_response(response)

        assert result is not None
        assert "<svg" in result
        assert "Dad Jokes Loading..." in result

    def test_extracts_svg_from_xml_code_block(self) -> None:
        response = f"```xml\n{MINIMAL_SVG}\n```"

        result = parse_svg_response(response)

        assert result is not None
        assert "<svg" in result

    def test_extracts_bare_svg_without_code_block(self) -> None:
        result = parse_svg_response(MINIMAL_SVG)

        assert result is not None
        assert "<svg" in result

    def test_extracts_svg_from_plain_code_block(self) -> None:
        response = f"```\n{MINIMAL_SVG}\n```"

        result = parse_svg_response(response)

        assert result is not None
        assert "<svg" in result

    def test_returns_none_for_no_svg(self) -> None:
        result = parse_svg_response("I cannot generate SVG right now.")

        assert result is None

    def test_returns_none_for_empty_string(self) -> None:
        assert parse_svg_response("") is None

    def test_returns_none_for_invalid_svg(self) -> None:
        result = parse_svg_response("```svg\n<svg><script>bad</script></svg>\n```")

        assert result is None


class TestGenerateDesignSvgs:
    def _mock_db(self, briefs: list[DesignBriefInput] | None = None) -> MagicMock:
        db = MagicMock()
        db.list_renderable_briefs.return_value = briefs or []
        db.insert_design_render.return_value = 1
        return db

    def _mock_llm(self, svg_content: str) -> MagicMock:
        llm = MagicMock()
        choice = MagicMock()
        choice.message.content = f"```svg\n{svg_content}\n```"
        llm.create.return_value = MagicMock(choices=[choice])
        return llm

    def test_no_briefs_returns_early(self) -> None:
        db = self._mock_db()
        llm = MagicMock()

        result = generate_design_svgs(db, llm, "test-model")

        assert result["status"] == "success"
        assert result["renders_created"] == 0
        assert not llm.create.called

    def test_generates_svg_from_brief(self) -> None:
        db = self._mock_db(briefs=[_brief(brief_id=42)])
        llm = self._mock_llm(MINIMAL_SVG)

        result = generate_design_svgs(db, llm, "test-model")

        assert result["status"] == "success"
        assert result["renders_created"] == 1
        db.insert_design_render.assert_called_once()
        call_kwargs = db.insert_design_render.call_args.kwargs
        assert call_kwargs["brief_id"] == 42
        assert "<svg" in call_kwargs["svg_content"]

    def test_skips_invalid_svg_response(self) -> None:
        db = self._mock_db(briefs=[_brief()])
        llm = MagicMock()
        choice = MagicMock()
        choice.message.content = "Sorry, I can't generate SVG."
        llm.create.return_value = MagicMock(choices=[choice])

        result = generate_design_svgs(db, llm, "test-model")

        assert result["renders_created"] == 0
        assert result["renders_failed"] == 1
        assert not db.insert_design_render.called

    def test_handles_llm_error_gracefully(self) -> None:
        db = self._mock_db(briefs=[_brief()])
        llm = MagicMock()
        llm.create.side_effect = RuntimeError("API down")

        result = generate_design_svgs(db, llm, "test-model")

        assert result["renders_created"] == 0
        assert result["renders_failed"] == 1

    def test_continues_after_one_failure(self) -> None:
        briefs = [_brief(brief_id=1), _brief(brief_id=2)]
        db = self._mock_db(briefs=briefs)

        call_count = 0

        def side_effect(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("Transient error")
            choice = MagicMock()
            choice.message.content = f"```svg\n{MINIMAL_SVG}\n```"
            return MagicMock(choices=[choice])

        llm = MagicMock()
        llm.create.side_effect = side_effect

        result = generate_design_svgs(db, llm, "test-model")

        assert result["renders_created"] == 1
        assert result["renders_failed"] == 1

    def test_passes_model_to_llm(self) -> None:
        db = self._mock_db(briefs=[_brief()])
        llm = self._mock_llm(MINIMAL_SVG)

        generate_design_svgs(db, llm, "my-model-id")

        call_kwargs = llm.create.call_args.kwargs
        assert call_kwargs["model"] == "my-model-id"

    def test_stores_llm_model_in_render(self) -> None:
        db = self._mock_db(briefs=[_brief()])
        llm = self._mock_llm(MINIMAL_SVG)

        generate_design_svgs(db, llm, "my-model-id")

        call_kwargs = db.insert_design_render.call_args.kwargs
        assert call_kwargs["llm_model"] == "my-model-id"
