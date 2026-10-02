"""Unit tests for periodic design brief reranking."""

import json
from datetime import date
from unittest.mock import MagicMock

from store_listing.orchestration.brief_reranking import (
    RerankableBrief,
    build_rerank_prompt,
    parse_rerank_response,
    rerank_briefs,
)


def _rerank_response(
    evaluations: list[dict] | None = None,
) -> str:
    if evaluations is None:
        evaluations = [
            {
                "brief_id": 1,
                "new_confidence": 75,
                "direction": "down",
                "reasoning": "Trend momentum has slowed significantly",
            },
            {
                "brief_id": 2,
                "new_confidence": 95,
                "direction": "up",
                "reasoning": "Search volume surging ahead of holiday season",
            },
        ]
    return json.dumps({"evaluations": evaluations})


SAMPLE_BRIEFS = [
    RerankableBrief(
        id=1,
        concept="Spooky Season Skeleton Tee",
        product_type="dtf_apparel",
        audience="Halloween enthusiasts",
        confidence=85,
        original_confidence=85,
        reasoning="Halloween trending strongly",
        batch_id="2026-09-15",
    ),
    RerankableBrief(
        id=2,
        concept="Thanksgiving Turkey Humor Mug",
        product_type="sublimation",
        audience="Thanksgiving hosts",
        confidence=70,
        original_confidence=70,
        reasoning="Thanksgiving approaching",
        batch_id="2026-09-20",
    ),
]

SAMPLE_TRENDS = [
    {"query": "halloween costume", "score": 9500, "delta": 2000, "source": "google"},
    {"query": "thanksgiving decor", "score": 4200, "delta": 800, "source": "google"},
    {"query": "funny t-shirt", "score": 3100, "delta": -200, "source": "google"},
]


class TestBuildRerankPrompt:
    def test_includes_brief_details(self) -> None:
        messages = build_rerank_prompt(SAMPLE_BRIEFS, SAMPLE_TRENDS, date(2026, 10, 2))
        user_msg = messages[-1]["content"]
        assert "Spooky Season Skeleton Tee" in user_msg
        assert "Thanksgiving Turkey Humor Mug" in user_msg

    def test_includes_trend_signals(self) -> None:
        messages = build_rerank_prompt(SAMPLE_BRIEFS, SAMPLE_TRENDS, date(2026, 10, 2))
        user_msg = messages[-1]["content"]
        assert "halloween costume" in user_msg
        assert "thanksgiving decor" in user_msg

    def test_includes_current_date(self) -> None:
        messages = build_rerank_prompt(SAMPLE_BRIEFS, SAMPLE_TRENDS, date(2026, 10, 2))
        user_msg = messages[-1]["content"]
        assert "2026-10-02" in user_msg

    def test_system_prompt_instructs_reranking(self) -> None:
        messages = build_rerank_prompt(SAMPLE_BRIEFS, SAMPLE_TRENDS, date(2026, 10, 2))
        system_msg = messages[0]["content"]
        assert "new_confidence" in system_msg
        assert "brief_id" in system_msg

    def test_returns_system_and_user_messages(self) -> None:
        messages = build_rerank_prompt(SAMPLE_BRIEFS, SAMPLE_TRENDS, date(2026, 10, 2))
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"

    def test_includes_original_confidence(self) -> None:
        messages = build_rerank_prompt(SAMPLE_BRIEFS, SAMPLE_TRENDS, date(2026, 10, 2))
        user_msg = messages[-1]["content"]
        assert "85" in user_msg
        assert "70" in user_msg


class TestParseRerankResponse:
    def test_parses_valid_response(self) -> None:
        result = parse_rerank_response(_rerank_response())
        assert len(result) == 2
        assert result[0]["brief_id"] == 1
        assert result[0]["new_confidence"] == 75
        assert result[0]["direction"] == "down"

    def test_parses_up_direction(self) -> None:
        result = parse_rerank_response(_rerank_response())
        assert result[1]["brief_id"] == 2
        assert result[1]["new_confidence"] == 95
        assert result[1]["direction"] == "up"

    def test_returns_empty_on_invalid_json(self) -> None:
        assert parse_rerank_response("not json") == []

    def test_returns_empty_on_missing_evaluations(self) -> None:
        assert parse_rerank_response('{"other": "data"}') == []

    def test_skips_entries_without_brief_id(self) -> None:
        result = parse_rerank_response(
            _rerank_response([{"new_confidence": 50, "direction": "down", "reasoning": "test"}])
        )
        assert len(result) == 0

    def test_clamps_confidence_to_100(self) -> None:
        result = parse_rerank_response(
            _rerank_response(
                [{"brief_id": 1, "new_confidence": 150, "direction": "up", "reasoning": "test"}]
            )
        )
        assert result[0]["new_confidence"] == 100

    def test_clamps_confidence_to_0(self) -> None:
        result = parse_rerank_response(
            _rerank_response(
                [{"brief_id": 1, "new_confidence": -10, "direction": "down", "reasoning": "test"}]
            )
        )
        assert result[0]["new_confidence"] == 0

    def test_defaults_direction_to_hold(self) -> None:
        result = parse_rerank_response(
            _rerank_response([{"brief_id": 1, "new_confidence": 85, "reasoning": "no change"}])
        )
        assert result[0]["direction"] == "hold"

    def test_handles_markdown_wrapped_json(self) -> None:
        raw = "```json\n" + _rerank_response() + "\n```"
        result = parse_rerank_response(raw)
        assert len(result) == 2

    def test_includes_reasoning(self) -> None:
        result = parse_rerank_response(_rerank_response())
        assert "slowed" in result[0]["reasoning"]
        assert "surging" in result[1]["reasoning"]


class TestRerankBriefs:
    def _mock_db(self) -> MagicMock:
        db = MagicMock()
        db.list_briefs_for_reranking.return_value = SAMPLE_BRIEFS
        db.list_top_active_seed_signals.return_value = SAMPLE_TRENDS
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

    def test_updates_confidence_for_each_evaluation(self) -> None:
        db = self._mock_db()
        llm = self._mock_llm(_rerank_response())
        result = rerank_briefs(db, llm, "model-x")
        assert result["status"] == "success"
        assert result["reranked"] == 2
        assert db.update_brief_confidence.call_count == 2

    def test_passes_correct_values_to_db_update(self) -> None:
        db = self._mock_db()
        llm = self._mock_llm(_rerank_response())
        rerank_briefs(db, llm, "model-x")
        calls = db.update_brief_confidence.call_args_list
        assert calls[0].kwargs["brief_id"] == 1
        assert calls[0].kwargs["confidence"] == 75
        assert calls[1].kwargs["brief_id"] == 2
        assert calls[1].kwargs["confidence"] == 95

    def test_only_updates_known_brief_ids(self) -> None:
        db = self._mock_db()
        response = _rerank_response(
            [
                {"brief_id": 1, "new_confidence": 60, "direction": "down", "reasoning": "fading"},
                {
                    "brief_id": 999,
                    "new_confidence": 50,
                    "direction": "down",
                    "reasoning": "unknown",
                },
            ]
        )
        llm = self._mock_llm(response)
        result = rerank_briefs(db, llm, "model-x")
        assert result["reranked"] == 1
        assert db.update_brief_confidence.call_count == 1

    def test_returns_zero_when_no_briefs(self) -> None:
        db = self._mock_db()
        db.list_briefs_for_reranking.return_value = []
        llm = MagicMock()
        result = rerank_briefs(db, llm, "model-x")
        assert result["status"] == "success"
        assert result["reranked"] == 0
        llm.with_raw_response.create.assert_not_called()

    def test_returns_error_on_llm_failure(self) -> None:
        db = self._mock_db()
        llm = MagicMock()
        llm.with_raw_response.create.side_effect = RuntimeError("API down")
        result = rerank_briefs(db, llm, "model-x")
        assert result["status"] == "error"
        db.update_brief_confidence.assert_not_called()

    def test_returns_error_on_empty_choices(self) -> None:
        db = self._mock_db()
        llm = MagicMock()
        raw_response = MagicMock()
        response = MagicMock()
        response.choices = []
        raw_response.parse.return_value = response
        raw_response.headers = {}
        llm.with_raw_response.create.return_value = raw_response
        result = rerank_briefs(db, llm, "model-x")
        assert result["status"] == "error"

    def test_returns_error_on_unparseable_response(self) -> None:
        db = self._mock_db()
        llm = self._mock_llm("not valid json at all")
        result = rerank_briefs(db, llm, "model-x")
        assert result["status"] == "success"
        assert result["reranked"] == 0
