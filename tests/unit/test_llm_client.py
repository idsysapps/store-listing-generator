"""Tests for the model getters.

An empty environment variable is "not configured", not "a model called the
empty string". The chart ships the model names empty when a cluster does not
override them, and `os.environ.get(name, default)` lets the empty value win
over the default: the artwork gate then asked the proxy for the model "" and
got a 400 on every image, so the gate was off (issue #213).

The tests at the end cover the model check: the cluster pointed at a model the
DGX proxy does not serve, so every brief and curation call failed with a
generic LLM error and nothing said which model was missing (issue #194).
"""

from __future__ import annotations

import os
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from store_listing.orchestration.llm_client import (
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    DEFAULT_VISION_MODEL,
    LLM_MAX_TOKENS,
    check_model_available,
    create_answer,
    get_artwork_vision_model,
    get_llm_client,
    get_llm_curation_model,
    get_llm_model,
    get_vision_client,
    get_vision_model,
    is_model_missing_error,
    list_served_models,
    model_error_message,
    reasoning_body,
)


class TestModelGetters:
    def test_the_text_model_falls_back_to_the_default(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            assert get_llm_model() == DEFAULT_MODEL

    def test_an_empty_text_model_falls_back_to_the_default(self) -> None:
        with patch.dict(os.environ, {"LLM_MODEL": ""}):
            assert get_llm_model() == DEFAULT_MODEL

    def test_the_text_model_can_be_set(self) -> None:
        with patch.dict(os.environ, {"LLM_MODEL": "my-model"}):
            assert get_llm_model() == "my-model"

    def test_the_curation_model_follows_the_text_model(self) -> None:
        with patch.dict(os.environ, {"LLM_MODEL": "my-model"}, clear=True):
            assert get_llm_curation_model() == "my-model"

    def test_an_empty_curation_model_falls_back_to_the_text_model(self) -> None:
        with patch.dict(os.environ, {"LLM_CURATION_MODEL": "", "LLM_MODEL": "my-model"}):
            assert get_llm_curation_model() == "my-model"

    def test_the_curation_model_can_be_set(self) -> None:
        with patch.dict(os.environ, {"LLM_CURATION_MODEL": "my-curator"}):
            assert get_llm_curation_model() == "my-curator"

    def test_the_vision_model_falls_back_to_the_default(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            assert get_vision_model() == DEFAULT_VISION_MODEL

    def test_an_empty_vision_model_falls_back_to_the_default(self) -> None:
        with patch.dict(os.environ, {"VISION_LLM_MODEL": ""}):
            assert get_vision_model() == DEFAULT_VISION_MODEL

    def test_the_artwork_model_follows_the_vision_model(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            assert get_artwork_vision_model() == DEFAULT_VISION_MODEL

    def test_an_empty_artwork_model_falls_back_to_the_vision_model(self) -> None:
        # The chart ships this empty when a cluster does not set it, and the
        # artwork gate was asking for the model "" (issue #213).
        with patch.dict(os.environ, {"ARTWORK_VISION_MODEL": ""}):
            assert get_artwork_vision_model() == DEFAULT_VISION_MODEL

    def test_the_artwork_model_can_be_set(self) -> None:
        with patch.dict(os.environ, {"ARTWORK_VISION_MODEL": "my-judge"}):
            assert get_artwork_vision_model() == "my-judge"


class TestVisionClient:
    def test_the_base_url_falls_back_to_the_llm_base_url(self) -> None:
        with patch.dict(os.environ, {"LLM_BASE_URL": "http://proxy:9000/v1"}):
            client = get_vision_client()

        assert str(client.base_url).rstrip("/") == "http://proxy:9000/v1"

    def test_an_empty_base_url_falls_back_to_the_llm_base_url(self) -> None:
        with patch.dict(
            os.environ, {"VISION_LLM_BASE_URL": "", "LLM_BASE_URL": "http://proxy:9000/v1"}
        ):
            client = get_vision_client()

        assert str(client.base_url).rstrip("/") == "http://proxy:9000/v1"

    def test_an_empty_base_url_falls_back_to_the_default(self) -> None:
        with patch.dict(os.environ, {"VISION_LLM_BASE_URL": "", "LLM_BASE_URL": ""}):
            client = get_vision_client()

        assert str(client.base_url).rstrip("/") == DEFAULT_BASE_URL.rstrip("/")

    def test_the_vision_base_url_can_be_set(self) -> None:
        with patch.dict(os.environ, {"VISION_LLM_BASE_URL": "http://vision:9000/v1"}):
            client = get_vision_client()

        assert str(client.base_url).rstrip("/") == "http://vision:9000/v1"

    def test_an_empty_api_key_falls_back_to_a_placeholder(self) -> None:
        # The local proxy does not check the key, and the OpenAI client
        # refuses to start without one.
        with patch.dict(os.environ, {"VISION_LLM_API_KEY": "", "LLM_API_KEY": ""}):
            client = get_vision_client()

        assert client.api_key == "local-placeholder"


class TestLLMClient:
    def test_the_base_url_falls_back_to_the_default(self) -> None:
        with patch.dict(os.environ, {"LLM_BASE_URL": "", "LLM_API_KEY": "key"}):
            client = get_llm_client()

        assert str(client.base_url).rstrip("/") == DEFAULT_BASE_URL.rstrip("/")

    def test_the_base_url_can_be_set(self) -> None:
        with patch.dict(os.environ, {"LLM_BASE_URL": "http://proxy:9000/v1", "LLM_API_KEY": "key"}):
            client = get_llm_client()

        assert str(client.base_url).rstrip("/") == "http://proxy:9000/v1"


class _ApiError(Exception):
    """An error that carries an HTTP status, like the OpenAI SDK raises."""

    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code


def _client_serving(*model_ids: str) -> MagicMock:
    client = MagicMock()
    client.models.list.return_value = SimpleNamespace(
        data=[SimpleNamespace(id=model_id) for model_id in model_ids]
    )
    return client


class TestServedModels:
    def test_lists_the_model_ids_the_endpoint_serves(self) -> None:
        client = _client_serving("qwen38_briefs", "qwen_vl_ocr")

        assert list_served_models(client) == ["qwen38_briefs", "qwen_vl_ocr"]

    def test_an_endpoint_that_cannot_list_its_models_is_not_a_failure(self) -> None:
        client = MagicMock()
        client.models.list.side_effect = RuntimeError("no /models here")

        assert list_served_models(client) == []

    def test_a_model_the_endpoint_serves_is_reported_as_usable(self) -> None:
        client = _client_serving("qwen38_briefs")

        assert check_model_available(client, "qwen38_briefs") == ""

    def test_a_model_the_endpoint_does_not_serve_names_the_ones_it_does(self) -> None:
        # The cluster pointed at a model the proxy does not serve, so every
        # brief and curation call failed (issue #194).
        client = _client_serving("qwen38_briefs", "qwen_vl_ocr")

        message = check_model_available(client, "qwen2.5:32b-instruct-q4_K_M")

        assert "qwen2.5:32b-instruct-q4_K_M" in message
        assert "qwen38_briefs" in message
        assert "qwen_vl_ocr" in message

    def test_an_endpoint_without_a_model_list_cannot_confirm_the_model(self) -> None:
        # The check is advisory: not being able to ask is not a failure.
        client = MagicMock()
        client.models.list.side_effect = RuntimeError("no /models here")

        assert check_model_available(client, "qwen38_briefs") == ""


class TestModelErrors:
    def test_a_404_about_a_model_is_a_missing_model(self) -> None:
        assert is_model_missing_error(_ApiError(404, "Error code: 404 - model not found"))

    def test_the_proxy_unknown_model_message_is_a_missing_model(self) -> None:
        error = _ApiError(
            400,
            "Error code: 400 - {'error': {'message': "
            "\"Unknown model: qwen2.5. Available: ['qwen38_briefs', 'qwen_vl_ocr']\"}}",
        )

        assert is_model_missing_error(error)

    def test_a_rate_limit_is_not_a_missing_model(self) -> None:
        assert not is_model_missing_error(_ApiError(429, "Error code: 429 - rate limit"))

    def test_an_unrelated_error_is_not_a_missing_model(self) -> None:
        assert not is_model_missing_error(RuntimeError("API down"))

    def test_the_message_names_the_models_the_endpoint_serves(self) -> None:
        error = _ApiError(
            400,
            "Unknown model: qwen2.5. Available: ['qwen38_briefs', 'qwen_vl_ocr']",
        )

        message = model_error_message(error, "qwen2.5:32b-instruct-q4_K_M")

        assert "qwen2.5:32b-instruct-q4_K_M" in message
        assert "qwen38_briefs" in message
        assert "qwen_vl_ocr" in message

    def test_a_missing_model_without_a_list_still_names_the_model(self) -> None:
        error = _ApiError(404, "Error code: 404 - model not found")

        message = model_error_message(error, "qwen2.5:32b-instruct-q4_K_M")

        assert "qwen2.5:32b-instruct-q4_K_M" in message

    def test_other_errors_keep_their_own_message(self) -> None:
        error = _ApiError(429, "Error code: 429 - rate limit exceeded")

        assert model_error_message(error, "qwen38_briefs") == str(error)


class TestTheThinkingOfAModel:
    """Qwen3 thinks before it answers, and the thinking costs tokens.

    The briefs model is Qwen3-14B with the reasoning parser on. Its thinking
    comes out of the same budget as the answer, so a brief that is JSON gets
    a few hundred tokens of answer and is cut in the middle (issue #232).
    """

    def test_a_qwen_model_is_asked_not_to_think(self) -> None:
        assert reasoning_body("qwen38_briefs") == {
            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}}
        }

    def test_a_model_that_does_not_think_is_left_alone(self) -> None:
        assert reasoning_body("llama-3.3-70b-versatile") == {}

    def test_the_model_name_is_matched_without_case(self) -> None:
        assert reasoning_body("Qwen/Qwen3-14B") != {}


class TestCreateAnswer:
    """A call that asks for no thinking, and the endpoint that refuses it."""

    def _answer(self) -> MagicMock:
        llm = MagicMock()
        raw_response = MagicMock()
        raw_response.headers = {}
        llm.with_raw_response.create.return_value = raw_response
        return llm

    def test_the_call_asks_for_no_thinking(self) -> None:
        llm = self._answer()

        raw, error = create_answer(
            llm, "qwen38_briefs", [{"role": "user", "content": "x"}], "briefs"
        )

        assert error is None
        assert raw is llm.with_raw_response.create.return_value
        call_kwargs = llm.with_raw_response.create.call_args.kwargs
        assert call_kwargs["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}
        assert call_kwargs["max_tokens"] == LLM_MAX_TOKENS

    def test_an_endpoint_that_does_not_know_the_field_is_asked_again(self) -> None:
        llm = self._answer()
        refused = MagicMock()
        refused.status_code = 400
        llm.with_raw_response.create.side_effect = [
            RuntimeError("chat_template_kwargs is not a field of the request"),
            llm.with_raw_response.create.return_value,
        ]

        raw, error = create_answer(
            llm, "qwen38_briefs", [{"role": "user", "content": "x"}], "briefs"
        )

        assert error is None
        assert raw is llm.with_raw_response.create.return_value
        assert llm.with_raw_response.create.call_count == 2
        assert "extra_body" not in llm.with_raw_response.create.call_args.kwargs

    def test_a_model_that_does_not_think_is_asked_once(self) -> None:
        llm = self._answer()
        llm.with_raw_response.create.side_effect = RuntimeError("boom")

        raw, error = create_answer(llm, "llama-3.3-70b-versatile", [], "briefs")

        assert raw is None
        assert error == "boom"
        assert llm.with_raw_response.create.call_count == 1

    def test_the_error_of_the_second_ask_is_reported(self) -> None:
        llm = self._answer()
        llm.with_raw_response.create.side_effect = RuntimeError("boom")

        raw, error = create_answer(llm, "qwen38_briefs", [], "briefs")

        assert raw is None
        assert error is not None
