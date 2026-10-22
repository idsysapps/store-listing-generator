"""Tests for the model getters.

An empty environment variable is "not configured", not "a model called the
empty string". The chart ships the model names empty when a cluster does not
override them, and `os.environ.get(name, default)` lets the empty value win
over the default: the artwork gate then asked the proxy for the model "" and
got a 400 on every image, so the gate was off (issue #213).
"""

from __future__ import annotations

import os
from unittest.mock import patch

from store_listing.orchestration.llm_client import (
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    DEFAULT_VISION_MODEL,
    get_artwork_vision_model,
    get_llm_client,
    get_llm_curation_model,
    get_llm_model,
    get_vision_client,
    get_vision_model,
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
