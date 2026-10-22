"""OpenAI-compatible LLM client for seed curation.

Uses Groq's free tier by default; swap to local vLLM by changing LLM_BASE_URL.
"""

import os
from typing import Final

from openai import OpenAI

DEFAULT_BASE_URL: Final[str] = "https://api.groq.com/openai/v1"
DEFAULT_MODEL: Final[str] = "llama-3.3-70b-versatile"
DEFAULT_VISION_MODEL: Final[str] = "qwen_vl_ocr"


def configured() -> bool:
    return bool(os.environ.get("LLM_API_KEY", ""))


def get_llm_client() -> OpenAI:
    return OpenAI(
        base_url=os.environ.get("LLM_BASE_URL") or DEFAULT_BASE_URL,
        api_key=os.environ.get("LLM_API_KEY", ""),
    )


def get_llm_model() -> str:
    return os.environ.get("LLM_MODEL") or DEFAULT_MODEL


def get_llm_curation_model() -> str:
    return os.environ.get("LLM_CURATION_MODEL") or get_llm_model()


def get_vision_client() -> OpenAI:
    """Client for the vision model that verifies the text rendered into an image.

    Defaults to the same OpenAI-compatible endpoint as the text LLM - on the DGX
    that is the local proxy, which routes the model to the right vLLM service.
    The local proxy does not check the key, so a placeholder is used when none
    is configured.

    An empty variable means "not configured": the chart ships the model names
    empty when a cluster does not override them, and an empty value must not
    win over the default (issue #213).
    """
    return OpenAI(
        base_url=(
            os.environ.get("VISION_LLM_BASE_URL")
            or os.environ.get("LLM_BASE_URL")
            or DEFAULT_BASE_URL
        ),
        api_key=(
            os.environ.get("VISION_LLM_API_KEY")
            or os.environ.get("LLM_API_KEY")
            or "local-placeholder"
        ),
    )


def get_vision_model() -> str:
    return os.environ.get("VISION_LLM_MODEL") or DEFAULT_VISION_MODEL


def get_artwork_vision_model() -> str:
    """The model that judges whether a render is artwork or a product.

    The OCR model reads text well, but judging a picture is a different job,
    so the model can be pointed at a general vision model instead.
    """
    return os.environ.get("ARTWORK_VISION_MODEL") or get_vision_model()
