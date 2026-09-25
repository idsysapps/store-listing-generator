"""OpenAI-compatible LLM client for seed curation.

Uses OpenRouter by default; swap to local vLLM by changing LLM_BASE_URL.
"""

import os
from typing import Final

from openai import OpenAI

DEFAULT_BASE_URL: Final[str] = "https://openrouter.ai/api/v1"
DEFAULT_MODEL: Final[str] = "nvidia/nemotron-3-ultra-550b-a55b:free"


def configured() -> bool:
    return bool(os.environ.get("LLM_API_KEY", ""))


def get_llm_client() -> OpenAI:
    return OpenAI(
        base_url=os.environ.get("LLM_BASE_URL", DEFAULT_BASE_URL),
        api_key=os.environ.get("LLM_API_KEY", ""),
    )


def get_llm_model() -> str:
    return os.environ.get("LLM_MODEL", DEFAULT_MODEL)
