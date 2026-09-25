"""OpenAI-compatible LLM client for seed curation.

Uses Groq's free tier by default; swap to local vLLM by changing LLM_BASE_URL.
"""

import os
from typing import Final

from openai import OpenAI

DEFAULT_BASE_URL: Final[str] = "https://api.groq.com/openai/v1"
DEFAULT_MODEL: Final[str] = "llama-3.3-70b-versatile"


def configured() -> bool:
    return bool(os.environ.get("LLM_API_KEY", ""))


def get_llm_client() -> OpenAI:
    return OpenAI(
        base_url=os.environ.get("LLM_BASE_URL", DEFAULT_BASE_URL),
        api_key=os.environ.get("LLM_API_KEY", ""),
    )


def get_llm_model() -> str:
    return os.environ.get("LLM_MODEL", DEFAULT_MODEL)
