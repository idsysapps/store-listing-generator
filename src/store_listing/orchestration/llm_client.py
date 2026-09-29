"""OpenAI-compatible LLM client for seed curation and SVG rendering.

Uses OpenRouter by default; swap to local vLLM by changing LLM_BASE_URL.
SVG rendering can use a separate, more code-capable model via SVG_LLM_*
env vars, falling back to the main LLM config when not set.
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


def svg_configured() -> bool:
    return bool(os.environ.get("SVG_LLM_API_KEY", "")) or configured()


def get_svg_llm_client() -> OpenAI:
    return OpenAI(
        base_url=os.environ.get(
            "SVG_LLM_BASE_URL",
            os.environ.get("LLM_BASE_URL", DEFAULT_BASE_URL),
        ),
        api_key=os.environ.get(
            "SVG_LLM_API_KEY",
            os.environ.get("LLM_API_KEY", ""),
        ),
    )


def get_svg_llm_model() -> str:
    return os.environ.get("SVG_LLM_MODEL", get_llm_model())
