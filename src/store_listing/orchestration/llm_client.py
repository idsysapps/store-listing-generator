"""OpenAI-compatible LLM client for seed curation.

Uses Groq's free tier by default; swap to local vLLM by changing LLM_BASE_URL.
"""

import os
import re
from collections.abc import Sequence
from typing import Final

from openai import OpenAI

DEFAULT_BASE_URL: Final[str] = "https://api.groq.com/openai/v1"
DEFAULT_MODEL: Final[str] = "llama-3.3-70b-versatile"
DEFAULT_VISION_MODEL: Final[str] = "qwen_vl_ocr"

# The DGX proxy answers an unknown model with
# "Unknown model: <name>. Available: ['a', 'b']" (deploy/dgx/openai-proxy.py).
_AVAILABLE_PATTERN = re.compile(r"Available:\s*\[(?P<models>[^\]]*)\]")


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


def list_served_models(client: OpenAI) -> list[str]:
    """The model ids the endpoint serves.

    The DGX proxy answers /v1/models with the models its routes serve, so a
    config that points at a model the proxy does not serve can be seen before
    the call is made (issue #194). An endpoint that cannot answer is not a
    failure: the list is simply empty.
    """
    try:
        response = client.models.list()
    except Exception:  # noqa: BLE001
        return []

    entries = getattr(response, "data", None)
    if not isinstance(entries, (list, tuple)):
        return []

    models: set[str] = set()
    for entry in entries:
        model_id = getattr(entry, "id", None)
        if model_id is None and isinstance(entry, dict):
            model_id = entry.get("id")
        if model_id:
            models.add(str(model_id))
    return sorted(models)


def is_model_missing_error(error: Exception) -> bool:
    """Whether the endpoint refused the call because it does not serve the model."""
    text = str(error).lower()
    if "model_not_found" in text or "unknown model" in text:
        return True
    status = getattr(error, "status_code", None)
    return status == 404 and "model" in text


def missing_model_message(model: str, served: Sequence[str]) -> str:
    """The message for a model the endpoint does not serve."""
    if served:
        return (
            f"the LLM endpoint does not serve the model '{model}'; it serves: {', '.join(served)}"
        )
    return f"the LLM endpoint does not serve the model '{model}'"


def _available_from_error(text: str) -> list[str]:
    """The models the endpoint named in the error message."""
    match = _AVAILABLE_PATTERN.search(text)
    if not match:
        return []
    return [
        model.strip().strip("'\"") for model in match.group("models").split(",") if model.strip()
    ]


def model_error_message(error: Exception, model: str) -> str:
    """The error message to report, naming the models the endpoint serves.

    A missing model used to read as a generic LLM failure, so nobody could
    see that the config pointed at a model the proxy does not serve
    (issue #194).
    """
    if not is_model_missing_error(error):
        return str(error)
    return missing_model_message(model, _available_from_error(str(error)))


def check_model_available(client: OpenAI, model: str) -> str:
    """Why the model cannot be used, or an empty string when it can.

    The check is advisory: an endpoint that does not answer /v1/models cannot
    confirm the model, and that is not a failure.
    """
    served = list_served_models(client)
    if not served or model in served:
        return ""
    return missing_model_message(model, served)
