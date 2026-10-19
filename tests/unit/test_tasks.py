"""Tests for the image generation Celery task wiring."""

from __future__ import annotations

import importlib
import os
from unittest.mock import MagicMock, patch

import pytest

from store_listing.orchestration.llm_client import get_vision_client, get_vision_model


@pytest.fixture
def tasks():
    module = importlib.import_module("store_listing.orchestration.tasks")
    importlib.reload(module)
    return module


def _run_task(
    tasks,
    task_name: str,
    env: dict[str, str],
    args: tuple[object, ...] = (),
) -> tuple[MagicMock, MagicMock]:
    """Run an image task with the pipeline and boto3 patched out."""
    pipeline = MagicMock()
    boto_client = MagicMock()
    with (
        patch.dict(os.environ, env),
        patch("store_listing.orchestration.image_pipeline.generate_images_for_briefs", pipeline),
        patch("store_listing.orchestration.image_pipeline.generate_image_for_brief", pipeline),
        patch("boto3.client", boto_client),
    ):
        getattr(tasks, task_name).run(*args)
    return pipeline, boto_client


class TestGenerateDesignImagesTask:
    def test_builds_the_comfyui_client_from_the_environment(self, tasks) -> None:
        pipeline, _ = _run_task(
            tasks,
            "generate_design_images_task",
            {"COMFYUI_BASE_URL": "http://dgx.local:8188"},
        )

        client = pipeline.call_args.kwargs["comfyui_client"]
        assert client.base_url == "http://dgx.local:8188"

    def test_defaults_to_localhost(self, tasks) -> None:
        env = {key: value for key, value in os.environ.items() if key != "COMFYUI_BASE_URL"}
        pipeline, _ = _run_task(tasks, "generate_design_images_task", env)

        client = pipeline.call_args.kwargs["comfyui_client"]
        assert client.base_url == "http://localhost:8188"

    def test_passes_a_text_validator_to_the_pipeline(self, tasks) -> None:
        pipeline, _ = _run_task(tasks, "generate_design_images_task", {})

        assert pipeline.call_args.kwargs["text_validator"] is not None

    def test_does_not_create_a_bedrock_runtime_client(self, tasks) -> None:
        _, boto_client = _run_task(tasks, "generate_design_images_task", {})

        services = [call.args[0] for call in boto_client.call_args_list]
        assert services == ["s3"]

    def test_does_not_read_the_leonardo_api_key(self, tasks) -> None:
        pipeline, _ = _run_task(
            tasks, "generate_design_images_task", {"LEONARDO_API_KEY": "should-be-ignored"}
        )

        assert "leonardo_client" not in pipeline.call_args.kwargs


class TestGenerateSingleBriefImageTask:
    def test_builds_the_comfyui_client_from_the_environment(self, tasks) -> None:
        pipeline, _ = _run_task(
            tasks,
            "generate_single_brief_image_task",
            {"COMFYUI_BASE_URL": "http://dgx.local:8188"},
            args=(7,),
        )

        client = pipeline.call_args.kwargs["comfyui_client"]
        assert client.base_url == "http://dgx.local:8188"

    def test_passes_a_text_validator_to_the_pipeline(self, tasks) -> None:
        pipeline, _ = _run_task(tasks, "generate_single_brief_image_task", {}, args=(7,))

        assert pipeline.call_args.kwargs["text_validator"] is not None

    def test_does_not_create_a_bedrock_runtime_client(self, tasks) -> None:
        _, boto_client = _run_task(tasks, "generate_single_brief_image_task", {}, args=(7,))

        services = [call.args[0] for call in boto_client.call_args_list]
        assert services == ["s3"]


class TestVisionClient:
    def test_model_defaults_to_the_ocr_model(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            assert get_vision_model() == "qwen_vl_ocr"

    def test_model_can_be_overridden(self) -> None:
        with patch.dict(os.environ, {"VISION_LLM_MODEL": "my-vl"}):
            assert get_vision_model() == "my-vl"

    def test_base_url_falls_back_to_the_llm_base_url(self) -> None:
        with patch.dict(os.environ, {"LLM_BASE_URL": "http://proxy:9000/v1"}):
            client = get_vision_client()
        assert str(client.base_url).rstrip("/") == "http://proxy:9000/v1"

    def test_base_url_can_be_overridden(self) -> None:
        with patch.dict(os.environ, {"VISION_LLM_BASE_URL": "http://vision:8003/v1"}):
            client = get_vision_client()
        assert str(client.base_url).rstrip("/") == "http://vision:8003/v1"
