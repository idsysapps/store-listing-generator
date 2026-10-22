"""Tests for the image generation Celery task wiring."""

from __future__ import annotations

import importlib
import logging
import os
from types import SimpleNamespace
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
    unset: tuple[str, ...] = (),
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
        for key in unset:
            os.environ.pop(key, None)
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
        pipeline, _ = _run_task(
            tasks, "generate_design_images_task", {}, unset=("COMFYUI_BASE_URL",)
        )

        client = pipeline.call_args.kwargs["comfyui_client"]
        assert client.base_url == "http://localhost:8188"

    def test_passes_a_text_validator_to_the_pipeline(self, tasks) -> None:
        pipeline, _ = _run_task(tasks, "generate_design_images_task", {})

        assert pipeline.call_args.kwargs["text_validator"] is not None

    def test_passes_an_artwork_validator_to_the_pipeline(self, tasks) -> None:
        pipeline, _ = _run_task(tasks, "generate_design_images_task", {})

        assert pipeline.call_args.kwargs["artwork_validator"] is not None

    def test_the_artwork_validator_can_be_switched_off(self, tasks) -> None:
        pipeline, _ = _run_task(
            tasks, "generate_design_images_task", {"ARTWORK_VALIDATION_ENABLED": "false"}
        )

        assert pipeline.call_args.kwargs["artwork_validator"] is None

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

    def test_passes_an_artwork_validator_to_the_pipeline(self, tasks) -> None:
        pipeline, _ = _run_task(tasks, "generate_single_brief_image_task", {}, args=(7,))

        assert pipeline.call_args.kwargs["artwork_validator"] is not None

    def test_passes_an_artwork_text_validator_to_the_pipeline(self, tasks) -> None:
        # The artwork must not carry words: the image model cannot spell
        # (issue #208).
        pipeline, _ = _run_task(tasks, "generate_single_brief_image_task", {}, args=(7,))

        assert pipeline.call_args.kwargs["artwork_text_validator"] is not None

    def test_the_lettering_gate_can_be_turned_off(self, tasks) -> None:
        pipeline, _ = _run_task(
            tasks,
            "generate_single_brief_image_task",
            {"ARTWORK_TEXT_VALIDATION_ENABLED": "false"},
            args=(7,),
        )

        assert pipeline.call_args.kwargs["artwork_text_validator"] is None

    def test_does_not_create_a_bedrock_runtime_client(self, tasks) -> None:
        _, boto_client = _run_task(tasks, "generate_single_brief_image_task", {}, args=(7,))

        services = [call.args[0] for call in boto_client.call_args_list]
        assert services == ["s3"]


class TestComfyUISettings:
    def test_the_step_count_defaults_to_four(self, tasks) -> None:
        with patch.dict(os.environ, {}, clear=True):
            assert tasks._comfyui_steps() == 4

    def test_the_step_count_can_be_set(self, tasks) -> None:
        with patch.dict(os.environ, {"COMFYUI_STEPS": "8"}):
            assert tasks._comfyui_steps() == 8

    def test_a_step_count_that_is_not_a_number_falls_back_to_the_default(self, tasks) -> None:
        with patch.dict(os.environ, {"COMFYUI_STEPS": "many"}):
            assert tasks._comfyui_steps() == 4

    def test_the_cfg_defaults_to_one(self, tasks) -> None:
        with patch.dict(os.environ, {}, clear=True):
            assert tasks._comfyui_cfg() == 1.0

    def test_the_cfg_can_be_set(self, tasks) -> None:
        with patch.dict(os.environ, {"COMFYUI_CFG": "2.5"}):
            assert tasks._comfyui_cfg() == 2.5

    def test_a_cfg_that_is_not_a_number_falls_back_to_the_default(self, tasks) -> None:
        with patch.dict(os.environ, {"COMFYUI_CFG": "high"}):
            assert tasks._comfyui_cfg() == 1.0

    def test_the_client_is_built_with_the_step_count(self, tasks) -> None:
        pipeline, _ = _run_task(
            tasks,
            "generate_design_images_task",
            {"COMFYUI_STEPS": "8", "COMFYUI_CFG": "2.0"},
        )

        client = pipeline.call_args.kwargs["comfyui_client"]
        assert client._steps == 8
        assert client._cfg == 2.0


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


class TestLlmModelCheck:
    """The configured model is checked against the models the endpoint serves.

    The cluster pointed at a model the DGX proxy does not serve, so every
    brief and curation call failed and the logs never said which model was
    missing (issue #194).
    """

    def _run_brief_task(
        self,
        tasks,
        served_models: list[str],
        model: str,
        caplog: pytest.LogCaptureFixture,
    ) -> dict:
        client = MagicMock()
        client.models.list.return_value = SimpleNamespace(
            data=[SimpleNamespace(id=served) for served in served_models]
        )
        briefs = MagicMock(return_value={"status": "success", "briefs_created": 3})
        caplog.set_level(logging.ERROR)
        with (
            patch.dict(os.environ, {"LLM_API_KEY": "key", "LLM_MODEL": model}),
            patch("store_listing.orchestration.llm_client.get_llm_client", return_value=client),
            patch("store_listing.orchestration.design_briefs.generate_design_briefs", briefs),
            patch(
                "store_listing.orchestration.tasks.enabled_product_types",
                return_value=["dtf_apparel"],
            ),
        ):
            result = tasks.generate_design_briefs_task.run()
        return result

    def test_a_model_the_endpoint_does_not_serve_is_logged_with_the_served_ones(
        self, tasks, caplog: pytest.LogCaptureFixture
    ) -> None:
        self._run_brief_task(tasks, ["qwen38_briefs", "qwen_vl_ocr"], "qwen2.5:32b", caplog)

        assert "qwen38_briefs" in caplog.text
        assert "qwen_vl_ocr" in caplog.text

    def test_a_served_model_is_not_complained_about(
        self, tasks, caplog: pytest.LogCaptureFixture
    ) -> None:
        self._run_brief_task(tasks, ["qwen38_briefs"], "qwen38_briefs", caplog)

        assert "does not serve" not in caplog.text

    def test_the_result_says_which_model_was_asked(
        self, tasks, caplog: pytest.LogCaptureFixture
    ) -> None:
        result = self._run_brief_task(tasks, ["qwen38_briefs"], "qwen38_briefs", caplog)

        assert result["llm_model"] == "qwen38_briefs"
