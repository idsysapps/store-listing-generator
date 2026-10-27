"""Tests for the ComfyUI API-format workflow graph builders."""

from __future__ import annotations

from typing import Any

from store_listing.orchestration.comfyui_workflows import (
    build_birefnet_background_graph,
    build_flux_txt2img_graph,
    build_qwen_image_txt2img_graph,
)


def _graph(**overrides: Any) -> dict[str, Any]:
    params: dict[str, Any] = {
        "prompt": "a red bicycle leaning on a white wall",
        "width": 1024,
        "height": 1024,
        "seed": 42,
        "steps": 4,
        "cfg": 1.0,
        "sampler_name": "euler",
        "scheduler": "simple",
        "negative_prompt": "",
        "unet_name": "flux1-schnell.safetensors",
        "clip_name1": "t5xxl_fp8_e4m3fn.safetensors",
        "clip_name2": "clip_l.safetensors",
        "vae_name": "ae.safetensors",
    }
    params.update(overrides)
    return build_flux_txt2img_graph(**params)


class TestFluxTxt2ImgGraph:
    def test_has_the_nine_expected_nodes(self) -> None:
        graph = _graph()
        assert set(graph) == {"1", "2", "3", "4", "5", "6", "7", "8", "9"}

    def test_node_class_types(self) -> None:
        graph = _graph()
        assert graph["1"]["class_type"] == "UNETLoader"
        assert graph["2"]["class_type"] == "DualCLIPLoader"
        assert graph["3"]["class_type"] == "VAELoader"
        assert graph["4"]["class_type"] == "CLIPTextEncode"
        assert graph["5"]["class_type"] == "CLIPTextEncode"
        assert graph["6"]["class_type"] == "EmptySD3LatentImage"
        assert graph["7"]["class_type"] == "KSampler"
        assert graph["8"]["class_type"] == "VAEDecode"
        assert graph["9"]["class_type"] == "SaveImage"

    def test_unet_loader_uses_default_weight_dtype(self) -> None:
        graph = _graph()
        inputs = graph["1"]["inputs"]
        assert inputs["unet_name"] == "flux1-schnell.safetensors"
        assert inputs["weight_dtype"] == "default"

    def test_dual_clip_loader_is_flux_type(self) -> None:
        graph = _graph()
        inputs = graph["2"]["inputs"]
        assert inputs["clip_name1"] == "t5xxl_fp8_e4m3fn.safetensors"
        assert inputs["clip_name2"] == "clip_l.safetensors"
        assert inputs["type"] == "flux"

    def test_vae_loader(self) -> None:
        graph = _graph()
        assert graph["3"]["inputs"]["vae_name"] == "ae.safetensors"

    def test_sampler_uses_flux_schnell_settings(self) -> None:
        graph = _graph()
        inputs = graph["7"]["inputs"]
        assert inputs["steps"] == 4
        assert inputs["cfg"] == 1.0
        assert inputs["sampler_name"] == "euler"
        assert inputs["scheduler"] == "simple"
        assert inputs["denoise"] == 1.0

    def test_sampler_wires_model_positive_negative_and_latent(self) -> None:
        graph = _graph()
        inputs = graph["7"]["inputs"]
        assert inputs["model"] == ["1", 0]
        assert inputs["positive"] == ["4", 0]
        assert inputs["negative"] == ["5", 0]
        assert inputs["latent_image"] == ["6", 0]

    def test_sampler_carries_the_seed(self) -> None:
        graph = _graph(seed=1234)
        assert graph["7"]["inputs"]["seed"] == 1234

    def test_latent_carries_requested_dimensions(self) -> None:
        graph = _graph(width=512, height=768)
        inputs = graph["6"]["inputs"]
        assert inputs["width"] == 512
        assert inputs["height"] == 768
        assert inputs["batch_size"] == 1

    def test_positive_encode_receives_the_prompt(self) -> None:
        graph = _graph(prompt="a spooky forest")
        inputs = graph["4"]["inputs"]
        assert inputs["text"] == "a spooky forest"
        assert inputs["clip"] == ["2", 0]

    def test_negative_encode_receives_the_negative_prompt(self) -> None:
        graph = _graph(negative_prompt="blurry, low quality")
        inputs = graph["5"]["inputs"]
        assert inputs["text"] == "blurry, low quality"
        assert inputs["clip"] == ["2", 0]

    def test_vae_decode_and_save_are_wired(self) -> None:
        graph = _graph()
        assert graph["8"]["inputs"]["samples"] == ["7", 0]
        assert graph["8"]["inputs"]["vae"] == ["3", 0]
        assert graph["9"]["inputs"]["images"] == ["8", 0]
        assert graph["9"]["inputs"]["filename_prefix"] == "store_listing"

    def test_is_deterministic_for_the_same_inputs(self) -> None:
        assert _graph() == _graph()

    def test_different_seeds_produce_different_graphs(self) -> None:
        assert _graph(seed=1) != _graph(seed=2)


class TestBirefnetBackgroundGraph:
    def _graph(self, **overrides: Any) -> dict[str, Any]:
        params: dict[str, Any] = {
            "image_name": "uploaded_abc.png",
            "bg_model_name": "birefnet.safetensors",
        }
        params.update(overrides)
        return build_birefnet_background_graph(**params)

    def test_has_the_five_expected_nodes(self) -> None:
        graph = self._graph()
        assert set(graph) == {"1", "2", "3", "4", "5"}

    def test_node_class_types(self) -> None:
        graph = self._graph()
        assert graph["1"]["class_type"] == "LoadBackgroundRemovalModel"
        assert graph["2"]["class_type"] == "LoadImage"
        assert graph["3"]["class_type"] == "RemoveBackground"
        assert graph["4"]["class_type"] == "JoinImageWithAlpha"
        assert graph["5"]["class_type"] == "SaveImage"

    def test_loads_the_named_model_and_image(self) -> None:
        graph = self._graph()
        assert graph["1"]["inputs"]["bg_removal_name"] == "birefnet.safetensors"
        assert graph["2"]["inputs"]["image"] == "uploaded_abc.png"

    def test_remove_background_takes_the_model_and_the_image(self) -> None:
        graph = self._graph()
        inputs = graph["3"]["inputs"]
        assert inputs["bg_removal_model"] == ["1", 0]
        assert inputs["image"] == ["2", 0]

    def test_alpha_comes_from_the_mask_and_image_is_reused(self) -> None:
        graph = self._graph()
        inputs = graph["4"]["inputs"]
        assert inputs["image"] == ["2", 0]
        assert inputs["alpha"] == ["3", 0]

    def test_saves_the_composited_image(self) -> None:
        graph = self._graph()
        assert graph["5"]["inputs"]["images"] == ["4", 0]
        assert graph["5"]["inputs"]["filename_prefix"] == "bg_removed"

    def test_filename_prefix_is_overridable(self) -> None:
        graph = self._graph(filename_prefix="custom")
        assert graph["5"]["inputs"]["filename_prefix"] == "custom"

    def test_is_deterministic(self) -> None:
        assert self._graph() == self._graph()


class TestQwenImageTxt2ImgGraph:
    """The Qwen-Image graph follows the official ComfyUI template.

    It is deliberately not the Flux node shape: Qwen-Image needs a single
    Qwen2.5-VL text encoder, the AuraFlow model sampling shift, and a plain
    SD3 latent. The Phase 0 bake-off (issue #254) used exactly this graph.
    """

    def _graph(self, **overrides: Any) -> dict[str, Any]:
        params: dict[str, Any] = {
            "prompt": "a groovy ghost with the word GROOVY",
            "width": 1328,
            "height": 1328,
            "seed": 42,
            "steps": 20,
            "cfg": 4.0,
            "sampler_name": "euler",
            "scheduler": "simple",
            "shift": 3.1,
            "negative_prompt": "blurry, misspelled",
            "unet_name": "qwen_image_fp8_e4m3fn.safetensors",
            "clip_name": "qwen_2.5_vl_7b_fp8_scaled.safetensors",
            "vae_name": "qwen_image_vae.safetensors",
        }
        params.update(overrides)
        return build_qwen_image_txt2img_graph(**params)

    def test_has_the_ten_expected_nodes(self) -> None:
        assert set(self._graph()) == {str(number) for number in range(1, 11)}

    def test_node_class_types(self) -> None:
        graph = self._graph()
        assert graph["1"]["class_type"] == "UNETLoader"
        assert graph["2"]["class_type"] == "CLIPLoader"
        assert graph["3"]["class_type"] == "VAELoader"
        assert graph["4"]["class_type"] == "ModelSamplingAuraFlow"
        assert graph["5"]["class_type"] == "CLIPTextEncode"
        assert graph["6"]["class_type"] == "CLIPTextEncode"
        assert graph["7"]["class_type"] == "EmptySD3LatentImage"
        assert graph["8"]["class_type"] == "KSampler"
        assert graph["9"]["class_type"] == "VAEDecode"
        assert graph["10"]["class_type"] == "SaveImage"

    def test_unet_loader(self) -> None:
        inputs = self._graph()["1"]["inputs"]
        assert inputs["unet_name"] == "qwen_image_fp8_e4m3fn.safetensors"
        assert inputs["weight_dtype"] == "default"

    def test_single_qwen_text_encoder(self) -> None:
        inputs = self._graph()["2"]["inputs"]
        assert inputs["clip_name"] == "qwen_2.5_vl_7b_fp8_scaled.safetensors"
        assert inputs["type"] == "qwen_image"
        assert inputs["device"] == "default"

    def test_vae_loader(self) -> None:
        assert self._graph()["3"]["inputs"]["vae_name"] == "qwen_image_vae.safetensors"

    def test_aura_flow_shift_wraps_the_model(self) -> None:
        inputs = self._graph(shift=2.5)["4"]["inputs"]
        assert inputs["model"] == ["1", 0]
        assert inputs["shift"] == 2.5

    def test_sampler_wires_the_shifted_model_and_both_encodes(self) -> None:
        inputs = self._graph()["8"]["inputs"]
        assert inputs["model"] == ["4", 0]
        assert inputs["positive"] == ["5", 0]
        assert inputs["negative"] == ["6", 0]
        assert inputs["latent_image"] == ["7", 0]

    def test_sampler_uses_qwen_image_settings(self) -> None:
        inputs = self._graph()["8"]["inputs"]
        assert inputs["steps"] == 20
        assert inputs["cfg"] == 4.0
        assert inputs["sampler_name"] == "euler"
        assert inputs["scheduler"] == "simple"
        assert inputs["denoise"] == 1.0

    def test_sampler_carries_the_seed(self) -> None:
        assert self._graph(seed=1234)["8"]["inputs"]["seed"] == 1234

    def test_latent_carries_the_requested_dimensions(self) -> None:
        inputs = self._graph(width=1024, height=1280)["7"]["inputs"]
        assert inputs["width"] == 1024
        assert inputs["height"] == 1280
        assert inputs["batch_size"] == 1

    def test_positive_and_negative_prompts(self) -> None:
        graph = self._graph(prompt="a ghost", negative_prompt="blurry")
        assert graph["5"]["inputs"]["text"] == "a ghost"
        assert graph["5"]["inputs"]["clip"] == ["2", 0]
        assert graph["6"]["inputs"]["text"] == "blurry"
        assert graph["6"]["inputs"]["clip"] == ["2", 0]

    def test_vae_decode_and_save_are_wired(self) -> None:
        graph = self._graph()
        assert graph["9"]["inputs"]["samples"] == ["8", 0]
        assert graph["9"]["inputs"]["vae"] == ["3", 0]
        assert graph["10"]["inputs"]["images"] == ["9", 0]
        assert graph["10"]["inputs"]["filename_prefix"] == "qwen_wordmark"

    def test_filename_prefix_is_overridable(self) -> None:
        graph = self._graph(filename_prefix="hero")
        assert graph["10"]["inputs"]["filename_prefix"] == "hero"

    def test_is_deterministic(self) -> None:
        assert self._graph() == self._graph()
