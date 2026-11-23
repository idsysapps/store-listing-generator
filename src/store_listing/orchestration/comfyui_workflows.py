"""Pure builders for the ComfyUI API-format workflow graphs used by the DGX pipeline.

The graph shapes here were validated against a live ComfyUI 0.37.1 instance on the
DGX Spark. They are kept as pure functions so they can be asserted on directly
without a running server.

Node links use the ComfyUI API format: ``["<node_id>", <output_index>]``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

__all__ = [
    "build_birefnet_background_graph",
    "build_flux2_dev_txt2img_graph",
    "build_flux_txt2img_graph",
    "build_qwen_image_txt2img_graph",
]

DEFAULT_FILENAME_PREFIX = "store_listing"
DEFAULT_BACKGROUND_FILENAME_PREFIX = "bg_removed"
QWEN_IMAGE_FILENAME_PREFIX = "qwen_wordmark"
FLUX2_FILENAME_PREFIX = "flux2-dev-personal"
FLUX2_UNET_NAME = "flux2_dev_fp8mixed.safetensors"
FLUX2_CLIP_NAME = "mistral_3_small_flux2_fp8.safetensors"
FLUX2_VAE_NAME = "flux2-vae.safetensors"
FLUX2_GUIDANCE = 5.0
FLUX2_STEPS = 20
FLUX2_WIDTH = 1024
FLUX2_HEIGHT = 1280


def build_flux_txt2img_graph(
    *,
    prompt: str,
    width: int,
    height: int,
    seed: int,
    steps: int,
    cfg: float,
    sampler_name: str,
    scheduler: str,
    negative_prompt: str,
    unet_name: str,
    clip_name1: str,
    clip_name2: str,
    vae_name: str,
    filename_prefix: str = DEFAULT_FILENAME_PREFIX,
) -> dict[str, Any]:
    """Return the 9-node Flux Schnell text-to-image graph.

    Flux Schnell is a distilled model: it is meant to run in 4 steps at cfg 1.0
    with the euler/simple sampler pair. The UNET emits only MODEL in this
    ComfyUI version, so the CLIP and VAE are loaded by separate nodes.
    """
    return {
        "1": {
            "class_type": "UNETLoader",
            "inputs": {"unet_name": unet_name, "weight_dtype": "default"},
        },
        "2": {
            "class_type": "DualCLIPLoader",
            "inputs": {"clip_name1": clip_name1, "clip_name2": clip_name2, "type": "flux"},
        },
        "3": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": vae_name},
        },
        "4": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": prompt, "clip": ["2", 0]},
        },
        "5": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": negative_prompt, "clip": ["2", 0]},
        },
        "6": {
            "class_type": "EmptySD3LatentImage",
            "inputs": {"width": width, "height": height, "batch_size": 1},
        },
        "7": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["1", 0],
                "seed": seed,
                "steps": steps,
                "cfg": cfg,
                "sampler_name": sampler_name,
                "scheduler": scheduler,
                "positive": ["4", 0],
                "negative": ["5", 0],
                "latent_image": ["6", 0],
                "denoise": 1.0,
            },
        },
        "8": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["7", 0], "vae": ["3", 0]},
        },
        "9": {
            "class_type": "SaveImage",
            "inputs": {"filename_prefix": filename_prefix, "images": ["8", 0]},
        },
    }


def build_qwen_image_txt2img_graph(
    *,
    prompt: str,
    width: int,
    height: int,
    seed: int,
    steps: int = 20,
    cfg: float = 4.0,
    sampler_name: str = "euler",
    scheduler: str = "simple",
    shift: float = 3.1,
    negative_prompt: str = "",
    unet_name: str,
    clip_name: str,
    vae_name: str,
    filename_prefix: str = QWEN_IMAGE_FILENAME_PREFIX,
) -> dict[str, Any]:
    """Return the Qwen-Image text-to-image graph from the official template.

    Qwen-Image is a 20B MMDiT. It is not a Flux graph: the text encoder is a
    single Qwen2.5-VL model loaded with ``type="qwen_image"``, the model needs
    the AuraFlow sampling shift, and the latent is the SD3 shape. The sampler
    runs 20 steps at cfg 4.0 - the settings the Phase 0 bake-off used (issue
    #254). It is much slower than Flux Schnell, so only the typographic hero
    path uses it.
    """
    return {
        "1": {
            "class_type": "UNETLoader",
            "inputs": {"unet_name": unet_name, "weight_dtype": "default"},
        },
        "2": {
            "class_type": "CLIPLoader",
            "inputs": {"clip_name": clip_name, "type": "qwen_image", "device": "default"},
        },
        "3": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": vae_name},
        },
        "4": {
            "class_type": "ModelSamplingAuraFlow",
            "inputs": {"model": ["1", 0], "shift": shift},
        },
        "5": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": prompt, "clip": ["2", 0]},
        },
        "6": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": negative_prompt, "clip": ["2", 0]},
        },
        "7": {
            "class_type": "EmptySD3LatentImage",
            "inputs": {"width": width, "height": height, "batch_size": 1},
        },
        "8": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["4", 0],
                "seed": seed,
                "steps": steps,
                "cfg": cfg,
                "sampler_name": sampler_name,
                "scheduler": scheduler,
                "positive": ["5", 0],
                "negative": ["6", 0],
                "latent_image": ["7", 0],
                "denoise": 1.0,
            },
        },
        "9": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["8", 0], "vae": ["3", 0]},
        },
        "10": {
            "class_type": "SaveImage",
            "inputs": {"filename_prefix": filename_prefix, "images": ["9", 0]},
        },
    }


def build_flux2_dev_txt2img_graph(
    *,
    prompt: str,
    seed: int,
    reference_images: Sequence[str] = (),
    width: int = FLUX2_WIDTH,
    height: int = FLUX2_HEIGHT,
    steps: int = FLUX2_STEPS,
    guidance: float = FLUX2_GUIDANCE,
    unet_name: str = FLUX2_UNET_NAME,
    clip_name: str = FLUX2_CLIP_NAME,
    vae_name: str = FLUX2_VAE_NAME,
    filename_prefix: str = FLUX2_FILENAME_PREFIX,
) -> dict[str, Any]:
    """The FLUX.2 dev graph: one prompt, optional reference latents, no negative.

    Personal custom designs use this graph. Sellable briefs stay on Flux Schnell.
    Each reference is a filename already uploaded to ComfyUI.
    """
    graph: dict[str, Any] = {
        "1": {
            "class_type": "UNETLoader",
            "inputs": {"unet_name": unet_name, "weight_dtype": "default"},
        },
        "2": {
            "class_type": "CLIPLoader",
            "inputs": {"clip_name": clip_name, "type": "flux2"},
        },
        "3": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": vae_name},
        },
        "4": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": prompt, "clip": ["2", 0]},
        },
    }
    conditioning = ["4", 0]
    next_id = 5
    for image_name in reference_images:
        load_id = str(next_id)
        scale_id = str(next_id + 1)
        encode_id = str(next_id + 2)
        reference_id = str(next_id + 3)
        graph[load_id] = {"class_type": "LoadImage", "inputs": {"image": image_name}}
        graph[scale_id] = {
            "class_type": "ImageScaleToTotalPixels",
            "inputs": {
                "image": [load_id, 0],
                "upscale_method": "lanczos",
                "megapixels": 1.0,
                "resolution_steps": 16,
            },
        }
        graph[encode_id] = {
            "class_type": "VAEEncode",
            "inputs": {"pixels": [scale_id, 0], "vae": ["3", 0]},
        }
        graph[reference_id] = {
            "class_type": "ReferenceLatent",
            "inputs": {"conditioning": conditioning, "latent": [encode_id, 0]},
        }
        conditioning = [reference_id, 0]
        next_id += 4

    guidance_id = str(next_id)
    latent_id = str(next_id + 1)
    scheduler_id = str(next_id + 2)
    sampler_id = str(next_id + 3)
    noise_id = str(next_id + 4)
    guider_id = str(next_id + 5)
    sample_id = str(next_id + 6)
    decode_id = str(next_id + 7)
    save_id = str(next_id + 8)
    graph[guidance_id] = {
        "class_type": "FluxGuidance",
        "inputs": {"conditioning": conditioning, "guidance": guidance},
    }
    graph[latent_id] = {
        "class_type": "EmptyFlux2LatentImage",
        "inputs": {"width": width, "height": height, "batch_size": 1},
    }
    graph[scheduler_id] = {
        "class_type": "Flux2Scheduler",
        "inputs": {"steps": steps, "width": width, "height": height},
    }
    graph[sampler_id] = {
        "class_type": "KSamplerSelect",
        "inputs": {"sampler_name": "euler"},
    }
    graph[noise_id] = {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}}
    graph[guider_id] = {
        "class_type": "BasicGuider",
        "inputs": {"model": ["1", 0], "conditioning": [guidance_id, 0]},
    }
    graph[sample_id] = {
        "class_type": "SamplerCustomAdvanced",
        "inputs": {
            "noise": [noise_id, 0],
            "guider": [guider_id, 0],
            "sampler": [sampler_id, 0],
            "sigmas": [scheduler_id, 0],
            "latent_image": [latent_id, 0],
        },
    }
    graph[decode_id] = {
        "class_type": "VAEDecode",
        "inputs": {"samples": [sample_id, 0], "vae": ["3", 0]},
    }
    graph[save_id] = {
        "class_type": "SaveImage",
        "inputs": {"images": [decode_id, 0], "filename_prefix": filename_prefix},
    }
    return graph


def build_birefnet_background_graph(
    *,
    image_name: str,
    bg_model_name: str,
    filename_prefix: str = DEFAULT_BACKGROUND_FILENAME_PREFIX,
) -> dict[str, Any]:
    """Return the 5-node background-removal graph.

    ``RemoveBackground`` outputs a foreground mask, which is fed straight into
    ``JoinImageWithAlpha`` as the alpha channel, producing a transparent PNG.
    ``image_name`` is the name of an image already uploaded to ComfyUI.
    """
    return {
        "1": {
            "class_type": "LoadBackgroundRemovalModel",
            "inputs": {"bg_removal_name": bg_model_name},
        },
        "2": {
            "class_type": "LoadImage",
            "inputs": {"image": image_name},
        },
        "3": {
            "class_type": "RemoveBackground",
            "inputs": {"bg_removal_model": ["1", 0], "image": ["2", 0]},
        },
        "4": {
            "class_type": "JoinImageWithAlpha",
            "inputs": {"image": ["2", 0], "alpha": ["3", 0]},
        },
        "5": {
            "class_type": "SaveImage",
            "inputs": {"filename_prefix": filename_prefix, "images": ["4", 0]},
        },
    }
