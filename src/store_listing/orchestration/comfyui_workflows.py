"""Pure builders for the ComfyUI API-format workflow graphs used by the DGX pipeline.

The graph shapes here were validated against a live ComfyUI 0.37.1 instance on the
DGX Spark. They are kept as pure functions so they can be asserted on directly
without a running server.

Node links use the ComfyUI API format: ``["<node_id>", <output_index>]``.
"""

from __future__ import annotations

from typing import Any

__all__ = ["build_birefnet_background_graph", "build_flux_txt2img_graph"]

DEFAULT_FILENAME_PREFIX = "store_listing"
DEFAULT_BACKGROUND_FILENAME_PREFIX = "bg_removed"


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
