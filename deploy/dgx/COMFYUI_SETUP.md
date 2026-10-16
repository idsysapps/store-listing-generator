# ComfyUI on DGX Spark (GB10 ARM64)

No pre-built ARM64 ComfyUI Docker images exist. The `comfyui-spark:latest` image
must be built natively on the DGX before first use.

## Quick Start

```bash
# On DGX — from deploy/dgx/ directory
./setup.sh          # Builds image + starts all services
./download-flux.sh  # Downloads Flux Schnell (~25GB total)
```

## What setup.sh Does

1. Clones [slangevi/comfyui-spark](https://github.com/slangevi/comfyui-spark) to `/opt/comfyui-build/`
2. Builds `comfyui-spark:latest` Docker image (~15GB, takes ~10 min)
3. Copies docker-compose.yml and openai-proxy.py to `/opt/trtllm/`
4. Starts all services (vLLM coder, vLLM briefs, ComfyUI, proxy)

## Model Directory

Models are stored on the host at `/opt/comfyui-data/models/`:

```
/opt/comfyui-data/
  models/
    unet/flux1-schnell.safetensors     # Flux Schnell UNET
    clip/clip_l.safetensors            # CLIP-L encoder
    clip/t5xxl_fp8_e4m3fn.safetensors  # T5-XXL fp8 encoder
    vae/ae.safetensors                 # Flux VAE
  custom_nodes/
  output/
  input/
```

## GB10-Specific Flags

The docker-compose sets these ComfyUI flags for GB10 unified memory:

- `--disable-pinned-memory` — avoids pinned memory overhead on unified fabric
- `--use-pytorch-cross-attention` — safe cross-attention for sm_121
- `--bf16-unet --bf16-vae --bf16-text-enc` — Blackwell native precision

**Do NOT use:** `--gpu-only` (fights unified memory), `--highvram` (causes OOM)

## Ports

| Service | Port | Purpose |
|---------|------|---------|
| vllm-coder | 8001 | Qwen3-30B-A3B NVFP4 |
| vllm-briefs | 8002 | Qwen3-14B NVFP4 |
| comfyui | 8188 | Flux Schnell |
| openai-proxy | 9000 | Unified LLM routing |

## Alternative Images

If `slangevi/comfyui-spark` doesn't work, these also target DGX Spark:

- [ftechmax/DGX-Spark-ComfyUI](https://github.com/ftechmax/DGX-Spark-ComfyUI) — includes SageAttention 2 + ONNX Runtime for sm_121
- [ecarmen16/SparkyUI](https://github.com/ecarmen16/SparkyUI) — SageAttention, mobile UI included
