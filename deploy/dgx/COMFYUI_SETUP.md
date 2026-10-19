# ComfyUI + vLLM on DGX Spark (GB10 ARM64)

No pre-built ARM64 ComfyUI Docker images exist. The `comfyui-spark:latest` image
must be built natively on the DGX before first use.

## Quick Start

```bash
# On DGX — from deploy/dgx/ directory
./setup.sh          # Builds image + starts all services
./download-flux.sh  # Downloads Flux Schnell + BiRefNet (~30 GB)
```

## What setup.sh Does

1. Clones [slangevi/comfyui-spark](https://github.com/slangevi/comfyui-spark) to `/opt/comfyui-build/`
2. Builds `comfyui-spark:latest` Docker image (~15GB, takes ~10 min)
3. Copies docker-compose.yml and openai-proxy.py to `/opt/trtllm/`
4. Starts all services (vLLM briefs, vLLM vision, ComfyUI, proxy)

## Updating the Stack

The compose file and the proxy are the only two files the running stack reads.
Copy them across and recreate the containers:

```bash
# From the repository root
scp deploy/dgx/docker-compose.yml deploy/dgx/openai-proxy.py \
    mathianasj@192.168.2.123:/opt/trtllm/
ssh -t mathianasj@192.168.2.123 \
    'docker compose -f /opt/trtllm/docker-compose.yml up -d'
```

`docker compose up -d` only recreates what changed, so the ComfyUI container and
its loaded models survive the update. The first start of a new service pulls its
model (Qwen2.5-VL-7B is ~16 GB).

To confirm the stack came up:

```bash
curl -s http://127.0.0.1:8188/system_stats   # ComfyUI
curl -s http://127.0.0.1:9000/v1/models       # the proxy's model list
```

### Autostart on boot

Every service in the compose file uses `restart: unless-stopped`, so Docker
restarts the stack on boot by itself. The old `trtllm.service` systemd unit
(which wrapped `docker compose up` for the retired TensorRT-LLM deployment) is
no longer part of the repository. If it is still installed on the host, remove
it so the stack is not started twice:

```bash
sudo systemctl disable --now trtllm.service
sudo rm /etc/systemd/system/trtllm.service
sudo systemctl daemon-reload
```

## Model Directory

Models are stored on the host at `/opt/comfyui-data/models/`:

```
/opt/comfyui-data/
  models/
    diffusion_models/flux1-schnell.safetensors   # Flux Schnell UNET (23.8 GB)
    text_encoders/clip_l.safetensors             # CLIP-L encoder (246 MB)
    text_encoders/t5xxl_fp8_e4m3fn.safetensors   # T5-XXL fp8 encoder (4.89 GB)
    vae/ae.safetensors                          # Flux VAE (335 MB)
    background_removal/birefnet.safetensors      # BiRefNet (444 MB)
  custom_nodes/
  output/
  input/
```

ComfyUI also scans the legacy `unet/` and `clip/` folder names as aliases for
`diffusion_models/` and `text_encoders/`, but the modern names above are what
`download-flux.sh` installs.

### Model Sources

All mirrors are public - no Hugging Face token and no licence click required:

| File | Source | Size |
|------|--------|------|
| `flux1-schnell.safetensors` | `Comfy-Org/flux1-schnell` | 23.8 GB |
| `t5xxl_fp8_e4m3fn.safetensors` | `comfyanonymous/flux_text_encoders` | 4.89 GB |
| `clip_l.safetensors` | `comfyanonymous/flux_text_encoders` | 246 MB |
| `ae.safetensors` | `Comfy-Org/Lumina_Image_2.0_Repackaged` | 335 MB |
| `birefnet.safetensors` | `ZhengPeng7/BiRefNet` (`model.safetensors`) | 444 MB |

> **Do not** point the download script at `black-forest-labs/FLUX.1-schnell`. That
> repo is gated: it returns 401 unauthenticated and 403 with a valid HF token
> unless the licence was accepted on the account.

`download-flux.sh` is idempotent and resumable - re-running it continues a
partial download and verifies the final file sizes.

### Verifying the installation

```bash
curl -s http://127.0.0.1:8188/models/diffusion_models
curl -s http://127.0.0.1:8188/models/text_encoders
curl -s http://127.0.0.1:8188/models/vae
curl -s http://127.0.0.1:8188/models/background_removal
```

## GB10-Specific Flags

The docker-compose sets these ComfyUI flags for GB10 unified memory:

- `--disable-pinned-memory` — avoids pinned memory overhead on unified fabric
- `--use-pytorch-cross-attention` — safe cross-attention for sm_121
- `--bf16-unet --bf16-vae --bf16-text-enc` — Blackwell native precision

**Do NOT use:** `--gpu-only` (fights unified memory), `--highvram` (causes OOM)

The vision model runs in `bfloat16`: Blackwell (sm_121) VL kernels are not
reliably supported in fp8.

## Ports

| Service | Port | Purpose |
|---------|------|---------|
| vllm-briefs | 8002 | `Qwen3-14B` — design brief generation |
| vllm-vision | 8003 | `Qwen2.5-VL-7B-Instruct` — reads the text back out of a generated image |
| comfyui | 8188 | Flux Schnell + BiRefNet image generation |
| openai-proxy | 9000 | Unified LLM routing (`qwen38_briefs`, `qwen_vl_ocr`) |

The openai-proxy exposes both models over one OpenAI-compatible endpoint at
`:9000`, so the application only needs a single base URL.

## GPU Memory Budget

GB10 has 128 GB of unified memory shared by every service:

| Service | `--gpu-memory-utilization` |
|---------|---------------------------|
| vllm-briefs | 0.20 |
| vllm-vision | 0.15 |
| ComfyUI | takes the remainder (~23 GB for Flux + BiRefNet) |

The coder service was removed, so roughly 30 % of the unified memory is now
available to ComfyUI. If ComfyUI still runs out of memory, raise
`vllm-briefs` to `0.30` and `vllm-vision` to `0.20` and re-check
`nvidia-smi` on the host.

## Alternative Images

If `slangevi/comfyui-spark` doesn't work, these also target DGX Spark:

- [ftechmax/DGX-Spark-ComfyUI](https://github.com/ftechmax/DGX-Spark-ComfyUI) — includes SageAttention 2 + ONNX Runtime for sm_121
- [ecarmen16/SparkyUI](https://github.com/ecarmen16/SparkyUI) — SageAttention, mobile UI included
