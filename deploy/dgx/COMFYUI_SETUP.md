# ComfyUI + vLLM on DGX Spark (GB10 ARM64)

No pre-built ARM64 ComfyUI Docker images exist. The `comfyui-spark:latest` image
must be built natively on the DGX before first use.

## Quick Start

```bash
# On DGX — from deploy/dgx/ directory
./setup.sh          # Builds image + starts all services
./download-flux.sh  # Downloads Flux Schnell + BiRefNet (~30 GB)

# Optional typographic wordmark models (see "Wordmark models" below)
./download-qwen-image.sh    # Qwen-Image 1.0, Apache 2.0 (~30 GB) - recommended
./download-flux2-klein.sh   # FLUX.2 Klein 4B, Apache 2.0 (~24 GB) - experimental
```

## What setup.sh Does

1. Clones [slangevi/comfyui-spark](https://github.com/slangevi/comfyui-spark) to `/opt/comfyui-build/`
2. Builds `comfyui-spark:latest` Docker image (~15GB, takes ~10 min)
3. Copies docker-compose.yml and openai-proxy.py to `/opt/trtllm/`
4. Starts all services (vLLM briefs, vLLM vision, ComfyUI, proxy)

## Updating the Stack

The compose file and the proxy are the only two files the running stack reads.
`/opt/trtllm` is root-owned, so copy the files to a writable directory first
and move them into place with `sudo`:

```bash
# From the repository root
scp deploy/dgx/docker-compose.yml deploy/dgx/openai-proxy.py \
    mathianasj@192.168.2.123:/tmp/
ssh -t mathianasj@192.168.2.123 \
    'sudo cp /tmp/docker-compose.yml /tmp/openai-proxy.py /opt/trtllm/ \
     && cd /opt/trtllm && sudo docker compose up -d'
```

`docker compose up -d` only recreates what changed, so the ComfyUI container and
its loaded models survive the update. The first start of a new service pulls its
model (Qwen2.5-VL-7B is ~16 GB).

> **This blocks for several minutes.** The services wait for each other to
> report healthy before the next one starts (see
> [Services start one at a time](#services-start-one-at-a-time)), and each
> healthcheck has a 300 s `start_period` so the model load does not count as a
> failure. Watch it from a second terminal with
> `ssh mathianasj@192.168.2.123 'cd /opt/trtllm && docker compose ps'`.

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

    # Optional wordmark models (download-qwen-image.sh / download-flux2-klein.sh)
    diffusion_models/qwen_image_fp8_e4m3fn.safetensors          # Qwen-Image 1.0 (20.4 GB)
    text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors         # Qwen2.5-VL-7B (9.4 GB)
    vae/qwen_image_vae.safetensors                              # Qwen-Image VAE (254 MB)
    diffusion_models/flux-2-klein-4b.safetensors                # FLUX.2 Klein distilled (7.8 GB)
    diffusion_models/flux-2-klein-base-4b.safetensors           # FLUX.2 Klein base (7.8 GB)
    text_encoders/qwen_3_4b.safetensors                         # Qwen3-4B (8.0 GB)
    vae/flux2-vae.safetensors                                   # FLUX.2 VAE (336 MB)
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

### Wordmark models

These are only needed for the typographic design path. Both are Apache 2.0, so
their output can be printed on products we sell.

| Model | Files | Size | Phase 0 result (20 phrases) |
|-------|-------|------|-----------------------------|
| **Qwen-Image 1.0** (20B) | `qwen_image_fp8_e4m3fn`, `qwen_2.5_vl_7b_fp8_scaled`, `qwen_image_vae` | ~30 GB | **20/20 exact**, ~100 s/image |
| FLUX.2 Klein 4B base | `flux-2-klein-base-4b`, `qwen_3_4b`, `flux2-vae` | ~16 GB | 15/20 exact, ~25 s/image |
| FLUX.2 Klein 4B distilled | `flux-2-klein-4b`, `qwen_3_4b`, `flux2-vae` | ~16 GB | 13/20 exact, ~5 s/image |

The bake-off prompted each model for a standalone bubble-letter wordmark on a
plain white background and OCR'd the result with the DGX Qwen2.5-VL model.
Spelling errors were confirmed by eye, not just OCR. Klein's style is close to
the reference but it misspells common words (`WEERD`, `COOD`, `HOCUS POUSS`),
so it is not safe without an OCR gate and retries. Qwen-Image rendered every
phrase exactly, including `SALT & PEPPER`, `2 SPOOKY 4 YOU`, and `BOO JEE`.

> Qwen-Image **2.1** is research-only and must not be used. FLUX.2 dev (32B) and
> FLUX.2 Klein 9B are non-commercial. Ideogram is API-only.

### Typographic hero path

The renderer has two image paths:

| Path | Model | What it draws | Text source |
|------|-------|---------------|-------------|
| `type_hero` / `wordmark_stack` | Qwen-Image | Kicker + hero word + subject | Qwen draws it, strict OCR gate |
| `full_bleed`, `emblem`, captions, no-text | Flux Schnell | Artwork only | Compositor, or no text at all |

The hero path is opt-in and disabled by default. Enable it for a trial batch
with:

```bash
TYPOGRAPHIC_HERO_ENABLED=1
# Optional: target specific briefs when they do not use a hero layout yet.
TYPOGRAPHIC_HERO_BRIEF_IDS=254,255
# Optional: Qwen renders to retry before the brief fails (default 2).
TYPOGRAPHIC_HERO_CANDIDATES=2
# Optional: upload the Phase 1 image without the composited descriptor.
TYPOGRAPHIC_HERO_DESCRIPTOR=0
```

Qwen renders the kicker and the hero word in one pass. The descriptor is not
part of that pass: three blocks in one prompt garbled the third block in the
Phase 1 tests, so the descriptor is composited deterministically afterwards.
Every candidate is read back by Qwen2.5-VL three times; at least two readings
must match the copy exactly, preserving `&`, apostrophes, hyphens, and digits.
A candidate that fails is retried with the next seed, and a brief that runs out
of candidates is left unrendered rather than uploaded with a typo.

The descriptor compositor finds the artwork's content bounding box, reflows the
artwork into the upper part of the canvas, and draws the descriptor in Anton in
a reserved bottom band. It never overlaps the artwork. The DTF file gets the
descriptor on the keyed artwork with a light outline. Set
`TYPOGRAPHIC_HERO_DESCRIPTOR=0` to upload the Phase 1 image unchanged.

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

GB10 has 128 GB of unified memory shared by every service, and the two vLLM
services are the tightest constraint:

| Service | `--gpu-memory-utilization` | Notes |
|---------|---------------------------|-------|
| vllm-briefs | 0.25 | fp8 weights, 16k context, eager mode |
| vllm-vision | 0.30 | 15.7 GB of bf16 weights plus the activation peak |
| ComfyUI | the remainder (~0.23) | Flux Schnell (23.8 GB) + BiRefNet |

The vision model needs the larger share: `Qwen2.5-VL-7B` is 15.7 GB in
bfloat16, and vLLM's profiling run needs headroom on top of that before it can
size the KV cache. At `0.15` the budget was smaller than the weights alone and
the engine died with `No available memory for the cache blocks`.

### Two settings that matter

- **`--enforce-eager`** on both vLLM services. CUDA graph capture needs several
  GB on top of the weights, and a cold capture was what pushed the first start
  over the budget. Eager mode avoids the capture entirely. Neither model is on
  a latency path - briefs run in a daily batch and the vision model only
  validates generated images.
- **`--limit-mm-per-prompt={"image":1,"video":0}`** on the vision model.
  Without `video:0`, vLLM profiles the video encoder at its maximum feature
  size during startup, which was the single largest memory spike in the boot.

### Services start one at a time

The `depends_on: service_healthy` chain (briefs → vision → ComfyUI → proxy) is
deliberate. When both vLLM services started together they raced for the unified
memory pool, and each computed its KV cache budget against a free-memory
snapshot the other had already consumed - both engines died with a negative KV
cache. Serialising the startup makes the budget deterministic.

If ComfyUI still runs out of memory, lower `vllm-vision` to `0.25` and re-check
`nvidia-smi` on the host.

## Alternative Images

If `slangevi/comfyui-spark` doesn't work, these also target DGX Spark:

- [ftechmax/DGX-Spark-ComfyUI](https://github.com/ftechmax/DGX-Spark-ComfyUI) — includes SageAttention 2 + ONNX Runtime for sm_121
- [ecarmen16/SparkyUI](https://github.com/ecarmen16/SparkyUI) — SageAttention, mobile UI included
