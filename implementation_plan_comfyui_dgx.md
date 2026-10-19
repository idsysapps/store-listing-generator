# Implementation Plan — Issue #111: ComfyUI + Flux Schnell on DGX (replace Leonardo.ai + Bedrock)

**Status:** Phases 0–8 implemented and merged into `feat/comfyui-dgx-image-generation`.
Phases 1–8 are live-validated against the DGX (Flux generation, BiRefNet background
removal and the ComfyUI error paths all pass the integration tests). Phase 9
(AWS decommissioning) is still outstanding — it is manual ops and waits for the path
to have run in production.

**Branch:** `feat/comfyui-dgx-image-generation`
**Supersedes:** the stale `implementation_plan_comfyui_dgx.md` (root) — see [Why the old plan was wrong](#why-the-old-plan-was-wrong)

### What changed from the plan as written

| Plan said | Reality |
|-----------|---------|
| Remove Leonardo and Bedrock immediately (D2) | Done, as agreed — but the vision model and the OCR gate shipped in the same change, so the app never had a window with no generation path. |
| `COMFYUI_BG_MODEL_NAME` default `birefnet.safetensors` | The client now **discovers** the installed background-removal model from `GET /object_info/LoadBackgroundRemovalModel`; the env var is now an override, not a default. |
| OCR gate reads the transparent image | The gate reads the **raw** render (what Flux actually produced) before background removal. |
| `ImageResult` keeps four fields | A fifth field, `background_removed`, was added so the pipeline can tell a real transparent PNG from a background-removal fallback. |
| ComfyUI needs a restart to pick up models | It does not — 0.37.1 resolves the model enums per request. A restart is not required. |
| Vision model on `sm_121` is a risk | `Qwen2.5-VL-7B-Instruct` in `bfloat16` is configured in `deploy/dgx/docker-compose.yml`; fp8 is deliberately not used. |
| Three vLLM services (coder, briefs, vision) | The **coder** service (`granite-4.2-8b`, `:8001`) was removed — it was not used by the application and its 30 % GPU reservation is better spent on ComfyUI. The proxy now routes `qwen38_briefs` and `qwen_vl_ocr` only, and `qwen38_briefs` is the default model. |
| The `deploy/dgx` directory held a TensorRT-LLM deployment | `Dockerfile`, `entrypoint.sh`, `model-configs/` and `trtllm.service` targeted the retired Triton/TensorRT-LLM stack and were referenced by nothing else. They were removed; the services autostart through Docker's `restart: unless-stopped` policy instead. |
| ComfyUI's `RemoveBackground` returns a foreground mask | It does not — on 0.37.1 it returns a **background** map (1 = background) even though the node documents a foreground mask, so joining it straight onto the image punches the subject out and keeps the background. The client now **measures** the polarity with a synthetic calibration image and flips the alpha only when needed, so it stays correct if upstream fixes the node. |


---

## 1. Goal

Make self-hosted **ComfyUI + Flux Schnell on the DGX Spark** the *only* image generation path for design briefs.
Delete Leonardo.ai, delete the Ideogram text-routing workaround, delete the AWS Bedrock image calls
(generation *and* background removal), delete the dead Pillow text compositor, and add an **OCR text-validation
gate** so text errors are caught before images reach S3 / the approval queue.

**Decisions taken (2026-10-19):**

| # | Decision | Choice |
|---|----------|--------|
| D1 | OCR mechanism | **Vision LLM on the DGX** — third vLLM service, routed through the existing `openai-proxy` on `:9000` |
| D2 | Leonardo / Bedrock | **Remove both immediately** — no fallback path remains |
| D3 | Background removal | **Move to the DGX** via ComfyUI's core BiRefNet node |
| D4 | Plan document | Written to `~/.opencode/plan/` **and** the repo root (for the PR) |

---

## 2. Verified current state (probed live, 2026-10-19)

Everything below was verified by reading the repo and by probing the live DGX at `http://192.168.2.123:8188`.
Do not re-derive it — but **do** re-verify the DGX state before starting Phase 1, because the models may
have been downloaded since.

### 2.1 Application code

| Fact | Detail |
|------|--------|
| Image generation module | `src/store_listing/orchestration/image_generation.py` (220 lines) |
| — Bedrock generation | `generate_image()` → `bedrock_client.invoke_model(stability.sd3-5-large-v1:0)` |
| — Leonardo generation | `generate_image_leonardo()` → `LeonardoClient` (async httpx), routes `text_top`/`text_top_bottom` briefs to **Ideogram 3.0** (`IDEOGRAM_MODEL = "ideogram-v3.0"`, `_needs_ideogram()`) |
| — Bedrock bg removal | `remove_background()` → `us.stability.stable-image-remove-background-v1:0` |
| Prompt builders | `build_prompt_from_brief()` (explicitly "Do not include any text or lettering") and `build_ideogram_prompt()` (adds quoted headline/tagline + font color) |
| Pipeline module | `src/store_listing/orchestration/image_pipeline.py` — `generate_images_for_briefs()`, `_generate_single_brief()`, `generate_image_for_brief()`. Picks Leonardo if a client is passed, else Bedrock. Uploads `_raw` + `_transparent` to S3, writes keys to `design_briefs`. |
| Celery entry points | `src/store_listing/orchestration/tasks.py` → `generate_design_images_task`, `generate_single_brief_image_task` (the latter is triggered by the Java `requestRegeneration` GraphQL mutation, which clears the image keys and sets `regeneration_feedback`) |
| Text compositor | `src/store_listing/orchestration/text_compositor.py` — `LayoutSpec`, `composite_text`. **Verified dead code**: nothing in `src/` imports it; only `tests/unit/test_text_compositor.py` does. |
| Leonardo client | `src/store_listing/orchestration/leonardo_client.py` — only consumed by `image_generation.py` and `tasks.py` |
| **ComfyUI client** | **DOES NOT EXIST.** There is no `comfyui_client.py` in `src/`, no `USE_DGX_IMAGE_GENERATION` gate, no `COMFYUI_BASE_URL` handling anywhere in `src/`. Only a stale `__pycache__/comfyui_client.cpython-314.pyc` from an abandoned WIP. **The issue's "Current scaffolding (already in place)" section is inaccurate — everything must be built from scratch.** |
| Blast radius | `image_generation.py`, `image_pipeline.py`, `tasks.py` only. `leonardo_client.py` and `text_compositor.py` are leaves. No frontend/Java change needed (the OCR gate sits inside the Python pipeline, before S3). |
| Dependencies | `httpx` ✅ direct dep, `Pillow` ✅, `openai` ✅, `boto3` ✅ (still needed for **S3**), `requests` ❌ transitive only — **do not use `requests`**, `aiohttp` ❌ not a dep |
| DB | `design_briefs` table already has `layout_type`, `headline_text`, `tagline_text`, `font_color`, `regeneration_feedback` → **no schema change needed for this issue** |

### 2.2 DGX Spark (live probe of `http://192.168.2.123:8188`)

| Fact | Detail |
|------|--------|
| ComfyUI | **v0.37.1**, running with `--listen 0.0.0.0 --port 8188 --base-directory /data --disable-pinned-memory --use-pytorch-cross-attention --bf16-unet --bf16-vae --bf16-text-enc` |
| Hardware | NVIDIA GB10 (Blackwell, sm_121), 130 GB unified memory, `cudaMallocAsync` |
| Node count | 961 node classes available |
| vLLM | `granite-4.2-8b` @ **:8001**, `Qwen3-14B` @ **:8002**, `openai-proxy` @ **:9000** — all healthy |
| **Models** | ⚠️ **NONE INSTALLED.** `/models/diffusion_models`, `/models/text_encoders`, `/models/vae`, `/models/checkpoints`, `/models/background_removal` all return `[]`; `UNETLoader.unet_name`, `DualCLIPLoader.clip_name1/2`, `CheckpointLoaderSimple.ckpt_name` enums are all empty. **`download-flux.sh` was run on 2026-10-19 and downloaded nothing** — it aborted on its first `sudo` and two of its four URLs are gated (full diagnosis in §2.7). Flux Schnell is still *not* installed. |
| Text encoding | No vision model on the DGX (both vLLM models are text-only) → D1 requires adding one |
| Background removal | `LoadBackgroundRemovalModel` (`comfy_extras.nodes_bg_removal`, **core node**) + `RemoveBackground` exist but the `background_removal` model folder is empty. `BriaRemoveImageBackground` is a **paid partner API**, not usable. |

### 2.3 ComfyUI REST API (verified against 0.37.1)

| Route | Behaviour |
|-------|-----------|
| `POST /prompt` | body `{"prompt": {node_id: {"class_type", "inputs"}}, "client_id"}` → `200 {"prompt_id", "number", "node_errors"}`; on invalid graph `400 {"error", "node_errors": {node_id: {"errors": [{type, message, details}]}}}` |
| `GET /history/{prompt_id}` | `200 {prompt_id: {"outputs": {node_id: {"images": [{filename, subfolder, type}]}}, "status": {"status_str": "success"|"error", "completed": bool, "messages": [...]}}}` |
| `GET /view?filename=&subfolder=&type=output` | raw image bytes |
| `POST /upload/image` | multipart form (`image` field) → `{"name", "subfolder", "type"}` — needed to feed a generated PNG back into a second workflow |
| `GET /object_info[/<Class>]` | node schema + enum of available models |
| `GET /queue` | `{"queue_running": [], "queue_pending": []}` |
| `GET /system_stats` | health/version |
| `/api/*` aliases | also work (`/api/prompt`, `/api/system_stats`, …) — use the classic non-`/api` routes |

### 2.4 Model folder layout (verified against upstream `folder_paths.py`)

ComfyUI resolves **both** the modern and the legacy directory names:

| Loader | Directory on the DGX host (`/opt/comfyui-data/models/`) | Legacy alias | File |
|--------|------------------------------------------------------|--------------|------|
| `UNETLoader` | `unet/` **or** `diffusion_models/` | `unet` → `diffusion_models` | `flux1-schnell.safetensors` (~23 GB) |
| `DualCLIPLoader` | `clip/` **or** `text_encoders/` | `clip` → `text_encoders` | `clip_l.safetensors`, `t5xxl_fp8_e4m3fn.safetensors` |
| `VAELoader` | `vae/` | — | `ae.safetensors` |
| `LoadBackgroundRemovalModel` | `background_removal/` | — | BiRefNet `model.safetensors` (~444 MB) |

➡️ The existing `deploy/dgx/download-flux.sh` targets `models/unet`, `models/clip`, `models/vae` — **those paths are correct** and need no change. It only needs BiRefNet added.

### 2.5 Flux Schnell graph — validated against the live server

This exact API-format graph was POSTed to the live DGX. The **structure is accepted**; the only errors are the
missing model files (proving the graph shape, node classes, and input names are all correct):

```
node 1  UNETLoader            {unet_name: "flux1-schnell.safetensors", weight_dtype: "default"}
node 2  DualCLIPLoader        {clip_name1: "t5xxl_fp8_e4m3fn.safetensors", clip_name2: "clip_l.safetensors", type: "flux"}
node 3  VAELoader             {vae_name: "ae.safetensors"}
node 4  CLIPTextEncode        {text: <prompt>, clip: ["2", 0]}
node 5  CLIPTextEncode        {text: "", clip: ["2", 0]}          # negative
node 6  EmptySD3LatentImage   {width: 1024, height: 1024, batch_size: 1}
node 7  KSampler              {seed, steps: 4, cfg: 1.0, sampler_name: "euler", scheduler: "simple",
                               denoise: 1.0, model: ["1",0], positive: ["4",0], negative: ["5",0], latent_image: ["6",0]}
node 8  VAEDecode             {samples: ["7",0], vae: ["3",0]}
node 9  SaveImage             {filename_prefix: "store_listing", images: ["8",0]}
```

Server responses received:
- `vae_name: 'ae.safetensors' not in ['pixel_space']`
- `clip_name2: 'clip_l.safetensors' not in []`, `clip_name1: 't5xxl_fp8_e4m3fn.safetensors' not in []`
- `unet_name: 'flux1-schnell.safetensors' not in []`

⚠️ **`UNETLoader` in 0.37.1 outputs only `MODEL`** (older ComfyUI also emitted CLIP and VAE from it). A separate
`DualCLIPLoader` + `VAELoader` is therefore **required** — do not "simplify" the graph by referencing `["1",1]`/`["1",2]`.

Other verified node facts: `KSampler.sampler_name` includes `euler`; `scheduler` includes `simple`;
`EmptySD3LatentImage{width,height,batch_size}`; `ImageCompositeMasked{destination,source,x,y,resize_source}`;
`JoinImageWithAlpha{image,alpha}`; `LoadImageOutput{image}`; `ImageToMask`, `MaskToImage`, `ImageInvert` all exist.

### 2.6 Repository hygiene (uncommitted / untracked, must be resolved in Phase 0)

| Path | State | Action |
|------|-------|--------|
| `deploy/dgx/openai-proxy.py` | **uncommitted**, +649/−171 — a full proxy rewrite (Responses-API translation, SSE streaming) | **Not part of #111** → land in its own PR first; #111 then adds one route on top |
| `deploy/dgx/docker-compose.yml` | **uncommitted** — swaps vLLM coder `Qwen3-30B-A3B-NVFP4` → `granite-4.2-8b`, briefs → `Qwen/Qwen3-14B` | Same as above |
| `test_comfyui.py` (repo root) | untracked scratch script using a **fabricated API** (`TextPromptInput`, `FluxSchnellNode`, `GET /image/{id}.png` — none of which exist) | **Delete** |
| `cost-book-master/` | untracked clone of an unrelated repo (has its own `.git`) | **Delete from the working dir** (or move outside the repo) |
| `implementation_plan_comfyui_dgx.md` (root) | untracked, stale | **Replace** with this document |
| `.claude/worktrees/hello-world/` | an old worktree copy of the repo | Leave alone; ensure it is git-ignored |

### 2.7 Diagnosed failure of `download-flux.sh` (live on the DGX, 2026-10-19)

The script was run as `mathianasj` (bash history: `./download-flux.sh`, no `sudo`) and downloaded **nothing**.
`find / -xdev -name "*.safetensors" -size +50M` → **zero hits anywhere on the host**. Three independent bugs,
each of which alone would have broken the run:

1. **`sudo` aborts the script at line 11.**
   ```
   ==> Creating model directories
   sudo: a terminal is required to read the password; either use the -S option to read
   from standard input or configure an askpass helper
   EXIT=1
   ```
   `set -euo pipefail` kills the script on the first `sudo mkdir`. Reproduced live.
   **`sudo` is not even required** — `/opt/comfyui-data/models` is owned by uid 1000, which is exactly the
   container's user (`docker exec comfyui id` → `uid=1000(comfy)`), and `models/{unet,clip,vae,diffusion_models,text_encoders}`
   are all uid 1000 and writable.
2. **Two of the four URLs are behind a gated Hugging Face repo.** `black-forest-labs/FLUX.1-schnell` returns
   **HTTP 401** unauthenticated and **HTTP 403** with the DGX's own `HF_TOKEN` (the token is valid, but the
   account has not accepted the FLUX licence). The script points at that repo for both
   `flux1-schnell.safetensors` and `ae.safetensors` → both fail even after the `sudo` fix.
   **Fix:** `Comfy-Org/flux1-schnell` is `gated: false` / `private: false` and hosts the byte-identical
   `flux1-schnell.safetensors`; `Comfy-Org/Lumina_Image_2.0_Repackaged/split_files/vae/ae.safetensors` is the
   VAE mirror the ComfyUI examples README itself recommends.
3. **`models/background_removal/` does not exist** on the host, so BiRefNet has nowhere to land.

**Ruled out (verified, not the cause):** the bind mount is correct (`/opt/comfyui-data` → `/data`, rw);
ownership is correct (uid 1000 host = uid 1000 in container); the legacy `models/unet` + `models/clip` paths
are still scanned by this build (`diffusion_models -> ['/data/models/unet', '/data/models/diffusion_models']`,
`text_encoders -> ['/data/models/text_encoders', '/data/models/clip']`); the `comfyui-direct.service` unit does
not exist on this host and port 8188 is served by the `comfyui` container; and ComfyUI itself is healthy — the
container log shows the §2.5 probe's validation errors, i.e. the API is working and correctly rejecting a graph
whose model files are missing.



```
                    ┌──────────────────────── DGX Spark (GB10) ────────────────────────┐
                    │                                                                    │
  Celery worker ───►│  ComfyUI :8188  (Flux Schnell)          vLLM :8001 :8002  :8003 │
  (Python app)      │      │                                    granite  Qwen3   Qwen2.5-VL
       │            │      │                                        └── openai-proxy :9000 ──┘
       │            │      ▼                                                  ▲
       │            │  models/unet, models/clip, models/vae,          vision OCR verdict
       │            │  models/background_removal (BiRefNet)                        │
       └────────────┴────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
                        S3 (raw + transparent PNG)  ──►  DB design_briefs image keys
                                                             │
                                                             ▼
                                            trend-service GraphQL → approval UI
```

**Flow per brief:** build Flux prompt (text folded in) → generate → *(dtf_apparel only)* BiRefNet background removal → OCR text validation → retry ≤ N on mismatch → upload to S3 → update DB.

---

## 4. Work plan

Every phase follows **TDD (RED → GREEN → REFACTOR)** per `AGENTS.md`. Public functions get full type hints
(`pyright` runs in CI over `src/`), and `ruff check` + `ruff format` must pass.

### Phase 0 — Repository hygiene and branch split

**Goal:** a clean diff for #111 that contains only image-generation work.

1. Create a branch off the current HEAD and move the **uncommitted LLM-routing work** onto it:
   - `git stash` the `deploy/dgx/openai-proxy.py` + `docker-compose.yml` changes, or commit them to a
     `feat/dgx-llm-routing` branch and open a PR for it. This is a prerequisite for Phase 1 (the vision route
     is added on top of the rewritten proxy).
2. Delete `test_comfyui.py` and `cost-book-master/` from the working tree.
3. Replace `implementation_plan_comfyui_dgx.md` at the repo root with this document.
4. Confirm `git status` shows only the plan file as new.

**Acceptance:** `git status` clean apart from the plan; LLM-routing changes isolated on their own branch.

---

### Phase 1 — DGX stack: models + vision service + proxy route

**Goal:** the DGX can actually serve Flux Schnell, BiRefNet, and vision OCR.

> ⚠️ **This is the blocking phase.** Nothing in Phases 2–6 can be validated end-to-end until Flux is installed.

1. **Fix `deploy/dgx/download-flux.sh`, then install the models on the DGX** (`/opt/comfyui-data/models/`).

   ⚠️ **The current script cannot work** — it has three independent bugs, all diagnosed live on 2026-10-19
   (see §2.7). The corrected script must:
   - **drop `sudo` entirely** — `/opt/comfyui-data/models` is owned by uid 1000, which is exactly the
     container's user (`uid=1000(comfy)`), so the script's own user can write without elevation;
   - **use public mirrors, not the gated repo** — `black-forest-labs/FLUX.1-schnell` returns 401 unauthenticated
     and 403 even with the DGX's own HF token (license not accepted), but `Comfy-Org/flux1-schnell` is
     `gated: false` and carries the byte-identical `flux1-schnell.safetensors` (23,782,506,688 bytes);
   - **create `models/background_removal/`** (it does not exist) and fetch BiRefNet.

   Verified target set — all public, no token required:

   | File | Source | Size | Destination |
   |------|--------|------|-------------|
   | `flux1-schnell.safetensors` | `Comfy-Org/flux1-schnell` | 23.8 GB | `models/diffusion_models/` |
   | `t5xxl_fp8_e4m3fn.safetensors` | `comfyanonymous/flux_text_encoders` | 4.89 GB | `models/text_encoders/` |
   | `clip_l.safetensors` | `comfyanonymous/flux_text_encoders` | 246 MB | `models/text_encoders/` |
   | `ae.safetensors` | `Comfy-Org/Lumina_Image_2.0_Repackaged/split_files/vae/` | 335 MB | `models/vae/` |
   | `birefnet.safetensors` | `ZhengPeng7/BiRefNet` → `model.safetensors` | 444 MB | `models/background_removal/` |

   Total ≈ 29.7 GB at the measured **16.4 MB/s** ⇒ ~30 minutes. 3.0 TB free on `/opt`.
   *(`models/unet` + `models/clip` remain valid legacy paths — verified in this build — but the modern
   directories are used above to match ComfyUI 0.37's own layout.)*

   The BiRefNet loader detects the model via the `bb.layers.1.blocks.0.attn.relative_position_index`
   state-dict key, which `ZhengPeng7/BiRefNet` provides. **Verify** after download that the
   `LoadBackgroundRemovalModel` combo lists `birefnet.safetensors`; if not, fall back to a BiRefNet-distilled
   checkpoint that does (see Risks).
2. **Add the vision service** to `deploy/dgx/docker-compose.yml`:
   ```yaml
   vllm-vision:
     image: ghcr.io/timothystewart6/vllm-gb10:v0.31.0-gb10.1
     container_name: vllm-vision
     restart: unless-stopped
     ipc: host
     network_mode: host
     deploy:
       resources:
         reservations:
           devices: [{driver: nvidia, count: all, capabilities: [gpu]}]
     volumes: [hf-cache:/root/.cache/huggingface]
     environment:
       - HUGGING_FACE_HUB_TOKEN=${HUGGING_FACE_HUB_TOKEN}
       - HF_TOKEN=${HUGGING_FACE_HUB_TOKEN}
     command:
       - "vllm"
       - "serve"
       - "Qwen/Qwen2.5-VL-7B-Instruct"
       - "--host=0.0.0.0"
       - "--port=8003"
       - "--gpu-memory-utilization=0.15"
       - "--max-model-len=8192"
       - "--max-num-seqs=2"
       - "--dtype=bfloat16"
       - "--trust-remote-code"
       - "--limit-mm-per-prompt={\"image\":1}"
       - "--served-model-name=qwen_vl_ocr"
     healthcheck:
       test: ["CMD", "curl", "-sf", "http://localhost:8003/health"]
       interval: 30s
       timeout: 10s
       retries: 30
       start_period: 300s
   ```
   *Memory check:* current steady state is 0.30 (coder) + 0.20 (briefs) = 0.50 of GB10's 128 GB unified memory.
   0.15 for the VL model leaves ~35 % for ComfyUI + Flux (23 GB) + BiRefNet — comfortable.
   *Do not use fp8 for the VL model on sm_121* — Blackwell VL kernels are not reliably supported; use `bfloat16`.
3. **Add the proxy route** to `deploy/dgx/openai-proxy.py`:
   ```python
   MODEL_ROUTES: dict[str, str] = {
       "granite_coder": "http://localhost:8001",
       "qwen38_briefs": "http://localhost:8002",
       "qwen_vl_ocr": "http://localhost:8003",  # NEW — vision OCR
   }
   ```
   The proxy's `_proxy_request` forwards the body JSON verbatim, so OpenAI-style
   `messages[].content = [{"type":"text"}, {"type":"image_url","image_url":{"url":"data:image/jpeg;base64,..."}}]`
   passes through untouched. **No proxy change is needed beyond the route** — but re-verify after the
   Phase 0 rewrite lands.
4. **Smoke-test on the DGX** (manually, from the DGX host):
   ```bash
   curl -s http://127.0.0.1:8188/models/diffusion_models   # expect ["flux1-schnell.safetensors"]
   curl -s http://127.0.0.1:8188/models/text_encoders      # expect clip_l + t5xxl
   curl -s http://127.0.0.1:8188/models/vae               # expect ["ae.safetensors"]
   curl -s http://127.0.0.1:8188/models/background_removal
   curl -s http://127.0.0.1:9000/v1/models                # expect qwen_vl_ocr
   ```
5. **Update `deploy/dgx/COMFYUI_SETUP.md`** with the verified model layout, the new service, and the env vars.

**Acceptance:** the 5 curls above return the expected content; a Flux generation through the graph in §2.5
returns a real PNG.

---

### Phase 2 — `ComfyUIClient` (new module, TDD)

**Goal:** a typed, testable client for the ComfyUI REST API.

**New file:** `src/store_listing/orchestration/comfyui_client.py`

```python
"""ComfyUI client for DGX-hosted image generation (Flux Schnell).

Speaks the ComfyUI REST API: POST /prompt -> poll GET /history/{id} -> GET /view.
Verified against ComfyUI 0.37.1 on the DGX Spark.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

import httpx

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "http://localhost:8188"
DEFAULT_TIMEOUT_SECONDS = 300.0
DEFAULT_POLL_INTERVAL_SECONDS = 1.0


class ComfyUIError(Exception):
    """Raised when ComfyUI rejects a request or a workflow fails to execute."""


@dataclass
class ComfyUIResult:
    """Outcome of a single ComfyUI workflow execution."""

    image_bytes: bytes | None
    prompt_id: str | None = None
    error: str | None = None
    seed: int | None = None


class ComfyUIClient:
    def __init__(
        self,
        *,
        base_url: str = DEFAULT_BASE_URL,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        poll_interval_seconds: float = DEFAULT_POLL_INTERVAL_SECONDS,
        http_client: httpx.Client | None = None,
    ) -> None: ...

    def generate_image(
        self,
        *,
        prompt: str,
        width: int = 1024,
        height: int = 1024,
        seed: int | None = None,
        steps: int = 4,
        cfg: float = 1.0,
        sampler_name: str = "euler",
        scheduler: str = "simple",
        negative_prompt: str = "",
        unet_name: str = "flux1-schnell.safetensors",
        clip_name1: str = "t5xxl_fp8_e4m3fn.safetensors",
        clip_name2: str = "clip_l.safetensors",
        vae_name: str = "ae.safetensors",
    ) -> ComfyUIResult:
        """Queue a Flux Schnell txt2img workflow and block until the image is ready."""

    def remove_background(
        self, image_bytes: bytes, *, bg_model_name: str = "birefnet.safetensors"
    ) -> ComfyUIResult:
        """Upload the image to ComfyUI, run BiRefNet, return a transparent RGBA PNG."""

    def health(self) -> bool:
        """True when GET /system_stats succeeds."""
```

**Implementation rules**

- Use **httpx** (a direct dependency) — *not* `requests`, *not* `aiohttp`. Stay **synchronous**: the whole
  existing pipeline (`image_generation.generate_image`) is sync, and `image_pipeline` is called from Celery tasks.
- `POST /prompt` → on `400`, parse `node_errors` and raise `ComfyUIError` with the **joined** `details` strings
  (this is what surfaces "model not found" from a mis-provisioned DGX). On `200`, take `prompt_id`.
- Poll `GET /history/{prompt_id}` every `poll_interval_seconds` until `status.completed` is true or
  `timeout_seconds` elapses. Read the image descriptor from `outputs[<SaveImage node id>]["images"][0]`
  (`filename`, `subfolder`, `type`).
- Fetch the bytes from `GET /view?filename=…&subfolder=…&type=…`.
- If `status.status_str == "error"`, extract the message from `status.messages` and raise `ComfyUIError`.
- `remove_background` first uploads the source PNG via `POST /upload/image` (multipart, field name `image`,
  `overwrite=true`, `type=input`), then runs the BiRefNet graph (§4.2) with `LoadImage` + `RemoveBackground` +
  `JoinImageWithAlpha` + `SaveImage`, and returns the RGBA PNG bytes.
- Log the `prompt_id` on every call so a failed run can be inspected in the ComfyUI UI.

**TDD — `tests/unit/test_comfyui_client.py`** (use `httpx.MockTransport`, no network):
1. `generate_image` returns PNG bytes on the happy path (queue → history → view).
2. `POST /prompt` returning `400` with `node_errors` → `ComfyUIError` whose message contains the node detail.
3. `GET /history` reporting `status_str == "error"` → `ComfyUIError` with the execution message.
4. History never completes before the timeout → `ComfyUIError` (use a tiny timeout + `poll_interval=0`).
5. `GET /view` returning an empty body → `ComfyUIError`.
6. `remove_background` uploads the image first, then runs the 5-node graph, and returns the RGBA PNG.
7. `health()` returns `True`/`False` on 200/non-200.

---

### Phase 3 — Workflow builders (new module, pure functions, TDD)

**Goal:** keep the ComfyUI graph shapes in pure, directly testable functions (AGENTS.md TEA/pure-function style).

**New file:** `src/store_listing/orchestration/comfyui_workflows.py`

```python
"""Pure builders for the ComfyUI API-format workflow graphs used by the DGX pipeline."""


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
) -> dict[str, Any]:
    """Return the 9-node Flux Schnell graph documented in §2.5."""


def build_birefnet_background_graph(
    *, image_name: str, bg_model_name: str, filename_prefix: str = "bg_removed"
) -> dict[str, Any]:
    """Return the 5-node background-removal graph (LoadBackgroundRemovalModel -> RemoveBackground
    -> JoinImageWithAlpha -> SaveImage)."""
```

**TDD — `tests/unit/test_comfyui_workflows.py`:**
1. The txt2img graph has exactly the 9 expected node ids with the expected `class_type`s.
2. `KSampler` inputs: `steps == 4`, `cfg == 1.0`, `sampler_name == "euler"`, `scheduler == "simple"`.
3. `EmptySD3LatentImage` carries the requested width/height/batch size.
4. The positive `CLIPTextEncode` receives the prompt; the negative one receives the negative prompt.
5. The background graph wires `RemoveBackground` → `JoinImageWithAlpha` → `SaveImage` with the right link ids.
6. `build_flux_txt2img_graph` is deterministic (same inputs → identical dict) apart from the seed.

---

### Phase 4 — Prompt folding (text into the Flux prompt)

**Goal:** replace the Ideogram text-routing workaround with text folded directly into the Flux prompt, and
delete the dead compositor.

**File:** `src/store_listing/orchestration/image_generation.py`

- **Delete:** `generate_image()` (Bedrock), `generate_image_leonardo()`, `remove_background()` (Bedrock),
  `build_ideogram_prompt()`, `_needs_ideogram()`, `IDEOGRAM_MODEL`, `IDEOGRAM_QUALITY`, `BEDROCK_MODEL_ID`,
  `BEDROCK_REGION`, the `LeonardoClient`/`LeonardoGenerationResult` imports, and the `asyncio`/`base64`/`json`
  imports they needed.
- **Delete files:** `src/store_listing/orchestration/leonardo_client.py`,
  `tests/unit/test_leonardo_client.py`, `src/store_listing/orchestration/text_compositor.py`,
  `tests/unit/test_text_compositor.py`.
- **Replace** `build_prompt_from_brief()` with a single layout-aware builder:

```python
def build_flux_prompt(brief: DesignBrief) -> str:
    """Build the Flux prompt for a brief, folding layout_type/headline/tagline/font_color in.

    full_bleed     -> scene only, explicitly no lettering
    text_top       -> headline at the top, scene, no tagline
    text_top_bottom-> headline at the top, scene, tagline at the bottom
    """
```

  Rules (carried over from the verified `build_ideogram_prompt` behaviour, minus the Ideogram-specific bits):
  - quote the text exactly: `Bold text reading "HEADLINE" in #RRGGBB at the top.`
  - `Text reading "TAGLINE" in #RRGGBB at the bottom.`
  - keep the scene/style/product-context sentence
  - keep `Do not draw anything else.`
  - append `brief.regeneration_feedback` when present
  - for `full_bleed` keep `Do not include any text or lettering.`
  - default `font_color` to `#000000` when `brief.font_color` is `None`
- **Add** a text-only prompt for `full_bleed` briefs that have a `headline_text` but a `full_bleed` layout —
  keep the existing behaviour (no text) and log a warning; this is a data-quality issue, not a code path.
- Keep `ImageResult` (it is the pipeline's contract) with the same fields: `image_bytes`, `prompt`, `error`,
  `raw_bytes`.

**TDD — rewrite `tests/unit/test_image_generation.py`:**
1. `full_bleed` brief → prompt contains the scene + style + product context, contains
   `do not include any text or lettering`, and does **not** contain the headline.
2. `text_top` brief → prompt contains `Bold text reading "SPOOKY SEASON"` and the font colour, and does **not**
   contain the tagline.
3. `text_top_bottom` brief → prompt contains both quoted strings, top and bottom.
4. `font_color=None` → defaults to `#000000`.
5. `regeneration_feedback` is appended verbatim.
6. `generate_image(brief, comfyui_client=...)` calls the client with the folded prompt and returns an
   `ImageResult` carrying the image bytes; a client error becomes `ImageResult(error=...)` (never raises).
7. `generate_image` for a `dtf_apparel` brief returns the raw bytes in `raw_bytes` and the transparent bytes in
   `image_bytes`.

---

### Phase 5 — OCR text validation (new module, TDD)

**Goal:** verify the rendered text matches the brief before the image is stored.

**New file:** `src/store_listing/orchestration/text_validation.py`

```python
"""Validate that the text rendered by Flux matches the text requested by the brief.

Uses a vision LLM on the DGX (routed through the openai-proxy) to read the text back
out of the generated image, then compares it with the expected headline/tagline.
"""

from __future__ import annotations

import base64
import difflib
import io
import re
from dataclasses import dataclass

from PIL import Image

OCR_PROMPT = (
    "Transcribe every word of text visible in this image, verbatim. "
    "Preserve capitalisation and reading order. "
    "If there is no text at all, reply with exactly: NO_TEXT"
)


@dataclass
class TextValidationResult:
    matched: bool
    expected: list[str]
    observed: str
    score: float
    error: str | None = None


def encode_image_for_vision(image_bytes: bytes, *, max_side: int = 1024) -> str:
    """Downscale + JPEG-encode to a base64 data URL for the vision model."""


def normalize_text(text: str) -> str:
    """Casefold, collapse whitespace, drop punctuation — for tolerant comparison."""


def text_matches(expected: str, observed: str, *, threshold: float = 0.85) -> tuple[bool, float]:
    """Return (matched, similarity) using difflib.SequenceMatcher on normalized text."""


class TextValidator:
    """Reads text back out of a generated image using a vision LLM on the DGX."""

    def __init__(self, *, client, model: str, threshold: float = 0.85) -> None: ...

    def validate(
        self, image_bytes: bytes, *, headline: str | None, tagline: str | None
    ) -> TextValidationResult: ...
```

**Rules**
- Only validate when the brief actually has text (`headline_text` or `tagline_text`); `full_bleed` briefs
  short-circuit to `matched=True` with no model call.
- Every expected string must be present in the observed text. Use the **best** match rather than requiring an
  exact substring, so a stray extra word does not fail the image but a *wrong* word does.
- `NO_TEXT` in the observation → `matched=False` (the text was requested but is missing).
- A vision-model error/timeout → `TextValidationResult(matched=False, error=...)`. Fail closed: an unverifiable
  image is not uploaded.
- The image is downscaled to ≤1024 px and JPEG-encoded (quality 80) before base64 — a 1024×1024 PNG is ~1.5 MB
  and base64 adds a third again; the VL model does not need full resolution to read a headline.
- The vision client is the **existing** `openai` SDK client pattern from `llm_client.py` — add
  `get_vision_client()` / `get_vision_model()` there, reading `VISION_LLM_BASE_URL` / `VISION_LLM_MODEL`
  (default model `qwen_vl_ocr`). The `openai` package is already a dependency.

**TDD — `tests/unit/test_text_validation.py`:**
1. `normalize_text` folds case, collapses whitespace, strips punctuation.
2. `text_matches("SPOOKY SEASON", "spooky season")` → matched, score 1.0.
3. `text_matches("SPOOKY SEASON", "SPOOKY SEAS0N")` → matched (score ≥ threshold).
4. `text_matches("SPOOKY SEASON", "HAPPY HALLOWEEN")` → not matched.
5. `TextValidator.validate` with a mocked client returning the expected text → `matched=True`.
6. …returning `NO_TEXT` → `matched=False`.
7. …raising an exception → `matched=False` and `error` is set (fail closed).
8. `headline=None, tagline=None` → no client call, `matched=True`.
9. `encode_image_for_vision` returns a `data:image/jpeg;base64,` URL and the decoded bytes are a valid JPEG
   no larger than `max_side` on the long edge.

---

### Phase 6 — Pipeline wiring: ComfyUI primary + OCR gate + DGX background removal

**Goal:** the end-to-end path, with the OCR gate in front of S3.

**File:** `src/store_listing/orchestration/image_pipeline.py`

- `generate_images_for_briefs(...)` and `generate_image_for_brief(...)`:
  - **remove** the `leonardo_client` parameter,
  - **add** `comfyui_client: ComfyUIClient` (required) and `text_validator: TextValidator | None = None`.
- `_generate_single_brief(...)`:
  1. Build the brief (unchanged).
  2. `generate_image(brief, comfyui_client=comfyui_client)` → `ImageResult`.
  3. **OCR gate loop** (new, the core of the issue):
     ```python
     attempts = 0
     while True:
         attempts += 1
         result = generate_image(
             brief, comfyui_client=comfyui_client, seed=seed, extra_prompt=retry_hint
         )
         if result.image_bytes is None:
             return False  # generation itself failed
         if text_validator is None or not brief.headline_text and not brief.tagline_text:
             break  # nothing to validate
         verdict = text_validator.validate(result.image_bytes, headline=..., tagline=...)
         if verdict.matched:
             break
         if attempts > max_retries:
             logger.error("Text validation failed after %d attempts: %s", attempts, verdict)
             return False  # never upload a wrong-text image
         retry_hint = f'The text must read exactly "{headline}". Render no other words.'
     ```
     `max_retries` comes from `MAX_TEXT_REGENERATION_ATTEMPTS` (default `2` → up to 3 generations).
     Each retry uses a **new seed** and an augmented prompt carrying the OCR feedback, so the model is nudged
     rather than repeating the same image.
  4. **Background removal** for `dtf_apparel` → `comfyui_client.remove_background(raw_bytes)` (BiRefNet),
     producing the transparent PNG. If it fails, log a warning and fall back to the raw image (the existing
     "Background removal failed, returning raw image" behaviour) — a DTF brief still ships.
  5. Upload `raw` + `transparent` to S3 and update the DB keys (unchanged).
- **Fail-closed decision:** when the OCR gate exhausts its retries the brief is **not** uploaded and is counted
  as an error. Rationale: the issue requires text errors to be caught *before* images reach S3/the approval queue.
  The failure reason (expected vs observed text) is logged so the brief can be re-queued with
  `regeneration_feedback` from the UI.

**TDD — rewrite `tests/unit/test_image_pipeline.py`:**
1. A `full_bleed` brief generates, uploads, and updates the DB keys.
2. A `dtf_apparel` brief calls `remove_background` and uploads the transparent PNG under the `_transparent` key.
3. A brief **with** a headline: the validator is called; a matching result uploads normally.
4. A brief with a headline: the validator fails twice → the image is **not** uploaded, the DB keys are not
   updated, and the result reports an error.
5. A brief with a headline: the validator fails once then passes → **two** generations happen and the second
   image is uploaded.
6. A brief with no text → the validator is never called.
7. `remove_background` raising → the raw image is still uploaded (no transparent key).
8. `generate_image` returning an error → the brief is counted as an error and nothing is uploaded.

---

### Phase 7 — Celery task wiring + dependency cleanup

**File:** `src/store_listing/orchestration/tasks.py`

- `generate_design_images_task()` and `generate_single_brief_image_task()`:
  - **remove** the `boto3.client("bedrock-runtime", ...)` construction and the `BEDROCK_*` env handling,
  - **remove** the `LEONARDO_API_KEY` / `LeonardoClient` construction,
  - **add**:
    ```python
    from store_listing.orchestration.comfyui_client import ComfyUIClient
    from store_listing.orchestration.text_validation import TextValidator
    from store_listing.orchestration.llm_client import get_vision_client, get_vision_model

    comfyui_client = ComfyUIClient(
        base_url=os.environ.get("COMFYUI_BASE_URL", "http://localhost:8188"),
        timeout_seconds=float(os.environ.get("COMFYUI_TIMEOUT_SECONDS", "300")),
    )
    text_validator = TextValidator(
        client=get_vision_client(), model=get_vision_model(), threshold=OCR_MATCH_THRESHOLD
    )
    ```
  - the S3 client stays (`boto3.client("s3", ...)`) — **boto3 remains a dependency for S3 only**.
  - update the task docstrings ("via Leonardo/Bedrock" → "via ComfyUI/Flux on the DGX").
- **File:** `src/store_listing/orchestration/llm_client.py` — add `get_vision_client()` /
  `get_vision_model()` (see §4.5).
- **File:** `src/store_listing/orchestration/design_briefs.py` — no change, but double-check that
  `scene_description` survives (it is selected by `google_trends.py` but is **absent from
  `src/store_listing/db/schema.sql`** — a pre-existing schema drift, out of scope, but worth a separate issue).

**TDD — new `tests/unit/test_tasks.py`:**
1. `generate_design_images_task` constructs a `ComfyUIClient` with the URL from `COMFYUI_BASE_URL`.
2. It passes the client and a validator down to `generate_images_for_briefs`.
3. It does **not** construct a bedrock-runtime client (assert no `bedrock-runtime` usage — e.g. patch
   `boto3.client` and assert it is only called with `"s3"`).
4. `generate_single_brief_image_task` does the same for `generate_image_for_brief`.

---

### Phase 8 — Deployment, configuration, docs

1. **`deploy/dgx/docker-compose.yml`** — add the `vllm-vision` service (Phase 1).
2. **`deploy/dgx/openai-proxy.py`** — add the `qwen_vl_ocr` route (Phase 1).
3. **`deploy/dgx/download-flux.sh`** — add the BiRefNet download into `models/background_removal/`.
4. **`deploy/dgx/COMFYUI_SETUP.md`** — document the verified model layout, the new service, the ports, and the
   `COMFYUI_*` / `VISION_LLM_*` env vars. Remove the stale claim that the vLLM services serve
   "Qwen3-30B-A3B NVFP4" (the compose file now serves `granite-4.2-8b` / `Qwen3-14B`).
5. **`README.md`** — add the env-var table and a short "Image generation on the DGX" section.
6. **`docker-compose.yml`** (app) — add the env vars to the `app` service so the container can reach the DGX.

**Env var reference (new)**

| Variable | Default | Purpose |
|----------|---------|---------|
| `COMFYUI_BASE_URL` | `http://localhost:8188` | ComfyUI on the DGX |
| `COMFYUI_TIMEOUT_SECONDS` | `300` | Per-workflow wall-clock limit |
| `COMFYUI_POLL_INTERVAL_SECONDS` | `1.0` | History poll interval |
| `COMFYUI_UNET_NAME` | `flux1-schnell.safetensors` | Flux diffusion model |
| `COMFYUI_CLIP_NAME1` | `t5xxl_fp8_e4m3fn.safetensors` | T5-XXL text encoder |
| `COMFYUI_CLIP_NAME2` | `clip_l.safetensors` | CLIP-L text encoder |
| `COMFYUI_VAE_NAME` | `ae.safetensors` | Flux VAE |
| `COMFYUI_BG_MODEL_NAME` | `birefnet.safetensors` | BiRefNet background removal |
| `COMFYUI_STEPS` | `4` | Flux Schnell step count |
| `COMFYUI_CFG` | `1.0` | Flux guidance scale (Schnell ⇒ 1.0) |
| `COMFYUI_WIDTH` / `COMFYUI_HEIGHT` | `1024` | Output size |
| `VISION_LLM_BASE_URL` | `LLM_BASE_URL` | OpenAI-compatible base URL (the DGX proxy) |
| `VISION_LLM_MODEL` | `qwen_vl_ocr` | Vision model name |
| `VISION_LLM_API_KEY` | `LLM_API_KEY` | Usually a dummy for a local proxy |
| `TEXT_VALIDATION_ENABLED` | `true` | Kill switch for the OCR gate |
| `TEXT_MATCH_THRESHOLD` | `0.85` | Similarity threshold for a text match |
| `MAX_TEXT_REGENERATION_ATTEMPTS` | `2` | Retries after a text mismatch (3 generations max) |
| `LEONARDO_API_KEY`, `LEONARDO_MODEL` | — | **REMOVED** |
| `BEDROCK_ACCESS_KEY_ID`, `BEDROCK_SECRET_ACCESS_KEY`, `BEDROCK_IMAGE_MODEL_ID`, `AWS_BEDROCK_REGION` | — | **REMOVED** (boto3 stays for S3) |

---

### Phase 9 — AWS decommissioning (manual ops, outside the PR)

Only after Phase 7 is merged and the DGX path has run in production:

1. Confirm no traffic: `aws bedrock list-invocation-metrics` / app logs show no `bedrock-runtime` calls.
2. Deregister the Stability marketplace subscriptions:
   ```bash
   aws bedrock list-marketplace-models --query 'marketplaceModelSummaries[?modelName==`stability.sd3-5-large-v1:0`]'
   aws bedrock list-marketplace-models --query 'marketplaceModelSummaries[?modelName==`us.stability.stable-image-remove-background-v1:0`]'
   # cancel the subscription in the AWS Marketplace console (Billing → Marketplace subscriptions)
   ```
3. Delete the IAM user and its access key:
   ```bash
   aws iam list-access-keys --user-name store-listing-bedrock
   aws iam delete-access-key --user-name store-listing-bedrock --access-key-id <AKID>
   aws iam delete-user --user-name store-listing-bedrock
   ```
4. Remove `BEDROCK_*` / `AWS_BEDROCK_REGION` from every secret store, deployment manifest, and CI variable.
5. Record the decommission in the issue.

---

## 5. Test matrix

| Layer | File | Covers |
|-------|------|--------|
| Unit | `tests/unit/test_comfyui_workflows.py` | graph shapes, node ids, wiring (pure functions) |
| Unit | `tests/unit/test_comfyui_client.py` | queue → poll → fetch; 400 node_errors; execution error; timeout; view failure; upload + bg removal; health |
| Unit | `tests/unit/test_text_validation.py` | normalization, similarity, pass/fail/error verdicts, JPEG encoding, no-call for textless briefs |
| Unit | `tests/unit/test_image_generation.py` (rewrite) | prompt folding per layout, font colour default, feedback, client error handling, dtf raw/transparent |
| Unit | `tests/unit/test_image_pipeline.py` (rewrite) | upload flow, dtf transparency, OCR gate accept/retry/exhaust, no-upload on failure, bg-removal failure tolerance |
| Unit | `tests/unit/test_tasks.py` (new) | client construction from env, no bedrock-runtime client, wiring into the pipeline |
| Unit | `tests/unit/test_leonardo_client.py` | **DELETE** |
| Unit | `tests/unit/test_text_compositor.py` | **DELETE** |
| Integration | `tests/integration/test_comfyui_integration_test.py` (new) | live DGX: real Flux generation returns a valid PNG; BiRefNet returns an RGBA PNG; the vision model returns the headline that was rendered. Skipped unless `COMFYUI_BASE_URL` is set and reachable |
| E2E | `tests/e2e/…` (new) | full brief → image in S3 → DB keys; no Leonardo/Bedrock calls. Requires the full local stack |

**Integration test sketch**

```python
@pytest.mark.skipif(not os.environ.get("COMFYUI_BASE_URL"), reason="requires a live ComfyUI")
def test_flux_generation_returns_png() -> None:
    client = ComfyUIClient(base_url=os.environ["COMFYUI_BASE_URL"])
    result = client.generate_image(prompt="a red bicycle on a beach", width=512, height=512)
    assert result.error is None
    assert result.image_bytes is not None
    assert Image.open(BytesIO(result.image_bytes)).format == "PNG"
```

---

## 6. Risks and mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Gated Hugging Face repos 401/403 without a licence click | **Observed** | High | Use the public mirrors in §2.7 (`Comfy-Org/flux1-schnell`, `Comfy-Org/Lumina_Image_2.0_Repackaged`). Never point the script at `black-forest-labs/*` — the DGX's own token gets a 403 there. |
| `download-flux.sh` calls `sudo`, which fails non-interactively | **Observed** | High | Remove `sudo` from the script — the data dir is already uid 1000, matching the container user. Keep `set -euo pipefail` so any partial run fails loudly. |
| **Flux Schnell renders text poorly** (4-step distilled model) → the OCR gate rejects most briefs | Medium | High | Measure the first-attempt pass rate during Phase 1 with a 20-brief sample. If < 70 %, switch to **Flux Dev** (28 steps, `unet_name=flux1-dev.safetensors`, `steps=28`, `cfg=1.0`) — same graph, one env-var change. If still short, open issue **#133** (local text-render model) as the follow-up. |
| BiRefNet checkpoint does not load into ComfyUI's core loader | Low | Medium | Verify the `bb.layers.1.blocks.0.attn.relative_position_index` key after download; fall back to a BiRefNet-distilled checkpoint, or to the `ComfyUI-RMBG` custom node, or keep Bedrock remove-bg as a temporary stopgap. |
| Qwen2.5-VL does not run on sm_121/ARM64 | Low | High | Try `Qwen/Qwen2.5-VL-7B-Instruct` in `bfloat16` first; fall back to `Qwen/Qwen2.5-VL-3B-Instruct` (smaller) or a `granite-vision` build. If vision on the DGX is not viable, fall back to Tesseract in the app container (decision D1 revisited) — the `TextValidator` interface isolates the change to one class. |
| GPU memory contention between ComfyUI, 2× vLLM and the VL model on 128 GB unified memory | **Observed** | Medium | The first live boot of the vision model died with `No available memory for the cache blocks`: 15.7 GB of bf16 weights plus 9 GB of CUDA graphs did not fit in the 0.15 budget, and both vLLM services started at once and raced for the unified pool. Fixed by raising the budgets (0.25 / 0.30), adding `--enforce-eager`, capping `--limit-mm-per-prompt` at `{"image":1,"video":0}` and serialising the startup with `depends_on: service_healthy`. |
| No fallback once Leonardo/Bedrock are removed (decision D2) | — | High | Documented, accepted. Mitigate operationally: the DGX health check (`/system_stats`) plus alerting on `generate_design_images_task` error counts. |
| `design_briefs.scene_description` / `recommended_materials` are used in queries but missing from `schema.sql` | Medium | Low | Pre-existing drift, **out of scope for #111**. Open a separate issue; do not fix inside this PR. |
| The LLM-routing rewrite (uncommitted) conflicts with the #111 proxy change | Medium | Low | Phase 0 isolates it into its own PR and lands it first. |

---

## 7. Rollback

| Stage | Rollback |
|-------|---------|
| During Phases 2–6 (branch not merged) | `git revert` the PR / drop the branch |
| After Phase 7, before Phase 9 | Re-deploy the previous container image; the DGX keeps serving ComfyUI and the app falls back to the previous code path (Leonardo/Bedrock) |
| After Phase 9 (AWS decommissioned) | Re-subscribe the Stability marketplace models and re-create the IAM user — documented in the Phase 9 runbook; the image bytes in S3 are unaffected |

---

## 8. Issue checklist → phase mapping

| #111 task | Phase |
|-----------|-------|
| Deploy ComfyUI on DGX with Flux Schnell models (confirm model/repo layout) | **Phase 1** (layout already confirmed in §2.4) |
| Implement real ComfyUI workflow invocation in `ComfyUIClient` | Phase 2 + 3 |
| Make DGX ComfyUI the **primary** generation path; demote Leonardo/Ideogram | Phase 4 + 6 + 7 |
| **Remove text compositor workaround**; fold `layout_type`/`headline_text`/`tagline_text`/`font_color` into the prompt | Phase 4 (file deleted in Phase 4) |
| **OCR text validation** with auto-regenerate + max retry | Phase 5 + 6 |
| Move background removal to the DGX (or document why it stays) | Phase 1 (BiRefNet model) + Phase 2/3/6 (graph) |
| Remove `leonardo_client.py`, `LEONARDO_API_KEY`, `LEONARDO_MODEL` | Phase 4 + 7 |
| Remove `boto3`/Bedrock from design generation | Phase 7 (boto3 retained for S3) |
| Update design generation tests | Phases 2–7 |
| Unenroll Stability marketplace models; remove the `store-listing-bedrock` IAM user | **Phase 9** (manual ops) |

**Related issues**
- **#133** (open) "render text with a local model on DGX instead of compositor overlay" — **partially superseded** by
  this plan: the compositor is deleted and text is rendered natively by Flux. Keep #133 open as the contingency
  if the OCR pass rate in Phase 1 is too low; otherwise close it as superseded by #111.
- **#134** (closed) Ideogram routing — its code is removed here.
- **#120** (closed) Leonardo switch — superseded by #111.

---

## 9. Definition of done

- [ ] `ComfyUIClient` + workflow builders exist, fully unit-tested, and the live-DGX integration test passes.
- [ ] The DGX serves Flux Schnell **and** BiRefNet **and** a vision model; the model lists are non-empty.
- [ ] `generate_image` folds layout/headline/tagline/font colour into the Flux prompt.
- [ ] The OCR gate runs before upload, retries up to `MAX_TEXT_REGENERATION_ATTEMPTS`, and never uploads an
      image whose text failed validation.
- [ ] `text_compositor.py`, `leonardo_client.py`, the Ideogram routing, and every Bedrock image call are gone.
- [ ] `boto3` is used for S3 only; no `bedrock-runtime` client anywhere in `src/`.
- [ ] `ruff check .`, `ruff format --check .`, `pyright`, and `pytest` all pass.
- [ ] `deploy/dgx/COMFYUI_SETUP.md` and `README.md` document the verified layout and every new env var.
- [ ] The Stability marketplace subscription and the `store-listing-bedrock` IAM user are deleted (Phase 9).
- [ ] `grep -rn "leonardo\|bedrock\|ideogram\|text_compositor" src/ tests/` returns nothing.

---

## 10. Why the old plan was wrong

The superseded `implementation_plan_comfyui_dgx.md` must not be followed. Its defects:

| Old plan said | Reality |
|----------------|---------|
| `POST /dynamic_ui/generate` with `{inputs, parameters}` | **No such endpoint.** The real API is `POST /prompt` with an API-format graph → `GET /history/{id}` → `GET /view`. (The `dynamic_ui` idea came from a *different* project's proxy.) |
| Use `requests` | `requests` is not a direct dependency; `httpx` is, and is what the rest of the codebase uses. |
| `asyncio.run(comfyui_client.generate_image(...))` from a sync function | The client should be **synchronous** — no event-loop juggling, no `asyncio.get_event_loop()` warnings. |
| "ComfyUIClient exists (untracked)" / "`USE_DGX_IMAGE_GENERATION` gates DGX generation in `image_generation.py` and `tasks.py`" | **None of that exists** in the tree. Only a stale `.pyc`. The issue's "Current scaffolding" section is wrong. |
| `UNETLoader` supplies the CLIP and VAE | In ComfyUI 0.37.1 `UNETLoader` outputs **only** `MODEL`; `DualCLIPLoader` + `VAELoader` are required. |
| Plan omits the OCR gate, the compositor deletion, the background-removal migration, and the Bedrock retirement | Those are the bulk of issue #111 and are covered in Phases 4–9. |
| `model=model_id or BEDROCK_MODEL_ID` (a Bedrock model id) | Nonsense for ComfyUI; the model is a **filename** in `models/unet/`. |
| "Total: ~5–7 days" | The real long pole is **model provisioning on the DGX** (≈25 GB download + a 16 GB VL model) and the text-quality measurement, not the Python work. |
