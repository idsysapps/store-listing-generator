#!/bin/bash
set -euo pipefail

# Downloads the Flux Schnell + BiRefNet model files into the ComfyUI data directory.
# Run once after building and starting the stack.
#
# Public mirrors only - no HF token and no licence click required:
#   - flux1-schnell.safetensors        Comfy-Org/flux1-schnell                     23.8 GB
#   - clip_l.safetensors               comfyanonymous/flux_text_encoders             246 MB
#   - t5xxl_fp8_e4m3fn.safetensors     comfyanonymous/flux_text_encoders            4.89 GB
#   - ae.safetensors                   Comfy-Org/Lumina_Image_2.0_Repackaged         335 MB
#   - birefnet.safetensors             ZhengPeng7/BiRefNet (model.safetensors)       444 MB
#
# NOTE: do NOT point this at black-forest-labs/FLUX.1-schnell - that repo is gated
# and returns 401/403 even with a valid HF token unless the licence was accepted.
#
# The script is idempotent and resumable: re-running it continues a partial
# download (wget -c) and skips files that are already complete.

COMFYUI_DATA="${COMFYUI_DATA_PATH:-/opt/comfyui-data}"
MODELS_DIR="$COMFYUI_DATA/models"

# The ComfyUI build on the DGX scans both the modern and the legacy folder names:
#   diffusion_models -> models/diffusion_models, models/unet
#   text_encoders    -> models/text_encoders,    models/clip
# We install into the modern folders and keep the legacy ones present for
# compatibility with older ComfyUI builds.

HF_BASE="https://huggingface.co"

download() {
    local url="$1"
    local dest="$2"
    local expected_size="$3"

    if [ -f "$dest" ]; then
        local actual_size
        actual_size=$(stat -c %s "$dest" 2>/dev/null || stat -f %z "$dest" 2>/dev/null || echo 0)
        if [ "$actual_size" = "$expected_size" ]; then
            echo "    - Already downloaded ($(du -h "$dest" | cut -f1)), skipping"
            return 0
        fi
        echo "    - Existing file is $actual_size bytes, expected $expected_size - resuming"
    fi

    wget -c --show-progress --tries=5 --timeout=60 -P "$(dirname "$dest")" "$url"
}

echo "==> Model directory: $MODELS_DIR"

if [ ! -d "$MODELS_DIR" ]; then
    echo "ERROR: $MODELS_DIR does not exist. Is ComfyUI set up on this host?" >&2
    exit 1
fi

if [ ! -w "$MODELS_DIR" ]; then
    echo "ERROR: $MODELS_DIR is not writable by the current user ($(id -un), uid $(id -u))." >&2
    echo "       The ComfyUI container runs as uid 1000 - run this script as that user." >&2
    exit 1
fi

echo "==> Creating model directories"
mkdir -p "$MODELS_DIR"/{diffusion_models,text_encoders,vae,background_removal}
mkdir -p "$MODELS_DIR"/{unet,clip}

echo "==> Downloading Flux Schnell UNET (~23.8 GB)..."
download \
    "$HF_BASE/Comfy-Org/flux1-schnell/resolve/main/flux1-schnell.safetensors" \
    "$MODELS_DIR/diffusion_models/flux1-schnell.safetensors" \
    23782506688

echo "==> Downloading T5-XXL text encoder (fp8, ~4.89 GB)..."
download \
    "$HF_BASE/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp8_e4m3fn.safetensors" \
    "$MODELS_DIR/text_encoders/t5xxl_fp8_e4m3fn.safetensors" \
    4893934904

echo "==> Downloading CLIP-L text encoder (~246 MB)..."
download \
    "$HF_BASE/comfyanonymous/flux_text_encoders/resolve/main/clip_l.safetensors" \
    "$MODELS_DIR/text_encoders/clip_l.safetensors" \
    246144152

echo "==> Downloading AE VAE (~335 MB)..."
download \
    "$HF_BASE/Comfy-Org/Lumina_Image_2.0_Repackaged/resolve/main/split_files/vae/ae.safetensors" \
    "$MODELS_DIR/vae/ae.safetensors" \
    335304388

echo "==> Downloading BiRefNet background removal model (~444 MB)..."
# Downloaded as model.safetensors and renamed - ComfyUI's core LoadBackgroundRemovalModel
# detects BiRefNet via the "bb.layers.1.blocks.0.attn.relative_position_index" key,
# which this checkpoint contains (verified against the safetensors header).
download \
    "$HF_BASE/ZhengPeng7/BiRefNet/resolve/main/model.safetensors" \
    "$MODELS_DIR/background_removal/birefnet.safetensors" \
    444473596

echo ""
echo "==> Verifying sizes"
status=0
verify() {
    local file="$1"
    local expected="$2"
    if [ ! -f "$file" ]; then
        echo "    MISSING: $file"
        status=1
        return
    fi
    local actual
    actual=$(stat -c %s "$file" 2>/dev/null || stat -f %z "$file" 2>/dev/null || echo 0)
    if [ "$actual" = "$expected" ]; then
        echo "    OK: $(basename "$file") ($actual bytes)"
    else
        echo "    SIZE MISMATCH: $file is $actual bytes, expected $expected"
        status=1
    fi
}
verify "$MODELS_DIR/diffusion_models/flux1-schnell.safetensors" 23782506688
verify "$MODELS_DIR/text_encoders/t5xxl_fp8_e4m3fn.safetensors" 4893934904
verify "$MODELS_DIR/text_encoders/clip_l.safetensors" 246144152
verify "$MODELS_DIR/vae/ae.safetensors" 335304388
verify "$MODELS_DIR/background_removal/birefnet.safetensors" 444473596

if [ "$status" -ne 0 ]; then
    echo ""
    echo "ERROR: one or more model files are missing or truncated. Re-run this script to resume." >&2
    exit 1
fi

IP="$(hostname -I 2>/dev/null | awk '{print $1}')" || IP=""
[ -n "$IP" ] || IP="<dgx-ip>"
echo ""
echo "==> Done. All model files are in place under $MODELS_DIR"
echo "    ComfyUI: http://$IP:8188"
echo "    API:     http://$IP:8188/api/prompt"
echo ""
echo "    Restart ComfyUI to pick up the models:"
echo "    docker compose -f /opt/trtllm/docker-compose.yml restart comfyui"
echo ""
echo "    Then verify:"
echo "    curl -s http://127.0.0.1:8188/models/diffusion_models"
echo "    curl -s http://127.0.0.1:8188/models/text_encoders"
echo "    curl -s http://127.0.0.1:8188/models/vae"
echo "    curl -s http://127.0.0.1:8188/models/background_removal"
