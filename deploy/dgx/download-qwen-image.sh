#!/bin/bash
set -euo pipefail

# Downloads Qwen-Image 1.0 (Apache 2.0) for the typographic wordmark layer.
#
# Qwen-Image is a 20B MMDiT that renders text far more reliably than the 4B
# FLUX.2 Klein models in the Phase 0 bake-off (20/20 exact phrases vs 13-15/20).
# All files are public - no HF token and no licence click.
#
#   - qwen_image_fp8_e4m3fn.safetensors      Comfy-Org/Qwen-Image_ComfyUI   20.4 GB
#   - qwen_2.5_vl_7b_fp8_scaled.safetensors  Comfy-Org/Qwen-Image_ComfyUI    9.4 GB
#   - qwen_image_vae.safetensors             Comfy-Org/Qwen-Image_ComfyUI    254 MB
#
# Total ~30 GB. The model is licensed Apache 2.0, so its output can be printed
# on products we sell. Do not substitute Qwen-Image 2.1 (research-only licence).
#
# The script is idempotent and resumable: re-running it continues a partial
# download (wget -c) and skips files that are already complete.

COMFYUI_DATA="${COMFYUI_DATA_PATH:-/opt/comfyui-data}"
MODELS_DIR="$COMFYUI_DATA/models"
HF_BASE="https://huggingface.co"

file_size() {
    stat -c %s "$1" 2>/dev/null || stat -f %z "$1" 2>/dev/null || echo 0
}

download() {
    local url="$1"
    local dest="$2"
    local expected_size="$3"

    local dir name url_name scratch actual_size
    dir="$(dirname "$dest")"
    name="$(basename "$dest")"
    url_name="${url##*/}"
    scratch="$dir/$url_name"

    if [ -f "$dest" ]; then
        actual_size="$(file_size "$dest")"
        if [ "$actual_size" = "$expected_size" ]; then
            echo "    - Already downloaded ($(du -h "$dest" | cut -f1)), skipping"
            return 0
        fi
        echo "    - $dest is $actual_size bytes, expected $expected_size - re-downloading"
    fi

    if [ -f "$scratch" ]; then
        actual_size="$(file_size "$scratch")"
        if [ "$actual_size" = "$expected_size" ]; then
            echo "    - Already downloaded under the upstream name, renaming"
            mv "$scratch" "$dest"
            return 0
        fi
        echo "    - Resuming $scratch from $actual_size bytes"
    fi

    wget -c --show-progress --tries=5 --timeout=60 -P "$dir" "$url"

    if [ "$url_name" != "$name" ]; then
        mv "$dir/$url_name" "$dest"
    fi
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

mkdir -p "$MODELS_DIR"/{diffusion_models,text_encoders,vae}

echo "==> Downloading the Qwen2.5-VL-7B text encoder (~9.4 GB)..."
download \
    "$HF_BASE/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors" \
    "$MODELS_DIR/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors" \
    9384670680

echo "==> Downloading the Qwen-Image UNET (~20.4 GB)..."
download \
    "$HF_BASE/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/diffusion_models/qwen_image_fp8_e4m3fn.safetensors" \
    "$MODELS_DIR/diffusion_models/qwen_image_fp8_e4m3fn.safetensors" \
    20430635136

echo "==> Downloading the Qwen-Image VAE (~254 MB)..."
download \
    "$HF_BASE/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/vae/qwen_image_vae.safetensors" \
    "$MODELS_DIR/vae/qwen_image_vae.safetensors" \
    253806246

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
    actual="$(file_size "$file")"
    if [ "$actual" = "$expected" ]; then
        echo "    OK: $(basename "$file") ($actual bytes)"
    else
        echo "    SIZE MISMATCH: $file is $actual bytes, expected $expected"
        status=1
    fi
}
verify "$MODELS_DIR/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors" 9384670680
verify "$MODELS_DIR/diffusion_models/qwen_image_fp8_e4m3fn.safetensors" 20430635136
verify "$MODELS_DIR/vae/qwen_image_vae.safetensors" 253806246

if [ "$status" -ne 0 ]; then
    echo ""
    echo "ERROR: one or more model files are missing or truncated. Re-run this script to resume." >&2
    exit 1
fi

echo ""
echo "==> Done. Restart ComfyUI to pick up the new models:"
echo "    docker compose -f /opt/trtllm/docker-compose.yml restart comfyui"
echo ""
echo "    Then verify:"
echo "    curl -s http://127.0.0.1:8188/models/diffusion_models"
echo "    curl -s http://127.0.0.1:8188/models/text_encoders"
echo "    curl -s http://127.0.0.1:8188/models/vae"
