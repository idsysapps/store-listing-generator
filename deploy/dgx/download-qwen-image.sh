#!/bin/bash
set -euo pipefail

# Downloads Qwen-Image 2.1 (INT8 ComfyUI pack) for the typographic wordmark layer.
#
# Runtime uses 2.1 only (issue #339). Older Qwen-Image 1.0 files may remain on
# disk and are left alone by this script.
#
#   - qwen_image_2.1_int8_convrot.safetensors   Comfy-Org/Qwen-Image-2.1  ~6.8 GB
#   - qwen3vl_8b_int8_convrot.safetensors       Comfy-Org/Qwen-Image-2.1  ~8.7 GB
#   - qwen_image_2.1_vae_bf16.safetensors       Comfy-Org/Qwen-Image-2.1  ~645 MB
#
# Total ~16 GB. The script is idempotent and resumable: re-running continues a
# partial download (wget -c) and skips files that are already complete.

COMFYUI_DATA="${COMFYUI_DATA_PATH:-/opt/comfyui-data}"
MODELS_DIR="$COMFYUI_DATA/models"
HF_BASE="https://huggingface.co/Comfy-Org/Qwen-Image-2.1/resolve/main"

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

echo "==> Downloading the Qwen3-VL-8B text encoder (~8.7 GB)..."
download \
    "$HF_BASE/text_encoders/qwen3vl_8b_int8_convrot.safetensors" \
    "$MODELS_DIR/text_encoders/qwen3vl_8b_int8_convrot.safetensors" \
    9350798360

echo "==> Downloading the Qwen-Image 2.1 UNET (~6.8 GB)..."
download \
    "$HF_BASE/diffusion_models/qwen_image_2.1_int8_convrot.safetensors" \
    "$MODELS_DIR/diffusion_models/qwen_image_2.1_int8_convrot.safetensors" \
    7256783064

echo "==> Downloading the Qwen-Image 2.1 VAE (~645 MB)..."
download \
    "$HF_BASE/vae/qwen_image_2.1_vae_bf16.safetensors" \
    "$MODELS_DIR/vae/qwen_image_2.1_vae_bf16.safetensors" \
    675509688

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
verify "$MODELS_DIR/text_encoders/qwen3vl_8b_int8_convrot.safetensors" 9350798360
verify "$MODELS_DIR/diffusion_models/qwen_image_2.1_int8_convrot.safetensors" 7256783064
verify "$MODELS_DIR/vae/qwen_image_2.1_vae_bf16.safetensors" 675509688

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
