#!/bin/bash
set -euo pipefail

# Downloads the FLUX.2 [klein] 4B files used for the wordmark bake-off.
#
# All four files are Apache 2.0 and public - no HF token and no licence click.
# The 9B and 32B FLUX.2 variants are deliberately absent: they are released
# under the FLUX Non-Commercial License, so their output cannot be printed on
# a product we sell.
#
#   - qwen_3_4b.safetensors                  Comfy-Org/flux2-klein    8.0 GB
#   - flux-2-klein-4b.safetensors            Comfy-Org/flux2-klein    7.8 GB  (distilled, 4 steps)
#   - flux-2-klein-base-4b.safetensors       Comfy-Org/flux2-klein    7.8 GB  (base, 20 steps)
#   - flux2-vae.safetensors                  Comfy-Org/flux2-dev      336 MB
#
# The official ComfyUI template (image_flux2_klein_text_to_image.json) uses the
# bf16 single-file checkpoints, not the fp8 ones, so the bake-off compares the
# model at its best. The fp8 files are an option later if memory is tight.
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

echo "==> Downloading the Qwen3-4B text encoder (~8.0 GB)..."
download \
    "$HF_BASE/Comfy-Org/flux2-klein/resolve/main/split_files/text_encoders/qwen_3_4b.safetensors" \
    "$MODELS_DIR/text_encoders/qwen_3_4b.safetensors" \
    8044982048

echo "==> Downloading FLUX.2 Klein 4B distilled (~7.8 GB)..."
download \
    "$HF_BASE/Comfy-Org/flux2-klein/resolve/main/split_files/diffusion_models/flux-2-klein-4b.safetensors" \
    "$MODELS_DIR/diffusion_models/flux-2-klein-4b.safetensors" \
    7751105712

echo "==> Downloading FLUX.2 Klein 4B base (~7.8 GB)..."
download \
    "$HF_BASE/Comfy-Org/flux2-klein/resolve/main/split_files/diffusion_models/flux-2-klein-base-4b.safetensors" \
    "$MODELS_DIR/diffusion_models/flux-2-klein-base-4b.safetensors" \
    7751105712

echo "==> Downloading the FLUX.2 VAE (~336 MB)..."
download \
    "$HF_BASE/Comfy-Org/flux2-dev/resolve/main/split_files/vae/flux2-vae.safetensors" \
    "$MODELS_DIR/vae/flux2-vae.safetensors" \
    336213556

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
verify "$MODELS_DIR/text_encoders/qwen_3_4b.safetensors" 8044982048
verify "$MODELS_DIR/diffusion_models/flux-2-klein-4b.safetensors" 7751105712
verify "$MODELS_DIR/diffusion_models/flux-2-klein-base-4b.safetensors" 7751105712
verify "$MODELS_DIR/vae/flux2-vae.safetensors" 336213556

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
