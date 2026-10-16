#!/bin/bash
set -euo pipefail

# Downloads Flux Schnell model files into the ComfyUI data directory.
# Run once after building and starting the stack.

COMFYUI_DATA="${COMFYUI_DATA_PATH:-/opt/comfyui-data}"
MODELS_DIR="$COMFYUI_DATA/models"

echo "==> Creating model directories"
sudo mkdir -p "$MODELS_DIR"/{unet,clip,vae}

echo "==> Downloading Flux Schnell UNET (~23GB)..."
if [ ! -f "$MODELS_DIR/unet/flux1-schnell.safetensors" ]; then
    sudo wget -q --show-progress -P "$MODELS_DIR/unet/" \
        "https://huggingface.co/black-forest-labs/FLUX.1-schnell/resolve/main/flux1-schnell.safetensors"
else
    echo "    - Already downloaded, skipping"
fi

echo "==> Downloading CLIP-L text encoder..."
if [ ! -f "$MODELS_DIR/clip/clip_l.safetensors" ]; then
    sudo wget -q --show-progress -P "$MODELS_DIR/clip/" \
        "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/clip_l.safetensors"
else
    echo "    - Already downloaded, skipping"
fi

echo "==> Downloading T5-XXL text encoder (fp8)..."
if [ ! -f "$MODELS_DIR/clip/t5xxl_fp8_e4m3fn.safetensors" ]; then
    sudo wget -q --show-progress -P "$MODELS_DIR/clip/" \
        "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp8_e4m3fn.safetensors"
else
    echo "    - Already downloaded, skipping"
fi

echo "==> Downloading VAE..."
if [ ! -f "$MODELS_DIR/vae/ae.safetensors" ]; then
    sudo wget -q --show-progress -P "$MODELS_DIR/vae/" \
        "https://huggingface.co/black-forest-labs/FLUX.1-schnell/resolve/main/ae.safetensors"
else
    echo "    - Already downloaded, skipping"
fi

IP=$(hostname -I | awk '{print $1}')
echo ""
echo "==> Done. Flux Schnell model files downloaded to $MODELS_DIR"
echo "    ComfyUI: http://$IP:8188"
echo "    API:     http://$IP:8188/api/prompt"
echo ""
echo "    Restart ComfyUI to pick up models:"
echo "    sudo docker compose -f /opt/trtllm/docker-compose.yml restart comfyui"
