#!/bin/bash
set -euo pipefail

INSTALL_DIR="/opt/trtllm"
COMFYUI_DATA="/opt/comfyui-data"
COMFYUI_BUILD_DIR="/opt/comfyui-build"

echo "==> Setting up DGX vLLM + ComfyUI stack"

echo "==> Creating directories"
sudo mkdir -p "$INSTALL_DIR"
sudo mkdir -p "$COMFYUI_DATA"/{models,custom_nodes,output,input}
sudo mkdir -p "$COMFYUI_DATA"/models/{checkpoints,diffusion_models,text_encoders,vae,background_removal,loras}
# Legacy folder names - the ComfyUI build scans these as aliases for
# diffusion_models (unet) and text_encoders (clip).
sudo mkdir -p "$COMFYUI_DATA"/models/{unet,clip}

echo "==> Copying config files"
sudo cp docker-compose.yml "$INSTALL_DIR/"
sudo cp openai-proxy.py "$INSTALL_DIR/"

if [ ! -f "$INSTALL_DIR/.env" ]; then
    echo ""
    read -rp "Enter your HuggingFace token (hf_...): " HF_TOKEN
    cat <<EOF | sudo tee "$INSTALL_DIR/.env" > /dev/null
HUGGING_FACE_HUB_TOKEN=$HF_TOKEN
COMFYUI_DATA_PATH=$COMFYUI_DATA
EOF
    sudo chmod 600 "$INSTALL_DIR/.env"
else
    if ! grep -q COMFYUI_DATA_PATH "$INSTALL_DIR/.env"; then
        echo "COMFYUI_DATA_PATH=$COMFYUI_DATA" | sudo tee -a "$INSTALL_DIR/.env" > /dev/null
    fi
fi

echo "==> Building ComfyUI ARM64 image for GB10..."
if ! docker image inspect comfyui-spark:latest > /dev/null 2>&1; then
    if [ ! -d "$COMFYUI_BUILD_DIR" ]; then
        sudo git clone https://github.com/slangevi/comfyui-spark.git "$COMFYUI_BUILD_DIR"
    fi
    cd "$COMFYUI_BUILD_DIR"
    sudo docker compose build
    echo "    - comfyui-spark:latest built successfully"
else
    echo "    - comfyui-spark:latest already exists, skipping build"
fi

echo "==> Stopping old services"
sudo systemctl stop ollama.service 2>/dev/null || true
sudo systemctl disable ollama.service 2>/dev/null || true
sudo docker stop vllm-briefs vllm-vision comfyui vllm-proxy 2>/dev/null || true

echo "==> Starting vLLM + ComfyUI stack"
cd "$INSTALL_DIR"
sudo docker compose up -d

IP=$(hostname -I | awk '{print $1}')
echo ""
echo "==> Stack starting. First run may take several minutes for model downloads."
echo "    Monitor: sudo docker compose -f $INSTALL_DIR/docker-compose.yml logs -f"
echo ""
echo "    Services:"
echo "      vllm-briefs  (port 8002) - Qwen3-14B (design briefs)"
echo "      vllm-vision  (port 8003) - Qwen2.5-VL-7B (image text validation)"
echo "      comfyui      (port 8188) - Flux Schnell + BiRefNet"
echo "      openai-proxy (port 9000) - Unified routing"
echo ""
echo "    Test:"
echo "      curl http://$IP:9000/v1/models"
echo "      curl http://$IP:8188/system_stats"
echo ""
echo "    Next: run ./download-flux.sh to download Flux Schnell model"
