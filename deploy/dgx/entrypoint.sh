#!/bin/bash
set -euo pipefail

TRTLLM_DIR="/workspace/TensorRT-LLM"
MODEL_REPO="$TRTLLM_DIR/triton_backend/all_models/llmapi"
TEMPLATE="$TRTLLM_DIR/_model_template"

CURRENT_VER=$(pip show tensorrt-llm 2>/dev/null | grep "^Version:" | awk '{print $2}' || echo "unknown")
echo "==> TensorRT-LLM version: $CURRENT_VER"

# --- GB10 unified memory fixes (v1.3.x) ---
# GB10 has unified CPU/GPU memory. torch.cuda.mem_get_info() returns (0, 0)
# in some process contexts, breaking TRT-LLM's KV cache allocation.
# We apply TWO fixes:
#   1. Direct sed-patch on TRT-LLM source to handle 0-memory gracefully
#   2. sitecustomize.py as a belt-and-suspenders fallback
GB10_MEM_BYTES=$(grep MemTotal /proc/meminfo | awk '{print int($2 * 1024 * 0.95)}')
echo "==> GB10 memory fix: system has ${GB10_MEM_BYTES} bytes"

TRTLLM_PKG="/opt/venv-tritonserver/lib/python3.12/site-packages/tensorrt_llm"

# Patch 1: Fix mem_get_info at the source — inject wrapper into _util.py
UTIL_PY="$TRTLLM_PKG/_torch/pyexecutor/_util.py"
if [ -f "$UTIL_PY" ] && ! grep -q "GB10_MEM_FIX" "$UTIL_PY"; then
    # Add the wrapper function after imports
    sed -i '1s/^/# GB10_MEM_FIX: patch torch.cuda.mem_get_info for unified memory\nimport torch.cuda as _tc\n_orig_mginfo = _tc.mem_get_info\ndef _gb10_mem_get_info(device=None):\n    f, t = _orig_mginfo(device)\n    if t == 0:\n        t = '"${GB10_MEM_BYTES}"'\n        f = t\n    return (f, t)\n_tc.mem_get_info = _gb10_mem_get_info\n/' "$UTIL_PY"
    echo "    - Patched _util.py with mem_get_info wrapper"
fi

# Patch 2: Same patch in resource_manager.py (called in worker processes)
RESMGR_PY="$TRTLLM_PKG/_torch/pyexecutor/resource_manager.py"
if [ -f "$RESMGR_PY" ] && ! grep -q "GB10_MEM_FIX" "$RESMGR_PY"; then
    sed -i '1s/^/# GB10_MEM_FIX: patch torch.cuda.mem_get_info for unified memory\nimport torch.cuda as _tc\n_orig_mginfo2 = _tc.mem_get_info\ndef _gb10_mem_get_info2(device=None):\n    f, t = _orig_mginfo2(device)\n    if t == 0:\n        t = '"${GB10_MEM_BYTES}"'\n        f = t\n    return (f, t)\n_tc.mem_get_info = _gb10_mem_get_info2\n/' "$RESMGR_PY"
    echo "    - Patched resource_manager.py with mem_get_info wrapper"
fi

# Patch 3: Same patch in py_executor_creator.py
CREATOR_PY="$TRTLLM_PKG/_torch/pyexecutor/py_executor_creator.py"
if [ -f "$CREATOR_PY" ] && ! grep -q "GB10_MEM_FIX" "$CREATOR_PY"; then
    sed -i '1s/^/# GB10_MEM_FIX: patch torch.cuda.mem_get_info for unified memory\nimport torch.cuda as _tc\n_orig_mginfo3 = _tc.mem_get_info\ndef _gb10_mem_get_info3(device=None):\n    f, t = _orig_mginfo3(device)\n    if t == 0:\n        t = '"${GB10_MEM_BYTES}"'\n        f = t\n    return (f, t)\n_tc.mem_get_info = _gb10_mem_get_info3\n/' "$CREATOR_PY"
    echo "    - Patched py_executor_creator.py with mem_get_info wrapper"
fi

# Sitecustomize fallback (catches any other torch.cuda.mem_get_info callers)
SITE_DIR=$(python3 -c "import site; print(site.getsitepackages()[0])")
cat > "$SITE_DIR/sitecustomize.py" << PYEOF
try:
    import torch.cuda
    _orig_mem_get_info = torch.cuda.mem_get_info
    def _patched_mem_get_info(device=None):
        free, total = _orig_mem_get_info(device)
        if total == 0:
            total = ${GB10_MEM_BYTES}
            free = total
        return (free, total)
    torch.cuda.mem_get_info = _patched_mem_get_info
except Exception:
    pass
PYEOF
echo "    - Installed sitecustomize.py at $SITE_DIR"

# Patch 5: Skip attention warmup on XQA JIT failure (safety net)
# If attn_backend "unfused" works, this patch is never triggered.
# If TRT-LLM falls back to TrtllmAttention, this catches the XQA NVRTC error.
PYEXEC="$TRTLLM_PKG/_torch/pyexecutor/py_executor.py"
if [ -f "$PYEXEC" ] && ! grep -q "GB10_WARMUP_SKIP" "$PYEXEC"; then
    python3 << 'WARMUP_PATCH'
path = "/opt/venv-tritonserver/lib/python3.12/site-packages/tensorrt_llm/_torch/pyexecutor/py_executor.py"
with open(path, 'r') as f:
    lines = f.readlines()
if not any('GB10_WARMUP_SKIP' in l for l in lines):
    target = 'self.model_engine.warmup(self.resource_manager)'
    for i, line in enumerate(lines):
        if target in line:
            indent = line[:len(line) - len(line.lstrip())]
            sub = indent + '    '
            replacement = [
                f"{indent}try:  # GB10_WARMUP_SKIP\n",
                f"{sub}self.model_engine.warmup(self.resource_manager)\n",
                f"{indent}except RuntimeError as _e:\n",
                f"{sub}if 'XqaJit' in str(_e) or 'NVRTC' in str(_e):\n",
                f"{sub}    import logging\n",
                f"{sub}    logging.getLogger('tensorrt_llm').warning(\n",
                f"{sub}        'GB10: Skipping warmup (XQA JIT unavailable on SM 12.1)')\n",
                f"{sub}else:\n",
                f"{sub}    raise\n",
            ]
            lines[i:i+1] = replacement
            break
    with open(path, 'w') as f:
        f.writelines(lines)
    print("    - Patched py_executor.py: warmup XQA skip")
WARMUP_PATCH
fi

# --- Step 1: Clone triton backend scripts (first run only) ---
if [ ! -d "$TEMPLATE" ] && [ ! -d "$MODEL_REPO/tensorrt_llm" ]; then
    CLONE_TAG="v$CURRENT_VER"
    echo "==> Cloning TensorRT-LLM $CLONE_TAG (first run)..."
    rm -rf "$TRTLLM_DIR"/* "$TRTLLM_DIR"/.[!.]* 2>/dev/null || true
    git clone --depth 1 --branch "$CLONE_TAG" https://github.com/NVIDIA/TensorRT-LLM.git /tmp/trtllm-clone
    mv /tmp/trtllm-clone/* /tmp/trtllm-clone/.[!.]* "$TRTLLM_DIR"/ 2>/dev/null || true
    rm -rf /tmp/trtllm-clone
fi

# --- Step 2: Apply version-specific patches ---
if [[ "$CURRENT_VER" == 1.2.* ]]; then
    echo "==> Applying v1.2.x patches for GB10 SM 12.1..."
    MODELENGINE="$TRTLLM_PKG/_torch/pyexecutor/model_engine.py"
    if [ -f "$MODELENGINE" ]; then
        sed -i 's/self\._run_cuda_graph_warmup(resource_manager)/pass  # disabled for GB10/' "$MODELENGINE"
        echo "    - Patched CUDA graph warmup"
    fi
fi

if [[ "$CURRENT_VER" == 1.3.* ]]; then
    echo "==> Applying v1.3.x patches for GB10 SM 12.1..."

    # XQA JIT fails on GB10 — NVRTC can't compile attention kernels for SM 12.1.
    # Set every known XQA-disable env var (inherited by MPI worker processes).
    export TRTLLM_DISABLE_XQA=1
    export TRTLLM_DISABLE_FUSED_ATTN=1
    export TRTLLM_GEN_XQA_JIT_DISABLED=1
    echo "    - Set TRTLLM_DISABLE_XQA=1, TRTLLM_DISABLE_FUSED_ATTN=1, TRTLLM_GEN_XQA_JIT_DISABLED=1"
fi

# --- Step 3: Set up model repository ---
echo "==> Setting up model repository..."

if [ -d "$MODEL_REPO/tensorrt_llm" ] && [ ! -d "$TEMPLATE" ]; then
    mv "$MODEL_REPO/tensorrt_llm" "$TEMPLATE"
elif [ -d "$MODEL_REPO/tensorrt_llm" ]; then
    rm -rf "$MODEL_REPO/tensorrt_llm"
fi

rm -rf "$MODEL_REPO/qwen3_coder" "$MODEL_REPO/qwen38_briefs"

cp -r "$TEMPLATE" "$MODEL_REPO/qwen3_coder"
sed -i 's/name: "tensorrt_llm"/name: "qwen3_coder"/g' "$MODEL_REPO/qwen3_coder/config.pbtxt" 2>/dev/null || true
cp /model-configs/qwen3-coder.yaml "$MODEL_REPO/qwen3_coder/1/model.yaml"

cp -r "$TEMPLATE" "$MODEL_REPO/qwen38_briefs"
sed -i 's/name: "tensorrt_llm"/name: "qwen38_briefs"/g' "$MODEL_REPO/qwen38_briefs/config.pbtxt" 2>/dev/null || true
cp /model-configs/qwen38-briefs.yaml "$MODEL_REPO/qwen38_briefs/1/model.yaml"

echo "==> Model configs:"
echo "--- qwen3_coder ---"
cat "$MODEL_REPO/qwen3_coder/1/model.yaml"
echo "--- qwen38_briefs ---"
cat "$MODEL_REPO/qwen38_briefs/1/model.yaml"

# --- Step 4: Launch ---
echo ""
echo "==> Launching Triton (no-mpi, single GPU)..."
cd /workspace
python3 "$TRTLLM_DIR/triton_backend/scripts/launch_triton_server.py" \
    --no-mpi \
    --tensorrt_llm_model_name qwen3_coder,qwen38_briefs \
    --model_repo="$MODEL_REPO"

echo "==> Starting OpenAI-compatible proxy on port 9000..."
python3 /openai-proxy.py &

echo "==> Triton server started, monitoring..."
while pgrep -x tritonserver > /dev/null 2>&1; do
    sleep 30
done
echo "==> tritonserver exited"
