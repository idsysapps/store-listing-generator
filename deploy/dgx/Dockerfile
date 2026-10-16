FROM nvcr.io/nvidia/tritonserver:26.07-trtllm-python-py3

ARG TRTLLM_VERSION=1.3.0rc29

RUN pip install --no-cache-dir \
    --extra-index-url https://pypi.nvidia.com \
    "tensorrt-llm==${TRTLLM_VERSION}" \
    && pip install --no-cache-dir --upgrade openai

ENTRYPOINT ["/bin/bash", "/entrypoint.sh"]
