FROM pytorch/pytorch:2.7.0-cuda12.8-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/tmp/huggingface \
    TOKENIZERS_PARALLELISM=false

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    ca-certificates \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Clone ComfyUI Core
RUN git clone --depth 1 https://github.com/comfyanonymous/ComfyUI.git /app/ComfyUI \
    && pip install --no-cache-dir -r /app/ComfyUI/requirements.txt

# Clone required custom nodes
RUN git clone --depth 1 https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git /app/ComfyUI/custom_nodes/ComfyUI-VideoHelperSuite \
    && git clone --depth 1 https://github.com/city96/ComfyUI-GGUF.git /app/ComfyUI/custom_nodes/ComfyUI-GGUF \
    && git clone --depth 1 https://github.com/kijai/ComfyUI-KJNodes.git /app/ComfyUI/custom_nodes/ComfyUI-KJNodes

# Install custom node dependencies (CRITICAL for UnetLoaderGGUF and video encoding)
RUN pip install --no-cache-dir gguf imageio-ffmpeg opencv-python-headless \
    && pip install --no-cache-dir -r /app/ComfyUI/custom_nodes/ComfyUI-VideoHelperSuite/requirements.txt \
    && pip install --no-cache-dir -r /app/ComfyUI/custom_nodes/ComfyUI-KJNodes/requirements.txt

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY workflow_api.json /app/workflow_api.json
COPY handler.py /app/handler.py

CMD ["python", "-u", "/app/handler.py"]
