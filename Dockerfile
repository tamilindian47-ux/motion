FROM runpod/base:0.8.1-cuda12.1.0

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

WORKDIR /comfyui

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget curl ffmpeg libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Clone ComfyUI core
RUN git clone https://github.com/comfyanonymous/ComfyUI.git . && \
    pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir runpod requests websocket-client

# Install essential Custom Nodes for video, depth, and identity
WORKDIR /comfyui/custom_nodes
RUN git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git && \
    pip install --no-cache-dir -r ComfyUI-VideoHelperSuite/requirements.txt

RUN git clone https://github.com/Fannovel16/comfyui_controlnet_aux.git && \
    pip install --no-cache-dir -r comfyui_controlnet_aux/requirements.txt

RUN git clone https://github.com/ZHO-ZHO-ZHO/ComfyUI-InstantID.git && \
    pip install --no-cache-dir -r ComfyUI-InstantID/requirements.txt

RUN git clone https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved.git

WORKDIR /comfyui

# Bake Uncensored Base Model (Realistic Vision v5.1 / SD1.5)
RUN wget -q --show-progress -O models/checkpoints/realisticVisionV51_v51VAE.safetensors \
    https://civitai.com/api/download/models/130072?type=Model&format=SafeTensor

# Bake AnimateDiff Motion Model (Lightning 4-step for fast serverless execution)
RUN mkdir -p models/animatediff_models && \
    wget -q --show-progress -O models/animatediff_models/animatediff_lightning_4step.safetensors \
    https://huggingface.co/ByteDance/AnimateDiff-Lightning/resolve/main/animatediff_lightning_4step_diffusers.safetensors

# Bake ControlNet Depth Model
RUN mkdir -p models/controlnet && \
    wget -q --show-progress -O models/controlnet/control_v11f1p_sd15_depth.pth \
    https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11f1p_sd15_depth.pth

# Bake InstantID Weights & Antelopev2 Face Analysis
RUN mkdir -p models/instantid && \
    wget -q --show-progress -O models/instantid/ip-adapter.bin \
    https://huggingface.co/InstantX/InstantID/resolve/main/ip-adapter.bin

RUN mkdir -p models/controlnet && \
    wget -q --show-progress -O models/controlnet/control_instant_id_sd15.bin \
    https://huggingface.co/InstantX/InstantID/resolve/main/ControlNetModel/diffusion_pytorch_model.bin

COPY handler.py /comfyui/handler.py
COPY workflow_api.json /comfyui/workflow_api.json

CMD ["python3", "-u", "/comfyui/handler.py"]
