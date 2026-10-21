FROM runpod/base:0.8.1-cuda12.1.0

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

WORKDIR /comfyui

# System dependencies for FFmpeg and OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget curl ffmpeg libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Clone ComfyUI Core
RUN git clone https://github.com/comfyanonymous/ComfyUI.git .

# Install all pinned dependencies from requirements.txt
COPY requirements.txt /comfyui/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Install Custom Nodes (ComfyUI Git repositories)
WORKDIR /comfyui/custom_nodes
RUN git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git && \
    git clone https://github.com/Fannovel16/comfyui_controlnet_aux.git && \
    git clone https://github.com/ZHO-ZHO-ZHO/ComfyUI-InstantID.git && \
    git clone https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved.git

WORKDIR /comfyui

# Download Models into the image layers (No Network Volume required)
# 1. Uncensored Base SD1.5 Checkpoint
RUN wget -q --show-progress -O models/checkpoints/realisticVisionV51_v51VAE.safetensors \
    "https://civitai.com/api/download/models/130072?type=Model&format=SafeTensor"

# 2. Fast 4-Step Motion Adapter
RUN mkdir -p models/animatediff_models && \
    wget -q --show-progress -O models/animatediff_models/animatediff_lightning_4step.safetensors \
    "https://huggingface.co/ByteDance/AnimateDiff-Lightning/resolve/main/animatediff_lightning_4step_diffusers.safetensors"

# 3. ControlNet Depth Model
RUN mkdir -p models/controlnet && \
    wget -q --show-progress -O models/controlnet/control_v11f1p_sd15_depth.pth \
    "https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11f1p_sd15_depth.pth"

# 4. InstantID Adapters & ControlNet
RUN mkdir -p models/instantid && \
    wget -q --show-progress -O models/instantid/ip-adapter.bin \
    "https://huggingface.co/InstantX/InstantID/resolve/main/ip-adapter.bin"

RUN wget -q --show-progress -O models/controlnet/control_instant_id_sd15.bin \
    "https://huggingface.co/InstantX/InstantID/resolve/main/ControlNetModel/diffusion_pytorch_model.bin"

# 5. Antelopev2 Models for InsightFace (InstantID face analysis)
RUN mkdir -p models/insightface/models/antelopev2 && \
    wget -q --show-progress -O /tmp/antelopev2.zip \
    "https://huggingface.co/MonsterMMORPG/tools/resolve/main/antelopev2.zip" && \
    unzip -q -o /tmp/antelopev2.zip -d models/insightface/models/antelopev2 && \
    rm -f /tmp/antelopev2.zip

COPY handler.py /comfyui/handler.py
COPY workflow_api.json /comfyui/workflow_api.json

CMD ["python3", "-u", "/comfyui/handler.py"]
