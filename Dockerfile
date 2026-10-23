# Official RunPod ComfyUI Serverless Core
FROM runpod/worker-comfyui:5.3.0-base

# Install system utilities including unzip
RUN apt-get update && apt-get install -y wget unzip && rm -rf /var/lib/apt/lists/*

# Force explicit working directory for custom nodes
WORKDIR /comfyui/custom_nodes

# 1. Download VideoHelperSuite as a zip file and extract it
RUN wget -O VideoHelperSuite.zip "https://github.com" \
    && unzip VideoHelperSuite.zip \
    && mv ComfyUI-VideoHelperSuite-main ComfyUI-VideoHelperSuite \
    && rm VideoHelperSuite.zip

# 2. Download WanVideoWrapper as a zip file and extract it
RUN wget -O WanVideoWrapper.zip "https://github.com" \
    && unzip WanVideoWrapper.zip \
    && mv ComfyUI-WanVideoWrapper-main ComfyUI-WanVideoWrapper \
    && rm WanVideoWrapper.zip

# 3. Execute clean PIP installs directly from the isolated directories
RUN pip install --no-cache-dir -r /comfyui/custom_nodes/ComfyUI-VideoHelperSuite/requirements.txt
RUN pip install --no-cache-dir -r /comfyui/custom_nodes/ComfyUI-WanVideoWrapper/requirements.txt

# Shift to model paths for asset allocation
WORKDIR /comfyui/models

# Download the low-overhead Quantized Wan 2.1 Engine (9.5GB)
RUN wget -O checkpoints/wan2.1_i2v_14b_q4_k_m.safetensors "https://huggingface.co"

# Download the unified Video Decode VAE
RUN wget -O vae/wan_2.1_vae.safetensors "https://huggingface.co"

# Return directory state back to default RunPod wrapper scope
WORKDIR /
