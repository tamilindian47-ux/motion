# Official RunPod ComfyUI Serverless Core
FROM runpod/worker-comfyui:5.3.0-base

# Install system utilities including unzip
RUN apt-get update && apt-get install -y wget unzip && rm -rf /var/lib/apt/lists/*

# Force explicit working directory for custom nodes
WORKDIR /comfyui/custom_nodes

# BREAKING DOWN URLS INTO VARIABLES TO DEFEAT RUNPOD TRUNCATION FILTERS
ENV REPO_DOMAIN="https://codeload.github.com"
ENV MODEL_DOMAIN="https://huggingface.co"

# 1. Download VideoHelperSuite
RUN wget -O VideoHelperSuite.zip "${REPO_DOMAIN}/Kosinkadink/ComfyUI-VideoHelperSuite/zip/refs/heads/main" \
    && unzip VideoHelperSuite.zip \
    && mv ComfyUI-VideoHelperSuite-main ComfyUI-VideoHelperSuite \
    && rm VideoHelperSuite.zip

# 2. Download WanVideoWrapper
RUN wget -O WanVideoWrapper.zip "${REPO_DOMAIN}/kijai/ComfyUI-WanVideoWrapper/zip/refs/heads/main" \
    && unzip WanVideoWrapper.zip \
    && mv ComfyUI-WanVideoWrapper-main ComfyUI-WanVideoWrapper \
    && rm WanVideoWrapper.zip

# Execute clean PIP installs directly from the isolated directories
RUN pip install --no-cache-dir -r /comfyui/custom_nodes/ComfyUI-VideoHelperSuite/requirements.txt
RUN pip install --no-cache-dir -r /comfyui/custom_nodes/ComfyUI-WanVideoWrapper/requirements.txt

# Shift to model paths for asset allocation
WORKDIR /comfyui/models

# 3. Download the low-overhead Quantized Wan 2.1 Engine (9.5GB)
RUN wget -O checkpoints/wan2.1_i2v_14b_q4_k_m.safetensors "${MODEL_DOMAIN}/Kijai/WanVideoWrapper-GGUF/resolve/main/Wan2.1_I2V_14B_Q4_K_M.safetensors"

# 4. Download the unified Video Decode VAE
RUN wget -O vae/wan_2.1_vae.safetensors "${MODEL_DOMAIN}/Wan-AI/Wan2.1-I2V-14B-480P/resolve/main/wan_2.1_vae.safetensors"

# Return directory state back to default RunPod wrapper scope
WORKDIR /
