# Official RunPod ComfyUI Serverless Core
FROM runpod/worker-comfyui:5.3.0-base

# Clean installation of Linux networking utilities
RUN apt-get update && apt-get install -y git-lfs wget && rm -rf /var/lib/apt/lists/*

# Force explicit working directory for video nodes
WORKDIR /comfyui/custom_nodes

# Flat commands with absolute URLs to completely avoid Git 128 truncation errors
RUN git clone https://github.com
RUN git clone https://github.com

# Execute clean PIP installs directly from the isolated directories
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
