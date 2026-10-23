# Use the official RunPod ComfyUI base worker (wipes out on teardown, no volumes)
FROM runpod/worker-comfyui:5.3.0-base

# Install system dependencies
RUN apt-get update && apt-get install -y git-lfs wget && rm -rf /var/lib/apt/lists/*

# Set up Custom Nodes directories
WORKDIR /comfyui/custom_nodes
RUN git clone https://github.com
RUN git clone https://github.com

# Install requirements for custom nodes
RUN pip install --no-cache-dir -r ComfyUI-VideoHelperSuite/requirements.txt
RUN pip install --no-cache-dir -r ComfyUI-WanVideoWrapper/requirements.txt

# Download required models directly into the image footprint
WORKDIR /comfyui/models

# Download Wan2.1 I2V Quantized Model (Fits inside 24GB VRAM without OOM)
RUN wget -O checkpoints/wan2.1_i2v_14b_q4_k_m.safetensors \
    "https://huggingface.co"

# Download the unified spatial-temporal VAE for Video Decoding
RUN wget -O vae/wan_2.1_vae.safetensors \
    "https://huggingface.co"

# Return to root for RunPod's internal boot script
WORKDIR /
