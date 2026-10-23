# Use the official RunPod ComfyUI base worker
FROM runpod/worker-comfyui:5.3.0-base

# Install system dependencies cleanly
RUN apt-get update && apt-get install -y git-lfs wget && rm -rf /var/lib/apt/lists/*

# Set up Custom Nodes directories and clone extensions on solid single lines
WORKDIR /comfyui/custom_nodes
RUN git clone https://github.com
RUN git clone https://github.com

# Install python requirements for the custom nodes
RUN pip install --no-cache-dir -r ComfyUI-VideoHelperSuite/requirements.txt
RUN pip install --no-cache-dir -r ComfyUI-WanVideoWrapper/requirements.txt

# Download required models directly into the internal image layer
WORKDIR /comfyui/models
RUN wget -O checkpoints/wan2.1_i2v_14b_q4_k_m.safetensors "https://huggingface.co"
RUN wget -O vae/wan_2.1_vae.safetensors "https://huggingface.co"

# Hand execution control back over to RunPod's internal core script
WORKDIR /
