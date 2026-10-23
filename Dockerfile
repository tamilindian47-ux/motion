# Use the official RunPod ComfyUI image which already has PyTorch + CUDA pre-installed
FROM runpod/worker-comfyui:5.3.0-base

# Set up system variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install basic system dependencies safely
RUN apt-get update && apt-get install -y \
    ffmpeg libgl1-mesa-glx wget unzip \
    && rm -rf /var/lib/apt/lists/*

# Install remaining light-weight packages needed by handler.py
RUN pip3 install --no-cache-dir runpod requests>=2.31.0 einops av

# Set up runtime asset directories inside the pre-existing ComfyUI folder footprint
RUN mkdir -p /comfyui/models/checkpoints /comfyui/models/vae

# Copy your production application handler into the image container root
COPY handler.py /handler.py

# Execute the custom python wrapper as the direct system listener
CMD ["python3", "-u", "/handler.py"]
