FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Set up system variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install basic system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip python3-dev git ffmpeg libgl1-mesa-glx wget unzip \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip3 install --no-cache-dir --upgrade pip

# Install PyTorch with CUDA support
RUN pip3 install --no-cache-dir torch torchvision torchaudio --index-url https://pytorch.org

# Install ComfyUI core requirements and RunPod API wrapper
RUN pip3 install --no-cache-dir runpod requests>=2.31.0 transformers>=4.44.0 accelerate>=0.33.0 sentencepiece einops av

# Set up project workspace directories
WORKDIR /
RUN git clone https://github.com

# Set up runtime asset directories
RUN mkdir -p /ComfyUI/models/checkpoints /ComfyUI/models/vae

# Copy your production application files into the image container root
COPY handler.py /handler.py

# Execute the custom python wrapper as the direct system listener
CMD ["python3", "-u", "/handler.py"]
