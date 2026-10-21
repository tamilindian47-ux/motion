FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    TOKENIZERS_PARALLELISM=false \
    HF_HOME=/tmp/huggingface \
    MODEL_DIR=/models/wan-turbo

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

ARG MODEL_REVISION=main

# Weights are included in the image. No network volume is needed.
RUN MODEL_REVISION="$MODEL_REVISION" python -c \
    "import os; from huggingface_hub import snapshot_download; snapshot_download(repo_id='yetter-ai/Wan2.2-TI2V-5B-Turbo-Diffusers', revision=os.environ['MODEL_REVISION'], local_dir='/models/wan-turbo', ignore_patterns=['assets/*', '*.mp4', '*.png', '*.jpg'])"

# Check Python imports during the build; no GPU inference is run.
RUN python -c \
    "from diffusers import WanImageToVideoPipeline, UniPCMultistepScheduler, AutoencoderKLWan; import runpod"

COPY handler.py .

CMD ["python", "-u", "handler.py"]
