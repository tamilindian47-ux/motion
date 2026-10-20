FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV HF_HUB_ENABLE_HF_TRANSFER=0

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    curl \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Clone MimicMotion repository
RUN git clone https://github.com/Tencent/MimicMotion.git /app/MimicMotion

COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download DWPose ONNX models & MimicMotion Checkpoint
RUN mkdir -p /app/MimicMotion/models/DWPose /app/MimicMotion/models/SVD && \
    wget -q -O /app/MimicMotion/models/DWPose/yolox_l.onnx https://huggingface.co/yzd-v/DWPose/resolve/main/yolox_l.onnx && \
    wget -q -O /app/MimicMotion/models/DWPose/dw-ll_ucoco_384.onnx https://huggingface.co/yzd-v/DWPose/resolve/main/dw-ll_ucoco_384.onnx && \
    wget -q -O /app/MimicMotion/models/MimicMotion_1-1.pth https://huggingface.co/tencent/MimicMotion/resolve/main/MimicMotion_1-1.pth

# Download SVD XT 1.1 directly via huggingface-cli snapshot without gated auth locks
RUN huggingface-cli download vdo/stable-video-diffusion-img2vid-xt-1-1 --local-dir /app/MimicMotion/models/SVD/stable-video-diffusion-img2vid-xt-1-1

COPY handler.py /app/

CMD ["python", "-u", "/app/handler.py"]
