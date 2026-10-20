FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    curl \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/Tencent/MimicMotion.git /app/MimicMotion

COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Download required weights during Docker build (or place them on /runpod-volume)
RUN mkdir -p /app/MimicMotion/models/DWPose /app/MimicMotion/models/SVD && \
    wget -q -O /app/MimicMotion/models/DWPose/yolox_l.onnx https://huggingface.co/yzd-v/DWPose/resolve/main/yolox_l.onnx && \
    wget -q -O /app/MimicMotion/models/DWPose/dw-ll_ucoco_384.onnx https://huggingface.co/yzd-v/DWPose/resolve/main/dw-ll_ucoco_384.onnx && \
    wget -q -O /app/MimicMotion/models/MimicMotion_1-1.pth https://huggingface.co/tencent/MimicMotion/resolve/main/MimicMotion_1-1.pth

COPY handler.py /app/

CMD ["python", "-u", "/app/handler.py"]
