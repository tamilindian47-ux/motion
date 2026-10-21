FROM pytorch/pytorch:2.3.1-cuda11.8-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    ca-certificates \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt

RUN python -m pip install --upgrade "pip<25" && \
    python -m pip install -r /app/requirements.txt

# Avoid competing CPU/GPU ONNX Runtime and OpenCV distributions.
RUN python -m pip uninstall -y \
        onnxruntime \
        opencv-python-headless \
        opencv-contrib-python \
        opencv-python && \
    python -m pip install --force-reinstall --no-deps \
        onnxruntime-gpu==1.16.3 \
        opencv-python==4.7.0.72

COPY prepare.py /app/prepare.py

# Downloads source and weights into the image.
# Google Drive availability/quota can still cause this step to fail.
RUN python /app/prepare.py

COPY runner.py /app/runner.py
COPY handler.py /app/handler.py

# Build-time checks: these do not execute GPU inference.
RUN python -m py_compile /app/runner.py /app/handler.py && \
    cd /opt/rehiface && \
    python -c "import inference; assert hasattr(inference, 'GenInput'); assert hasattr(inference, 'FaceSwap')" && \
    python -m pip check && \
    python -m pip freeze > /app/BUILD_PACKAGES.txt

WORKDIR /opt/rehiface

CMD ["python", "-u", "/app/handler.py"]
