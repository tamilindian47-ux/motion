FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    MAX_JOBS=2 \
    TOKENIZERS_PARALLELISM=false

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git \
        ffmpeg \
        build-essential && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

ARG WAN_REF=main

RUN git clone https://github.com/Wan-Video/Wan2.2.git /app/Wan2.2 && \
    cd /app/Wan2.2 && \
    git checkout "$WAN_REF" && \
    git rev-parse HEAD > /app/WAN_COMMIT

RUN pip install --no-cache-dir \
    packaging \
    ninja \
    wheel \
    setuptools

# Upstream requirements include FlashAttention.
# Compilation can take substantial time and build-machine RAM.
RUN cd /app/Wan2.2 && \
    pip install --no-cache-dir \
        --no-build-isolation \
        -r requirements.txt

COPY requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir -r /app/requirements.txt

ARG MODEL_REVISION=main

RUN MODEL_REVISION="$MODEL_REVISION" python -c \
    "import os; from huggingface_hub import snapshot_download; snapshot_download('Wan-AI/Wan2.2-TI2V-5B', revision=os.environ['MODEL_REVISION'], local_dir='/models/Wan2.2-TI2V-5B')"

COPY handler.py /app/handler.py

CMD ["python", "-u", "/app/handler.py"]
