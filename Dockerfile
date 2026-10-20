FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/Tencent/MimicMotion.git /app/MimicMotion

COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

COPY handler.py /app/

CMD ["python", "-u", "/app/handler.py"]
