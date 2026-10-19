FROM pytorch/pytorch:2.7.0-cuda12.8-cudnn9-devel
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 HF_HOME=/tmp/huggingface TOKENIZERS_PARALLELISM=false SAM2_BUILD_CUDA=0 MAX_JOBS=2
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends git ffmpeg libgl1 libglib2.0-0 ca-certificates build-essential && rm -rf /var/lib/apt/lists/*
COPY constraints.txt /app/constraints.txt
COPY requirements.txt /app/requirements.txt
RUN python -m pip install --no-cache-dir -c /app/constraints.txt -r /app/requirements.txt
RUN git clone --depth 1 https://github.com/Wan-Video/Wan2.2.git /app/Wan2.2
RUN python -m pip install --no-cache-dir --no-build-isolation -c /app/constraints.txt "git+https://github.com/facebookresearch/sam2.git@0e78a118995e66bb27d78518c4bd9a3e95b4e266"
RUN python -m pip install --no-cache-dir --no-build-isolation -c /app/constraints.txt flash-attn==2.8.3
RUN python -m pip check
RUN python -c "import torch,runpod,cv2,decord,onnxruntime,sam2,flash_attn; assert torch.__version__.split('+')[0]=='2.7.0'; assert torch.version.cuda=='12.8'"
RUN cd /app/Wan2.2 && python generate.py --help && python wan/modules/animate/preprocess/preprocess_data.py --help
COPY handler.py /app/handler.py
RUN python -m py_compile /app/handler.py
CMD ["python", "-u", "/app/handler.py"]
