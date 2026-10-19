FROM pytorch/pytorch:2.7.0-cuda12.8-cudnn9-devel
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 HF_HOME=/tmp/huggingface TOKENIZERS_PARALLELISM=false SAM2_BUILD_CUDA=0 MAX_JOBS=2
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends git ffmpeg libgl1 libglib2.0-0 ca-certificates build-essential && rm -rf /var/lib/apt/lists/*
COPY constraints.txt /app/constraints.txt
COPY requirements.txt /app/requirements.txt
RUN python -m pip install --no-cache-dir -c /app/constraints.txt -r /app/requirements.txt
RUN python -m pip install --no-cache-dir -U setuptools wheel ninja
RUN git clone --depth 1 https://github.com/Wan-Video/Wan2.2.git /app/Wan2.2
RUN git clone https://github.com/facebookresearch/sam2.git /tmp/sam2 && git -C /tmp/sam2 checkout 0e78a118995e66bb27d78518c4bd9a3e95b4e266 && cd /tmp/sam2 && python -m pip install -v --no-cache-dir --no-build-isolation -c /app/constraints.txt .
RUN python -m pip check || echo "pip check reported issues (non-fatal)"
RUN python -c "import torch,runpod,cv2,decord,onnxruntime,sam2; assert torch.__version__.split('+')[0]=='2.7.0'; assert torch.version.cuda=='12.8'; print('IMPORT_CHECK_OK')"
RUN cd /app/Wan2.2 && (python generate.py --help >/tmp/g.txt 2>&1 && echo GENERATE_OK || (tail -20 /tmp/g.txt; echo GENERATE_CHECK_FAILED))
RUN cd /app/Wan2.2 && (python wan/modules/animate/preprocess/preprocess_data.py --help >/tmp/p.txt 2>&1 && echo PREPROCESS_OK || (tail -20 /tmp/p.txt; echo PREPROCESS_CHECK_FAILED))
COPY handler.py /app/handler.py
RUN python -m py_compile /app/handler.py
CMD ["python", "-u", "/app/handler.py"]
