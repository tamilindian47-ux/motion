FROM pytorch/pytorch:2.3.1-cuda11.8-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

# ------------------------------------------------------------
# System dependencies
# ------------------------------------------------------------

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ffmpeg \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
        ca-certificates \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ------------------------------------------------------------
# Python dependencies
# ------------------------------------------------------------

COPY requirements.txt /app/requirements.txt

RUN python -m pip install --upgrade "pip<25"

RUN python -m pip install -r /app/requirements.txt

# Remove conflicting OpenCV distributions and CPU ONNX Runtime.
# Reinstall the selected wheels because uninstalling overlapping
# distributions can remove shared package files.
RUN python -m pip uninstall -y \
        onnxruntime \
        opencv-python-headless \
        opencv-contrib-python \
        opencv-python

RUN python -m pip install \
        --force-reinstall \
        --no-deps \
        onnxruntime-gpu==1.16.3 \
        opencv-python==4.7.0.72

# ------------------------------------------------------------
# Download pinned ReHiFace-S source and model files
# ------------------------------------------------------------

COPY prepare.py /app/prepare.py

RUN python -u /app/prepare.py

# ------------------------------------------------------------
# Worker files
# ------------------------------------------------------------

COPY runner.py /app/runner.py
COPY handler.py /app/handler.py

# ------------------------------------------------------------
# Separate validation steps.
# Each check gets its own Docker build step and error output.
# ------------------------------------------------------------

RUN echo "CHECK 1: Python worker syntax" && \
    python -m py_compile \
        /app/runner.py \
        /app/handler.py

RUN echo "CHECK 2: Installed Python and package versions" && \
    python --version && \
    python -m pip --version && \
    python -m pip list --format=freeze

RUN echo "CHECK 3: Import basic numerical and GPU libraries" && \
    python -u -c "\
import sys; \
print('Python:', sys.version, flush=True); \
import numpy; \
print('NumPy:', numpy.__version__, flush=True); \
import cv2; \
print('OpenCV:', cv2.__version__, flush=True); \
import torch; \
print('PyTorch:', torch.__version__, flush=True); \
print('PyTorch compiled CUDA:', torch.version.cuda, flush=True); \
import onnxruntime; \
print('ONNX Runtime:', onnxruntime.__version__, flush=True); \
print('Available ORT providers:', onnxruntime.get_available_providers(), flush=True)"

WORKDIR /opt/rehiface

RUN echo "CHECK 4: Required source files exist" && \
    python -u -c "\
from pathlib import Path; \
required = [ \
    Path('inference.py'), \
    Path('options/hifi_test_options.py'), \
    Path('pretrain_models/9O_865k.onnx'), \
    Path('pretrain_models/CurricularFace.tjm'), \
    Path('pretrain_models/gfpganv14_fp32_bs1_scale.onnx'), \
    Path('pretrain_models/pfpld_robust_sim_bs1_8003.onnx'), \
    Path('pretrain_models/scrfd_500m_bnkps_shape640x640.onnx'), \
    Path('pretrain_models/xseg_230611_16_17.onnx') \
]; \
missing = [str(path) for path in required if not path.is_file()]; \
print('Missing files:', missing, flush=True); \
assert not missing, 'Required source/model files are missing'"

RUN echo "CHECK 5: Import upstream inference module" && \
    python -u -c "\
print('Importing inference...', flush=True); \
import inference; \
print('Imported from:', inference.__file__, flush=True); \
assert hasattr(inference, 'GenInput'), 'inference.GenInput is missing'; \
assert hasattr(inference, 'FaceSwap'), 'inference.FaceSwap is missing'; \
print('Upstream inference import passed.', flush=True)"

RUN echo "CHECK 6: Import upstream options" && \
    python -u -c "\
from options.hifi_test_options import HifiTestOptions; \
print('HifiTestOptions import passed.', flush=True)"

RUN echo "CHECK 7: Validate installed dependency requirements" && \
    python -m pip check

RUN python -m pip freeze > /app/BUILD_PACKAGES.txt

# ------------------------------------------------------------
# Start RunPod worker
# ------------------------------------------------------------

CMD ["python", "-u", "/app/handler.py"]
