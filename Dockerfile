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

# Remove conflicting distributions, then restore selected wheels.
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
# Build validation
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

RUN echo "CHECK 7: Upgrade typing-extensions and validate dependencies" && \
    python -m pip install --upgrade "typing-extensions>=4.16.0,<5" && \
    python -m pip check

# ------------------------------------------------------------
# Supply CUDA 11.8 NVRTC and its unversioned library link
# ------------------------------------------------------------

RUN python -m pip install --no-deps \
        nvidia-cuda-nvrtc-cu11==11.8.89

RUN python - <<'PY'
from pathlib import Path
import site

candidates = [
    Path(root) / "nvidia" / "cuda_nvrtc" / "lib"
    for root in site.getsitepackages()
]

libdir = next(
    (
        path for path in candidates
        if path.is_dir() and list(path.glob("libnvrtc.so.*"))
    ),
    None,
)

if libdir is None:
    raise RuntimeError("CUDA 11 NVRTC library directory was not found")

target = sorted(libdir.glob("libnvrtc.so.*"))[0]
link = libdir / "libnvrtc.so"

if not link.exists():
    if link.is_symlink():
        link.unlink()
    link.symlink_to(target.name)

Path("/etc/ld.so.conf.d/rehiface-nvrtc.conf").write_text(
    str(libdir) + "\n"
)

print("NVRTC directory:", libdir, flush=True)
print("NVRTC link:", link, "->", link.resolve(), flush=True)
PY

RUN ldconfig

# ------------------------------------------------------------
# Register existing CUDA/cuDNN library directories
# ------------------------------------------------------------

RUN python - <<'PY'
from pathlib import Path

root = Path("/opt/conda")

patterns = [
    "libcudnn*.so*",
    "libcublas*.so*",
    "libcudart.so*",
    "libnvrtc.so*",
]

directories = set()
cudnn_infer = []

for pattern in patterns:
    for library in root.rglob(pattern):
        if library.is_file():
            directories.add(str(library.parent))

            if library.name.startswith("libcudnn_cnn_infer.so.8"):
                cudnn_infer.append(str(library))

print("cuDNN CNN inference libraries found:", flush=True)

for library in sorted(cudnn_infer):
    print(library, flush=True)

if not cudnn_infer:
    raise RuntimeError(
        "libcudnn_cnn_infer.so.8 is absent from /opt/conda. "
        "Library-path registration cannot fix an absent library."
    )

config = Path("/etc/ld.so.conf.d/rehiface-conda-cuda.conf")
config.write_text("\n".join(sorted(directories)) + "\n")

print("Registered library directories:", flush=True)
print(config.read_text(), flush=True)
PY

RUN ldconfig

# ------------------------------------------------------------
# Verify standalone library loading
# ------------------------------------------------------------

RUN echo "CHECK 8: Load NVRTC and cuDNN inference libraries" && \
    python -u -c "\
import ctypes; \
ctypes.CDLL('libnvrtc.so'); \
print('NVRTC load passed', flush=True); \
ctypes.CDLL('libcudnn_cnn_infer.so.8'); \
print('cuDNN CNN inference library load passed', flush=True)"

RUN python -m pip freeze > /app/BUILD_PACKAGES.txt

# ------------------------------------------------------------
# Start RunPod worker
# ------------------------------------------------------------

CMD ["python", "-u", "/app/handler.py"]
