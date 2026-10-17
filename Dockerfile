FROM pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/tmp/huggingface \
    TOKENIZERS_PARALLELISM=false

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt

# Prevent dependency resolution from replacing the CUDA-enabled
# PyTorch installation supplied by the base image.
RUN printf '%s\n' \
    'torch==2.7.0' \
    'torchvision==0.22.0' \
    'torchaudio==2.7.0' \
    > /app/torch-constraints.txt \
    && python -m pip install --no-cache-dir \
        --constraint /app/torch-constraints.txt \
        --requirement /app/requirements.txt

RUN python -m pip check

# This runs during image build, without requiring a GPU.
# Fail the build rather than deploying incompatible dependencies.
RUN python - <<'PY'
import importlib.metadata
import torch
from transformers import CLIPVisionModel
from diffusers import AutoencoderKLWan, WanImageToVideoPipeline

actual_torch = importlib.metadata.version("torch")
print("Installed torch distribution:", actual_torch)
print("Torch runtime version:", torch.__version__)
print("Bundled CUDA:", torch.version.cuda)

assert actual_torch.split("+")[0] == "2.7.0", actual_torch
assert torch.version.cuda == "12.8", torch.version.cuda
assert hasattr(torch, "accelerator"), "Missing torch.accelerator"

print("BUILD IMPORT CHECK PASSED")
PY

COPY handler.py /app/handler.py

# Checks Python syntax without starting the serverless worker.
RUN python -m py_compile /app/handler.py

# Test real GPU kernel execution before starting the handler.
# Python stdout is unbuffered so diagnostics appear in worker logs.
CMD ["bash", "-c", "python -u - <<'PY'\nimport importlib.metadata\nimport torch\n\nactual = importlib.metadata.version('torch')\nprint('STARTUP: torch distribution =', actual, flush=True)\nprint('STARTUP: bundled CUDA =', torch.version.cuda, flush=True)\nassert actual.split('+')[0] == '2.7.0', actual\nassert torch.cuda.is_available(), 'CUDA is unavailable'\nprint('STARTUP: GPU =', torch.cuda.get_device_name(0), flush=True)\nprint('STARTUP: capability =', torch.cuda.get_device_capability(0), flush=True)\nprint('STARTUP: compiled architectures =', torch.cuda.get_arch_list(), flush=True)\nx = torch.ones((32, 32), device='cuda')\ny = x @ x\nassert y[0, 0].item() == 32.0\ntorch.cuda.synchronize()\nprint('STARTUP: CUDA KERNEL TEST PASSED', flush=True)\nPY\nresult=$?\nif [ \"$result\" -ne 0 ]; then exit \"$result\"; fi\nexec python -u /app/handler.py"]
