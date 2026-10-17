import base64
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import requests
import runpod
import torch
from huggingface_hub import snapshot_download
from PIL import Image

logging.basicConfig(level=logging.INFO, stream=sys.stdout)
LOG = logging.getLogger("wan")
ROOT = Path("/app/Wan2.2")
MODEL = Path(os.getenv("WAN_MODEL_DIR", "/tmp/wan-animate-model"))
PRE = ROOT / "wan/modules/animate/preprocess/preprocess_data.py"
LOCK = threading.Lock()
READY = False
MAX_BYTES = 5 * 1024 * 1024


def run(args, timeout=1800):
    LOG.info("Running: %s", args)
    subprocess.run([str(x) for x in args], cwd=ROOT, check=True, timeout=timeout)


def save(raw, path, limit):
    if not isinstance(raw, str) or not raw:
        raise ValueError("Expected URL or base64 input")
    if raw.startswith(("https://", "http://")):
        with requests.get(raw, stream=True, timeout=(15, 120)) as response:
            response.raise_for_status()
            total = 0
            with path.open("wb") as output:
                for chunk in response.iter_content(1024 * 1024):
                    total += len(chunk)
                    if total > limit:
                        raise ValueError("Input exceeds size limit")
                    output.write(chunk)
    else:
        raw = raw.split(",", 1)[1] if raw.startswith("data:") else raw
        if len(raw) > 4 * ((limit + 2) // 3) + 16:
            raise ValueError("Encoded input exceeds size limit")
        decoded = base64.b64decode(raw, validate=True)
        if len(decoded) > limit:
            raise ValueError("Input exceeds size limit")
        path.write_bytes(decoded)
    if path.stat().st_size == 0:
        raise ValueError("Empty input")


def check_gpu():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    LOG.info("Torch %s CUDA %s GPU %s", torch.__version__, torch.version.cuda,
             torch.cuda.get_device_name(0))
    x = torch.ones((32, 32), device="cuda")
    assert (x @ x)[0, 0].item() == 32
    torch.cuda.synchronize()
    del x
    torch.cuda.empty_cache()


def model():
    global READY
    if READY:
        return
    MODEL.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        "Wan-AI/Wan2.2-Animate-14B",
        revision=os.getenv("WAN_MODEL_REVISION", "main"),
        local_dir=str(MODEL), max_workers=4,
        ignore_patterns=["*FLUX*", "*FLUX*/**", "*.git*"],
    )
    required = ["process_checkpoint/pose2d/vitpose_h_wholebody.onnx",
                "process_checkpoint/det/yolov10m.onnx",
                "process_checkpoint/sam2/sam2_hiera_large.pt"]
    missing = [name for name in required if not (MODEL / name).is_file()]
    if missing:
        raise RuntimeError("Missing checkpoints: " + ", ".join(missing))
    READY = True


def handler(job):
    with LOCK:
        started = time.monotonic()
        data = job.get("input") or {}
        mode = data.get("mode", "replace")
        if mode not in ("animate", "replace"):
            raise ValueError("mode must be animate or replace")
        if not data.get("character_image") or not data.get("driving_video"):
            raise ValueError("character_image and driving_video are required")
        if not PRE.is_file():
            raise RuntimeError("Missing Wan2.2 source; use the matching Dockerfile")
        width, height = int(data.get("width", 832)), int(data.get("height", 480))
        if not (256 <= width <= 1280 and 256 <= height <= 1280):
            raise ValueError("Width and height must be between 256 and 1280")
        steps = int(data.get("num_inference_steps", 20))
        if not 1 <= steps <= 50:
            raise ValueError("Steps must be 1 to 50")
        check_gpu()
        with tempfile.TemporaryDirectory(prefix="wan-job-") as folder:
            work = Path(folder)
            image, video = work / "character.png", work / "driver.mp4"
            processed, output = work / "processed", work / "result.mp4"
            save(data["character_image"], image, 8 * 1024 * 1024)
            with Image.open(image) as im:
                im.convert("RGB").save(image, format="PNG")
            save(data["driving_video"], video, 64 * 1024 * 1024)
            probe = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                 "-of", "json", str(video)],
                capture_output=True, text=True, check=True, timeout=30)
            duration = float(json.loads(probe.stdout)["format"]["duration"])
            if not 0 < duration <= 5.1:
                raise ValueError("First-test clips must be at most 5 seconds")
            model()
            pre = [sys.executable, PRE, "--ckpt_path", MODEL / "process_checkpoint",
                   "--video_path", video, "--refer_path", image,
                   "--save_path", processed, "--resolution_area", width, height,
                   "--fps", "16"]
            if mode == "replace":
                pre += ["--replace_flag", "--iterations", "3", "--k", "7",
                        "--w_len", "1", "--h_len", "1"]
            else:
                pre += ["--retarget_flag"]
            run(pre)
            command = [sys.executable, ROOT / "generate.py", "--task", "animate-14B",
                       "--ckpt_dir", MODEL, "--src_root_path", processed,
                       "--refert_num", "1", "--offload_model", "True",
                       "--convert_model_dtype", "--t5_cpu", "--sample_steps", steps,
                       "--base_seed", int(data.get("seed", 42)), "--save_file", output]
            if mode == "replace":
                command += ["--replace_flag", "--use_relighting_lora"]
            run(command)
            if not output.is_file() or output.stat().st_size == 0:
                raise RuntimeError("Generation produced no video")
            if output.stat().st_size > MAX_BYTES:
                raise RuntimeError("Output exceeds 5 MiB inline limit; use a shorter/lower-resolution clip")
            return {"status": "success", "mode": mode,
                    "video_base64": base64.b64encode(output.read_bytes()).decode("ascii"),
                    "render_time_sec": round(time.monotonic() - started, 2)}


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
